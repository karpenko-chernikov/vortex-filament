from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,signal,subprocess,sys,threading,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'series_CD.json').read_text(encoding='utf-8'))
BASE=json.loads((ROOT/'base_config.json').read_text(encoding='utf-8'))
PILOT=12.0
FINAL=60.0
GRIDS=(64,80,96)
ACTIVE=set()
ACTIVE_LOCK=threading.Lock()
BOARD=None

RST='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
WHITE='\033[97m'
GRAY='\033[90m'
BLUE='\033[34m'


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def source_files():
    return [ROOT/p for p in ('run_manager.py','base_config.json','series_CD.json','campaign_cd.py','solver_mesh_check.py','requirements.txt')]+list((ROOT/'solver').glob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'inputs').glob('*'))+list((ROOT/'references_A').glob('*'))


def digest():
    h=hashlib.sha256()
    for p in sorted(source_files()):
        h.update(str(p.relative_to(ROOT)).encode('utf-8'));h.update(p.read_bytes())
    return h.hexdigest()


def scenario_dir(participant,item):
    return ROOT/f'results_{participant}'/item['name']


def prepare(participant,item,target):
    dest=scenario_dir(participant,item)
    dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'RUN_FINGERPRINT.json'
    provenance={'scenario':item,'source_sha256':digest(),'participant':participant}
    if marker.exists():
        old=read_json(marker)
        if old!=provenance:raise RuntimeError('Настройки или исходный код изменились; продолжение несовместимо: '+str(dest))
    else:
        if (dest/'results_v42'/'cases').exists():
            raise RuntimeError('Найдены контрольные точки без совпадающего манифеста: '+str(dest))
        write_json(marker,provenance)
    for sub in ('solver','tools','inputs'):
        (dest/sub).mkdir(exist_ok=True)
        for src in (ROOT/sub).iterdir():
            if src.is_file() and src.suffix in ('.py','.csv','.json'):
                shutil.copy2(src,dest/sub/src.name)
    shutil.copy2(ROOT/'run_manager.py',dest/'run_manager.py')
    conf=json.loads(json.dumps(BASE))
    conf['experiment_label']=item['name']
    conf['physical_time_s']=target
    conf['parameters']['H_water']=item['H_water_m']
    conf['parameters']['coupling_rate_s']=item['coupling_rate_s']
    conf['grids']=[{'name':f'grid_{nr}x{nr*2}','Nr':nr,'Nz':nr*2,'dt':item['dt_us']*1e-6} for nr in GRIDS]
    write_json(dest/'config.json',conf)
    return dest


def console(message):
    board=BOARD
    if board is not None:
        board.emit(message)
    else:
        print(message, flush=True)


def show_child_line(line):
    text=line.strip()
    if not text or text.startswith('[ПРОГРЕСС]'):
        return False
    if os.environ.get('VOF_DASHBOARD')!='1':
        return True
    markers=('[ENV]','[RUN]','[OK]','[FAILED]','[WARNING]','[ANALYZE]','[PACKAGE]',
             '[CASE]','[STAGE','[REFERENCE]','=====','STATUS:','ARCHIVE:','Error','Traceback',
             'BOTTOM','SATELLITE','FloatingPoint','NaN','Прервано')
    return any(marker in text for marker in markers)


def stop_process(proc):
    if proc.poll() is not None:
        return
    if os.name=='nt':
        subprocess.run(['taskkill','/F','/T','/PID',str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except OSError:
        try: proc.terminate()
        except OSError: pass


def run_command(command,cwd,logpath):
    logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('a',encoding='utf-8') as out:
        out.write('\n--- '+time.strftime('%Y-%m-%d %H:%M:%S')+' --- '+' '.join(command)+'\n');out.flush()
        env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
        proc=subprocess.Popen(command,cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              env=env,text=True,encoding='utf-8',errors='replace',bufsize=1,
                              start_new_session=True)
        with ACTIVE_LOCK:
            ACTIVE.add(proc)
        try:
            for line in proc.stdout:
                out.write(line);out.flush()
                if show_child_line(line):
                    console(line.rstrip())
            return proc.wait()
        except KeyboardInterrupt:
            stop_process(proc)
            try: proc.wait(timeout=12)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
            raise
        finally:
            with ACTIVE_LOCK:
                ACTIVE.discard(proc)


def statuses(directory):
    d=read_json(directory/'results_v42'/'STATUS.json')
    grids=d.get('cases',[])
    target=read_json(directory/'config.json').get('physical_time_s',0)
    complete=len(grids)==3 and all(x.get('execution')=='COMPLETE' and
        x.get('physical_time_reached_s',0)>=target-1e-9 for x in grids)
    qc=complete and all(x.get('numerical_qc')=='PASS' for x in grids)
    return {'case':directory.name,'complete':bool(complete),'numerical_qc':bool(qc),
            'grids':grids,'reached_time_s':min([float(x.get('physical_time_reached_s',0)) for x in grids],default=0),
            'spatial_convergence':d.get('spatial_convergence','NOT_EVALUATED')}


def run_stage(participant,item,target,artifacts=False):
    dest=prepare(participant,item,target)
    console(f'[RUN] {item["name"]}: 3 сетки, dt={item["dt_us"]} мкс, H={item["H_water_m"]*1e3:.0f} мм, t={target:g} с')
    cmd=[sys.executable,'-u','run_manager.py','--mode','full']
    if not artifacts:cmd.append('--no-artifacts')
    code=run_command(cmd,dest,dest/'CAMPAIGN_LOG.txt')
    s=statuses(dest);s['return_code']=code
    mark='OK' if s['complete'] and s['numerical_qc'] else 'WARNING'
    console(f'[{mark}] {item["name"]}: {s["reached_time_s"]:.2f}/{target:.2f} с, QC={s["numerical_qc"]}')
    return s


def run_analysis(participant,target,stage):
    cmd=[sys.executable,'-u',str(ROOT/'tools'/'convergence_cd.py'),
         '--participant',participant,'--time',str(target),'--stage',stage]
    out=ROOT/f'results_{participant}'/'CD_ANALYSIS'/'analysis_log.txt'
    return run_command(cmd,ROOT,out)


def pack(participant,summary):
    directory=ROOT/f'results_{participant}'
    directory.mkdir(exist_ok=True)
    final=directory/'Send_to_AZ.zip'
    pending=directory/'Send_to_AZ.pending.zip'
    write_json(directory/'SUMMARY_CD.json',summary)
    with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p in source_files()+[ROOT/'README.md']:
            z.write(p,arcname='SOURCE/'+str(p.relative_to(ROOT)))
        for p in directory.rglob('*'):
            if not p.is_file() or p==pending or p==final or p.suffix=='.npz':continue
            if p.suffix=='.zip' or p.name=='PROGRESS.log' or p.name.startswith('SEND_TO_CHATGPT_v42'):continue
            if p.suffix.lower() in ('.json','.csv','.md','.txt','.log','.png','.gif','.mp4'):
                z.write(p,arcname=str(p.relative_to(directory)))
        for item in PLAN[participant]:
            series=scenario_dir(participant,item)
            for g in (series/'results_v42'/'cases').glob('*') if (series/'results_v42'/'cases').exists() else []:
                state=[]
                for p in g.glob('segment_*/snapshot_*.npz'):
                    try:
                        import numpy as np
                        with np.load(p) as a:t=float(a['t'])
                        state.append((t,p))
                    except Exception:pass
                state.sort(key=lambda x:x[0])
                if state:
                    targets=(0.0,3.0,6.0,9.0,PILOT,30.0,45.0,FINAL)
                    chosen={min(state,key=lambda x:abs(x[0]-tt))[1] for tt in targets
                            if max(x[0] for x in state)>=tt-.15}
                    chosen.add(state[-1][1]); chosen.add(state[0][1])
                    for p in sorted(chosen):
                        z.write(p,arcname=str(p.relative_to(directory)))
                for p in g.glob('segment_*/FAILED_*.npz'):
                    z.write(p,arcname=str(p.relative_to(directory)))
    pending.replace(final)
    console(f'[PACKAGE] {final} ({final.stat().st_size/1e6:.2f} MB)')
    return final


def preflight(participant):
    info=PLAN[participant][0]
    from solver_mesh_check import check_mesh_stability
    check_mesh_stability(info['H_water_m'],grid_sizes=GRIDS,dt_us_values=(20,10,5))
    dest=ROOT/f'results_{participant}'/'_preflight'
    conf=json.loads(json.dumps(BASE))
    conf['experiment_label']='preflight_'+participant
    conf['physical_time_s']=.001
    conf['parameters']['H_water']=info['H_water_m']
    conf['parameters']['coupling_rate_s']=info['coupling_rate_s']
    conf['grids']=[{'name':'test','Nr':48,'Nz':96,'dt':1e-5}]
    dest.mkdir(parents=True,exist_ok=True)
    for sub in ('solver','tools','inputs'):
        (dest/sub).mkdir(exist_ok=True)
        for f in (ROOT/sub).glob('*'):
            if f.is_file() and f.suffix in ('.py','.csv','.json'):
                shutil.copy2(f,dest/sub/f.name)
    shutil.copy2(ROOT/'run_manager.py',dest/'run_manager.py')
    write_json(dest/'config.json',conf)
    code=run_command([sys.executable,'-u','run_manager.py','--mode','smoke'],dest,dest/'PREFLIGHT_LOG.txt')
    if code:raise RuntimeError(f'CFD preflight failed, rc={code}')
    print('[OK] Короткий предтест прошёл: шаг, запись и проверка баланса.',flush=True)


def format_duration(seconds):
    if seconds is None or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
        return 'ещё нет оценки'
    seconds=int(seconds)
    days,seconds=divmod(seconds,86400)
    hours,seconds=divmod(seconds,3600)
    minutes,seconds=divmod(seconds,60)
    if days: return f'{days}д {hours}ч {minutes:02d}м'
    if hours: return f'{hours}ч {minutes:02d}м'
    if minutes: return f'{minutes}м {seconds:02d}с'
    return f'{seconds}с'


def color_bar(fraction, width=18):
    fraction=0.0 if not isinstance(fraction,(int,float)) or not math.isfinite(fraction) else fraction
    fraction=0.0 if fraction<0 else 1.0 if fraction>1 else fraction
    filled=int(round(fraction*width))
    if 0<fraction<1 and filled==0: filled=1
    if fraction<1 and filled==width: filled=width-1
    return f'{GREEN}{"█"*filled}{GRAY}{"░"*(width-filled)}{RST}'


def pid_alive(pid):
    try: os.kill(int(pid), 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    except (OSError, TypeError, ValueError): return False
    return True


def set_campaign_phase(participant, text):
    write_json(ROOT/f'results_{participant}'/'CAMPAIGN_PHASE.json', {'phase':text,'updated_unix':time.time()})


class LiveBoard:
    def __init__(self, participant):
        self.participant=participant
        self.lines=0
        self.lock=threading.Lock()
        self.stop_event=threading.Event()
        self.thread=None
        self.last_log=0.0

    def start(self):
        if sys.stdout.isatty():
            sys.stdout.write('\033[?25l')
            sys.stdout.flush()
        self.paint()
        self.thread=threading.Thread(target=self._loop, name='vof-board', daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.wait(2):
            try: self.paint()
            except Exception as exc: self.emit(f'[WARNING] сводка: {exc}')

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)
        try: self.paint(final=True)
        except Exception: pass
        if sys.stdout.isatty():
            sys.stdout.write('\033[?25h')
            sys.stdout.flush()

    def emit(self, message):
        with self.lock:
            if self.lines and sys.stdout.isatty():
                sys.stdout.write(f'\033[{self.lines}A')
                for _ in range(self.lines):
                    sys.stdout.write('\033[2K\n')
                sys.stdout.write(f'\033[{self.lines}A')
                self.lines=0
            print(message, flush=True)

    def paint(self, final=False):
        text=self.compose()
        with self.lock:
            tty=sys.stdout.isatty() and not final
            if not tty:
                if self.lines and sys.stdout.isatty():
                    sys.stdout.write(f'\033[{self.lines}A')
                    for _ in range(self.lines):
                        sys.stdout.write('\033[2K\n')
                    sys.stdout.write(f'\033[{self.lines}A')
                    self.lines=0
                for line in text:
                    print(line, flush=True)
                self.lines=0
            else:
                if self.lines:
                    sys.stdout.write(f'\033[{self.lines}A')
                for line in text:
                    sys.stdout.write('\033[2K'+line+'\n')
                extra=self.lines-len(text)
                for _ in range(max(0, extra)):
                    sys.stdout.write('\033[2K\n')
                if extra>0:
                    sys.stdout.write(f'\033[{extra}A')
                self.lines=len(text)
                sys.stdout.flush()
        now=time.time()
        if final or now-self.last_log>=60:
            try:
                log=ROOT/f'results_{self.participant}'/'PROGRESS.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a', encoding='utf-8') as handle:
                    handle.write(time.strftime('%H:%M:%S ')+' '.join(
                        line.replace('\033[0m','').replace('\033[1m','').replace('\033[2m','')
                        for line in text)+'\n')
            except Exception:
                pass
            self.last_log=now

    def compose(self):
        participant=self.participant
        root=ROOT/f'results_{participant}'
        found={}
        if root.exists():
            for path in root.glob('*/results_v42/cases/*/live_progress.json'):
                data=read_json(path)
                if not data: continue
                found[(data.get('series') or path.parents[3].name, data.get('case') or path.parent.name)]=data
        done=0
        total_steps=0
        running=0
        waiting=0
        rows=[]
        jobs=[]
        for item in PLAN[participant]:
            for nr in GRIDS:
                case=f'grid_{nr}x{nr*2}'
                jobs.append((item, case, nr))
                total_steps += math.ceil(PILOT/(item['dt_us']*1e-6))
        for item, case, nr in jobs:
            data=found.get((item['name'], case))
            planned=math.ceil(PILOT/(item['dt_us']*1e-6))
            if not data:
                rows.append(f'  {GRAY}{item["name"][:3]}  {item["dt_us"]:>2} мкс  {nr}×{nr*2:<7}  ещё не начата{RST}')
                continue
            nsteps=max(1, int(data.get('nsteps') or planned))
            step=min(nsteps, max(0, int(data.get('step') or 0)))
            phase=str(data.get('phase') or '')
            if phase.startswith('готово'):
                step=nsteps
            fraction=step/nsteps
            done += step
            total_steps += nsteps-planned
            age=max(0.0, time.time()-float(data.get('updated_unix') or time.time()))
            alive=pid_alive(data.get('pid'))
            if 'ждёт' in phase:
                waiting += 1
                color=YELLOW
                state='ждёт ядро'
            elif phase.startswith('ошибка') or phase.startswith('прервано'):
                color=RED
                state='останов'
            elif phase.startswith('готово'):
                color=GREEN
                state='готово'
            elif not alive:
                color=RED
                state='процесс пропал'
            elif age>75:
                color=RED
                state=f'нет пульса {int(age)} с'
            else:
                running += 1
                color=WHITE
                state=phase or 'счёт'
            t_s=float(data.get('t_s') or 0)
            t_end=float(data.get('t_end_s') or PILOT)
            eta=format_duration(data.get('eta_s'))
            rows.append(
                f'  {color}{item["name"][:3]}  {item["dt_us"]:>2} мкс  {nr}×{nr*2:<7}{RST} '
                f'{color_bar(fraction)} {BOLD}{fraction*100:5.1f}%{RST}  '
                f't={t_s:5.2f}/{t_end:4.0f}с  {DIM}{state}{RST}  осталось {eta}')
        overall=done/total_steps if total_steps else 0
        jobs_left=sum(1 for item, case, nr in jobs if not str((found.get((item['name'], case)) or {}).get('phase','')).startswith(('готово','ошибка','прервано')))
        stepping=[]
        for data in found.values():
            phase=str(data.get('phase') or '')
            rate=data.get('steps_per_s')
            if phase.startswith(('счёт','компил','снимок')) and isinstance(rate,(int,float)) and rate>0:
                stepping.append(float(rate))
        try: workers=max(1, int(os.environ.get('VOF_WORKERS','1')))
        except ValueError: workers=1
        if stepping and jobs_left:
            throughput=(sum(stepping)/len(stepping))*min(workers, jobs_left)
            eta=format_duration((total_steps-done)/throughput)
        elif jobs_left==0 and total_steps:
            eta='0с'
        else:
            eta='уточняется'
        phase=read_json(root/'CAMPAIGN_PHASE.json').get('phase') or 'пилот до 12 с'
        clock=time.strftime('%H:%M:%S')
        total=len(jobs)
        lines=[
            f'{BOLD}{CYAN}VOF  очередь {participant}{RST}   {WHITE}{clock}{RST}',
            f'{BOLD}Считается сейчас: {GREEN}{running}{RST}{BOLD}    Всего задач: {WHITE}{total}{RST}{BOLD}    Ждут ядро: {YELLOW}{waiting}{RST}',
            f'{DIM}Этап:{RST} {BLUE}{phase}{RST}    {DIM}осталось примерно{RST} {BOLD}{eta}{RST}',
            f'Общий {color_bar(overall, 28)} {BOLD}{overall*100:5.1f}%{RST}',
        ]
        lines.extend(rows)
        lines.append(f'{DIM}Строка обновляется на месте. Пульс есть, даже если процент секунду стоит.{RST}')
        return lines


def configure(test):
    os.environ.setdefault('VOF_SLOT_DIR', str(ROOT/'.vof_slots'))
    if test:
        os.environ['VOF_DASHBOARD']='0'
        os.environ.setdefault('VOF_WORKERS','1')
    else:
        os.environ.setdefault('VOF_WORKERS', str(os.cpu_count() or 1))
        os.environ['VOF_DASHBOARD']='0'


def late_reference(participant, ref_item):
    ref_dest=prepare(participant, ref_item, FINAL)
    console('[REFERENCE] Поздняя проверка: dt=5 мкс, сетка 64×128, до 60 с')
    ref_rc=run_command([sys.executable,'-u','run_manager.py','--mode','full','--no-artifacts','--grid-only','grid_64x128'],
                       ref_dest, ref_dest/'LATE_TEMPORAL_LOG.txt')
    return {'dt_us':5,'grid':'64x128','return_code':ref_rc}


def main():
    global BOARD
    ap=argparse.ArgumentParser()
    ap.add_argument('--participant',choices=['C','D'],required=True)
    ap.add_argument('--test',action='store_true')
    ap.add_argument('--pack-only',action='store_true')
    args=ap.parse_args()
    participant=args.participant
    configure(args.test or args.pack_only)
    root=ROOT/f'results_{participant}'
    root.mkdir(exist_ok=True)
    summary={'participant':participant,'physical_scenario':{'H_water_m':PLAN[participant][0]['H_water_m'],
            'coupling_rate_s':PLAN[participant][0]['coupling_rate_s']},
            'pilot_time_s':PILOT,'final_time_s':FINAL,'stage1':[],
            'selected_dt_us':None,'stage2':None,
            'model_validation':'NOT_ESTABLISHED','status':'NOT_STARTED'}
    board=None
    try:
        if args.test:
            preflight(participant)
            return 0
        if args.pack_only:
            prev=read_json(root/'SUMMARY_CD.json')
            if prev: summary=prev
        else:
            print('Сначала короткая проверка, что расчёт вообще стартует.', flush=True)
            preflight(participant)
            os.environ['VOF_DASHBOARD']='1'
            board=LiveBoard(participant)
            BOARD=board
            set_campaign_phase(participant, 'пилот: все 9 задач до 12 с')
            board.start()
            console(f'[СТАРТ] Очередь {participant}. Одновременно до {os.environ.get("VOF_WORKERS")} процессов. Схема и шаг не менялись.')
            def one(item):
                try: return run_stage(participant, item, PILOT, artifacts=False)
                except Exception as exc:
                    console(f'[FAILED] {item["name"]} {exc!r}')
                    return {'case':item['name'],'failed':repr(exc)}
            with ThreadPoolExecutor(max_workers=len(PLAN[participant])) as pool:
                futures=[pool.submit(one, item) for item in PLAN[participant]]
                summary['stage1']=[fut.result() for fut in futures]
            set_campaign_phase(participant, 'сравнение шагов 20 / 10 / 5 мкс')
            if run_analysis(participant, PILOT, 'pilot'):
                console('[WARNING] Разбор пилота не закончился')
            decision=read_json(root/'CD_ANALYSIS'/'PILOT_DECISION.json')
            selected=decision.get('selected_dt_us')
            summary['selected_dt_us']=selected
            summary['dt_selection_reason']=decision.get('reason')
            item=next((x for x in PLAN[participant] if x['dt_us']==selected), None)
            stage1_ok=(s:=next((s for s in summary['stage1'] if s.get('case')==item['name']), None)) and s.get('complete') and s.get('numerical_qc') if item else False
            if item and stage1_ok:
                set_campaign_phase(participant, f'продолжение до 60 с, шаг {selected} мкс')
                console(f'[STAGE2] Продолжение 12 → 60 с: {item["name"]}')
                ref_item=next((x for x in PLAN[participant] if x['dt_us']==5), None) if selected==10 else None
                pilot_status=next((s for s in summary['stage1'] if ref_item and s.get('case')==ref_item['name']), {}) if ref_item else {}
                use_ref=bool(ref_item and any(x.get('case')=='grid_64x128' and x.get('execution')=='COMPLETE' and x.get('numerical_qc')=='PASS' for x in pilot_status.get('grids',[])))
                try:
                    if use_ref:
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            main_fut=pool.submit(run_stage, participant, item, FINAL, True)
                            ref_fut=pool.submit(late_reference, participant, ref_item)
                            summary['stage2']=main_fut.result()
                            summary['late_temporal_reference']=ref_fut.result()
                    else:
                        if selected==10:
                            console('[WARNING] Пилот dt=5 мкс на 64×128 не готов, поздняя проверка не запускается')
                        summary['stage2']=run_stage(participant, item, FINAL, artifacts=True)
                except Exception as exc:
                    console(f'[FAILED] STAGE2: {exc!r}')
                    summary['stage2']={'case':item['name'],'failed':repr(exc)}
                set_campaign_phase(participant, 'итоговая проверка сеток')
                if run_analysis(participant, FINAL, 'final'):
                    console('[WARNING] Финальное сравнение сеток не закончилось')
            else:
                console('[WARNING] Нет целого пилота на выбранном шаге. Продолжение до 60 с не запущено.')
            selection=read_json(root/'CD_ANALYSIS'/'final_grid_selection.json')
            summary['grid_selection']=selection
            stage2=summary.get('stage2') or {}
            if stage2.get('complete') and stage2.get('numerical_qc'):
                summary['status']='PASS_PROVISIONAL' if (selection.get('status')=='PROVISIONAL_CONVERGED' and selection.get('temporal_at_60s')=='PASS') else 'COMPLETE_NOT_CONVERGED'
            else:
                summary['status']='INCOMPLETE'
    except KeyboardInterrupt:
        summary['status']='INTERRUPTED'
        console('[WARNING] Остановлено. Снимки на месте, тот же запуск продолжит расчёт.')
        with ACTIVE_LOCK:
            procs=list(ACTIVE)
        for proc in procs:
            stop_process(proc)
    except Exception as exc:
        summary['status']='FAILED'
        summary['error']=repr(exc)
        console(f'[FAILED] {exc!r}')
    finally:
        if board is not None:
            board.stop()
            BOARD=None
        if not args.test:
            pack(participant, summary)
            print('\n========== РЕЗУЛЬТАТ ==========', flush=True)
            print('STATUS:', summary['status'], flush=True)
            print('DT (мкс):', summary.get('selected_dt_us'), flush=True)
            sel=summary.get('grid_selection') or {}
            print('СЕТКА:', sel.get('recommended_grid') or 'НЕ ВЫБРАНА', flush=True)
            print('СХОДИМОСТЬ:', sel.get('status','НЕ ОЦЕНИВАЛАСЬ'), flush=True)
            print(f'АРХИВ: results_{participant}/Send_to_AZ.zip', flush=True)
    return 0 if summary['status']=='PASS_PROVISIONAL' else 2

if __name__=='__main__':raise SystemExit(main())
