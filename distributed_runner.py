from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,signal,subprocess,sys,threading,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SERIES=json.loads((ROOT/'series.json').read_text(encoding='utf-8'))
BASE=json.loads((ROOT/'base_config.json').read_text(encoding='utf-8'))
ACTIVE=set()
ACTIVE_LOCK=threading.Lock()
DASHBOARD=None
DASHBOARD_EVERY_S=20


def digest(paths):
    h=hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def project_files():
    return [ROOT/'run_manager.py',ROOT/'base_config.json',ROOT/'series.json',*sorted((ROOT/'solver').glob('*.py')),*sorted((ROOT/'tools').glob('*.py')),*sorted((ROOT/'inputs').glob('*.csv'))]


def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    t.replace(path)


def prepare(participant, item):
    participant_dir=ROOT/f'results_{participant}'
    directory=participant_dir/item['name']
    directory.mkdir(parents=True,exist_ok=True)
    files=project_files()
    sha=digest(files)
    marker=directory/'RUN_FINGERPRINT.json'
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old['source_sha256']!=sha or old['scenario']!=item:
            raise RuntimeError('Исходники или параметры изменены; нельзя продолжить старую контрольную точку: '+item['name'])
    elif (directory/'results_v42').exists():
        raise RuntimeError('Найдены контрольные точки без контрольной суммы: '+item['name'])
    else:
        save_json(marker,{'source_sha256':sha,'scenario':item})
    for sub in ('solver','tools','inputs'):
        (directory/sub).mkdir(exist_ok=True)
        for p in (ROOT/sub).iterdir():
            if p.is_file() and p.suffix in ('.py','.csv','.json'):
                shutil.copy2(p,directory/sub/p.name)
    shutil.copy2(ROOT/'run_manager.py',directory/'run_manager.py')
    config=json.loads(json.dumps(BASE))
    config['experiment_label']=item['name']
    config['parameters']['H_water']=item['H_water_m']
    config['parameters']['coupling_rate_s']=item['coupling_rate_s']
    save_json(directory/'config.json',config)
    return directory


def format_duration(seconds):
    if seconds is None or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
        return 'ещё нет оценки'
    seconds=int(seconds)
    days,seconds=divmod(seconds,86400)
    hours,seconds=divmod(seconds,3600)
    minutes,seconds=divmod(seconds,60)
    if days:
        return f'{days}д {hours}ч {minutes:02d}м'
    if hours:
        return f'{hours}ч {minutes:02d}м'
    if minutes:
        return f'{minutes}м {seconds:02d}с'
    return f'{seconds}с'


def progress_bar(fraction, width=24):
    fraction=0.0 if not isinstance(fraction,(int,float)) or not math.isfinite(fraction) else fraction
    fraction=0.0 if fraction<0 else 1.0 if fraction>1 else fraction
    filled=int(fraction*width)
    if 0<fraction<1 and filled==0:
        filled=1
    if fraction<1 and filled==width:
        filled=width-1
    return '█'*filled+'░'*(width-filled)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, TypeError, ValueError):
        return False
    return True


def terminal_phase(phase):
    return str(phase).startswith(('готово','ошибка','останов','прервано'))


def stepping_phase(phase):
    return str(phase).startswith(('счёт','компил','снимок'))


def configure_runtime(test):
    os.environ.setdefault('VOF_SLOT_DIR', str(ROOT/'.vof_slots'))
    if test:
        os.environ['VOF_DASHBOARD']='0'
        os.environ.setdefault('VOF_WORKERS','1')
    else:
        os.environ.setdefault('VOF_WORKERS', str(os.cpu_count() or 1))
        os.environ.setdefault('VOF_DASHBOARD','1')


def console(message):
    board=DASHBOARD
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
             '[CASE]','=====','SPATIAL','EXPERIMENTAL','[FINAL]','Error','Traceback',
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
        try:
            proc.terminate()
        except OSError:
            pass


class CampaignDashboard:
    def __init__(self, participant):
        self.participant=participant
        self.stop=threading.Event()
        self.lock=threading.Lock()
        self.thread=None
        self.dest=ROOT/f'results_{participant}'

    def start(self):
        self.print_status()
        self.thread=threading.Thread(target=self._loop, name='vof-dashboard', daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.wait(DASHBOARD_EVERY_S):
            try:
                self.print_status()
            except Exception as exc:
                self.emit(f'[WARNING] сводка прогресса: {exc}')

    def stop_and_join(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        try:
            self.print_status()
        except Exception as exc:
            self.emit(f'[WARNING] итоговая сводка: {exc}')

    def emit(self, message):
        with self.lock:
            print(message, flush=True)

    def print_status(self):
        text=self._compose()
        with self.lock:
            for line in text:
                print(line, flush=True)
            print(flush=True)
        try:
            self.dest.mkdir(parents=True, exist_ok=True)
            with (self.dest/'PROGRESS.log').open('a', encoding='utf-8') as handle:
                handle.write('\n'.join(text)+'\n\n')
        except Exception:
            pass

    def _compose(self):
        participant=self.participant
        expected=[]
        for item in SERIES[participant]:
            for grid in BASE['grids']:
                nsteps=math.ceil(float(BASE['physical_time_s'])/float(grid['dt']))
                expected.append((item['name'], grid['name'], nsteps))
        found={}
        root=ROOT/f'results_{participant}'
        if root.exists():
            for path in root.glob('*/results_v42/cases/*/live_progress.json'):
                try:
                    data=json.loads(path.read_text(encoding='utf-8'))
                except Exception:
                    continue
                series=data.get('series') or path.parents[3].name
                case=data.get('case') or path.parent.name
                found[(series, case)]=data
        now=time.time()
        done_steps=0
        total_steps=0
        rates=[]
        jobs_left=0
        rows=[]
        for series, case, planned in expected:
            data=found.get((series, case))
            if not data:
                total_steps += planned
                jobs_left += 1
                rows.append(f'  {series:<12} {case:<16} ещё не начат')
                continue
            phase=str(data.get('phase') or 'запуск')
            abandoned=phase.startswith(('ошибка','прервано','останов'))
            step=planned if phase.startswith('готово') else min(planned, max(0, int(data.get('step') or 0)))
            fraction=step/planned if planned else 0.0
            if not abandoned:
                total_steps += planned
                done_steps += step
                if not terminal_phase(phase):
                    jobs_left += 1
            if stepping_phase(phase):
                rate=data.get('steps_per_s')
                if isinstance(rate,(int,float)) and math.isfinite(rate) and rate>0:
                    rates.append(float(rate))
            age=max(0, now-float(data.get('updated_unix') or now))
            alive=pid_alive(data.get('pid'))
            if phase.startswith('ошибка'):
                pulse='ошибка'
            elif terminal_phase(phase):
                pulse='завершён'
            elif not alive:
                pulse='ПРОЦЕСС ЗАВЕРШИЛСЯ'
            elif age>75:
                pulse=f'НЕТ ПУЛЬСА {int(age)} с'
            else:
                pulse=f'жив, пульс {int(age)} с'
            eta=format_duration(data.get('eta_s'))
            block_age=float(data.get('block_age_s') or 0.0)
            t_s=float(data.get('t_s') or 0.0)
            t_end=float(data.get('t_end_s') or BASE['physical_time_s'])
            divv=data.get('divergence_linf_1_s')
            health=f'  div={divv:.1e}' if isinstance(divv,(int,float)) and math.isfinite(divv) else ''
            speed=data.get('steps_per_s')
            speed_txt=f'  {speed:.1f} шаг/с' if isinstance(speed,(int,float)) and math.isfinite(speed) and speed>0 else ''
            rows.append(
                f'  {series:<12} {case:<16} [{progress_bar(fraction, 20)}] {fraction*100:5.1f}%  '
                f't={t_s:6.2f}/{t_end:.0f}с  осталось {eta}{speed_txt}  блок {block_age:.0f}с  {phase}  {pulse}{health}')
        overall=done_steps/total_steps if total_steps else 0.0
        try:
            workers=max(1, int(os.environ.get('VOF_WORKERS','1')))
        except ValueError:
            workers=1
        postprocessing=any('постобработ' in row for row in rows)
        if rates and jobs_left and done_steps<total_steps:
            throughput=(sum(rates)/len(rates))*min(workers, max(1, jobs_left))
            overall_eta=format_duration((total_steps-done_steps)/throughput)
        elif postprocessing:
            overall_eta='постобработка'
        elif total_steps and done_steps>=total_steps:
            overall_eta='0с'
        else:
            overall_eta='ещё нет оценки'
        clock=time.strftime('%H:%M:%S')
        lines=[
            f'┌ VOF  участник {participant}  {clock}  сводка каждые {DASHBOARD_EVERY_S} с',
            f'│ Общий [{progress_bar(overall)}] {overall*100:5.1f}%   шагов {done_steps/1e6:.2f} / {total_steps/1e6:.2f} млн   осталось ≈ {overall_eta}',
            f'│ Одновременно до {workers} расчётов. Схема, сетки и шаг времени не менялись.',
            f'│ Архив по окончании: results_{participant}/Send_to_AZ.zip',
        ]
        lines.extend(rows)
        lines.append('└ Пульс обновляется, даже если процент ещё стоит: идёт компиляция или блок шагов.')
        return lines


def run_process(cmd,cwd,logfile):
    with logfile.open('a',encoding='utf-8') as output:
        output.write('\n--- '+time.strftime('%Y-%m-%d %H:%M:%S')+' ---\n')
        output.flush()
        env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
        proc=subprocess.Popen(cmd,cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',env=env,bufsize=1,
                              start_new_session=True)
        with ACTIVE_LOCK:
            ACTIVE.add(proc)
        try:
            for line in proc.stdout:
                output.write(line)
                output.flush()
                if show_child_line(line):
                    console(line.rstrip())
            return proc.wait()
        except KeyboardInterrupt:
            stop_process(proc)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
        finally:
            with ACTIVE_LOCK:
                ACTIVE.discard(proc)


def classify(directory,rc):
    statusfile=directory/'results_v42'/'STATUS.json'
    data={}
    if statusfile.exists():
        try:data=json.loads(statusfile.read_text(encoding='utf-8'))
        except Exception:pass
    grid=data.get('cases',[])
    all_complete=len(grid)==3 and all(s.get('execution')=='COMPLETE' for s in grid)
    numerical_ok=all_complete and all(s.get('numerical_qc')=='PASS' for s in grid)
    convergence=data.get('spatial_convergence','NOT_EVALUATED')
    if not grid or any(g.get('execution')=='FAILED' for g in grid):
        tag='FAILED'
    elif not all_complete:
        tag='WARNING'
    elif not numerical_ok:
        tag='WARNING'
    else:
        tag='OK' if convergence=='PASS' else 'WARNING'
    return {'series':directory.name,'status':tag,'return_code':rc,'all_grids_complete':all_complete,'numerical_qc':numerical_ok,'spatial_convergence':convergence,'grids':grid}


def pack(participant, results):
    out=ROOT/f'results_{participant}'
    out.mkdir(exist_ok=True)
    archive=out/'Send_to_AZ.zip'
    tmp=out/'Send_to_AZ.pending.zip'
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for name in ('README.md','series.json','base_config.json'):
            p=ROOT/name
            z.write(p,arcname=name)
        summary=out/'SUMMARY.json'
        if summary.exists():z.write(summary,arcname='SUMMARY.json')
        progress_log=out/'PROGRESS.log'
        if progress_log.exists():z.write(progress_log,arcname='PROGRESS.log')
        for item in SERIES[participant]:
            folder=out/item['name']
            for file in sorted(folder.rglob('*')):
                if not file.is_file():continue
                if any(v in file.parts for v in ('.venv','.venv_v42','__pycache__')):continue
                if file.suffix=='.npz':continue
                if file.name.startswith('SEND_TO_CHATGPT') and file.suffix=='.zip':continue
                if file.name=='postprocess.py':continue
                if file.name.endswith('.py') or file.suffix in ('.csv','.json','.md','.txt','.log','.png','.gif','.mp4','.html'):
                    z.write(file,arcname=str(file.relative_to(out)))
            for grid in (folder/'results_v42'/'cases').glob('*') if (folder/'results_v42'/'cases').exists() else []:
                snapshots=[]
                for segment in grid.glob('segment_*'):
                    snapshots.extend(segment.glob('snapshot_*.npz'))
                if snapshots:
                    def read_step(p):
                        try:return int(p.stem.split('_')[-1])
                        except Exception:return -1
                    snapshots.sort(key=read_step)
                    selected={snapshots[0],snapshots[-1],snapshots[len(snapshots)//3],snapshots[(2*len(snapshots))//3]}
                    for p in sorted(selected):z.write(p,arcname=str(p.relative_to(out)))
                for bad in grid.glob('segment_*/FAILED_*.npz'):
                    z.write(bad,arcname=str(bad.relative_to(out)))
    tmp.replace(archive)
    print('[PACKAGE]',archive,round(archive.stat().st_size/1e6,2),'MB',flush=True)
    return archive


def main():
    global DASHBOARD
    parser=argparse.ArgumentParser()
    parser.add_argument('--participant',choices=('A','B'),required=True)
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--pack-only',action='store_true')
    args=parser.parse_args()
    participant=args.participant
    configure_runtime(args.test or args.pack_only)
    results=[]
    dest=ROOT/f'results_{participant}'
    dest.mkdir(exist_ok=True)

    def run_one(item):
        console(f"[CASE] {item['name']}  H={item['H_water_m']}  coupling={item['coupling_rate_s']}")
        try:
            directory=prepare(participant, item)
            if args.pack_only:
                status=classify(directory, 0)
            else:
                mode='smoke' if args.test else 'full'
                rc=run_process([sys.executable,'-u','run_manager.py','--mode',mode], directory, directory/'RUN_LOG.txt')
                if args.test:
                    status={'series':item['name'],'status':'OK' if rc==0 else 'FAILED','return_code':rc}
                else:
                    status=classify(directory, rc)
            console(f"[{status['status']}] {item['name']} {status.get('spatial_convergence','')}")
            return status
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            console(f"[FAILED] {item['name']} {exc!r}")
            return {'series':item['name'],'status':'FAILED','reason':repr(exc)}

    dashboard=None
    executor=None
    try:
        if args.test or args.pack_only:
            for item in SERIES[participant]:
                results.append(run_one(item))
        else:
            dashboard=CampaignDashboard(participant)
            DASHBOARD=dashboard
            workers=os.environ.get('VOF_WORKERS')
            n_series=len(SERIES[participant])
            n_grids=len(BASE['grids'])
            console(f'[СТАРТ] Очередь {participant}: {n_series} серии × {n_grids} сетки. Одновременно до {workers} процессов.')
            console('[СТАРТ] Схема, шаг времени и сетки те же, что в исходном расчёте. Ниже сводка не реже одного раза в минуту.')
            console(f'[СТАРТ] Архив по окончании: results_{participant}/Send_to_AZ.zip')
            dashboard.start()
            executor=ThreadPoolExecutor(max_workers=n_series)
            futures=[executor.submit(run_one, item) for item in SERIES[participant]]
            for fut in futures:
                results.append(fut.result())
    except KeyboardInterrupt:
        console('[WARNING] Остановлено. Сохранённые снимки на месте, повторный запуск продолжит с них.')
        with ACTIVE_LOCK:
            procs=list(ACTIVE)
        for proc in procs:
            stop_process(proc)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if dashboard is not None:
            dashboard.stop_and_join()
            DASHBOARD=None
        if not args.test:
            all_pass=all(i['status']=='OK' for i in results) and len(results)==3
            summary={'participant':participant,'count':len(results),'total':3,'all_criteria_pass':all_pass,'cases':results}
            save_json(dest/'SUMMARY.json',summary)
            pack(participant,results)
            for item in results:
                console(f"[{item['status']}] {item['series']}")
            console('[FINAL] ' + ('OK' if all_pass else 'WARNING or FAILED'))
    return 0 if len(results)==3 and all(i['status']=='OK' for i in results) else 2

if __name__=='__main__':raise SystemExit(main())
