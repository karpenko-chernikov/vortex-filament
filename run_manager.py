

from __future__ import annotations
import argparse, csv, json, math, os, re, subprocess, sys, time, traceback, zipfile, threading, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
CFG=ROOT/'config.json'
SOLVER=ROOT/'solver/two_phase_axisymmetric.py'
RESULTS=ROOT/'results_v42'

def worker_count():
    raw=os.environ.get('VOF_WORKERS','').strip()
    cpus=os.cpu_count() or 1
    if not raw:
        return cpus
    try:
        return max(1, int(raw))
    except ValueError:
        return cpus

def blas_thread_count():
    return max(1, (os.cpu_count() or 1)//worker_count())

def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds<0:
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

def atomic_write_json(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

def publish_progress(case, payload):
    payload=dict(payload)
    payload['updated_unix']=time.time()
    atomic_write_json(Path(case)/'live_progress.json', payload)

def _lock_paths(prefix, count):
    directory=Path(os.environ.get('VOF_SLOT_DIR', str(ROOT/'.vof_slots')))
    directory.mkdir(parents=True, exist_ok=True)
    paths=[]
    for i in range(count):
        path=directory/f'{prefix}_{i:02d}.lock'
        fd=os.open(path, os.O_CREAT|os.O_RDWR, 0o644)
        os.close(fd)
        paths.append(path)
    return paths

def _try_lock(path):
    if os.name=='nt':
        import msvcrt
        handle=open(path, 'r+b')
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            return None
        return handle
    import fcntl
    handle=open(path, 'r+b')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle

def _unlock(handle):
    if handle is None:
        return
    try:
        if os.name=='nt':
            import msvcrt
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()

def wait_for_slot(prefix, count, notify=None, interval=15):
    paths=_lock_paths(prefix, count)
    last_notify=time.monotonic()
    while True:
        for path in paths:
            handle=_try_lock(path)
            if handle is not None:
                return handle
        now=time.monotonic()
        if notify is not None and now-last_notify>=interval:
            try:
                notify()
            except Exception:
                pass
            last_notify=now
        time.sleep(0.5)

def solver_env():
    env=os.environ.copy()
    env['PYTHONUNBUFFERED']='1'
    threads=str(blas_thread_count())
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]=threads
    if threads=='1':
        flag='--xla_cpu_multi_thread_eigen=false'
        current=env.get('XLA_FLAGS','')
        if flag not in current:
            env['XLA_FLAGS']=(current+' '+flag).strip()
    return env

def echo_solver_line(line):
    text=line.rstrip()
    if not text.strip() or text.lstrip().startswith('[ПРОГРЕСС]'):
        return
    if os.environ.get('VOF_DASHBOARD')=='1' and re.match(r'\s*\d+/\d+\s', text):
        return
    print('  '+text, flush=True)

def progress_payload(label, target, nsteps, step, phase, wall_s=0.0):
    fraction=0.0 if nsteps<=0 else min(1.0, step/nsteps)
    return {
        'series':ROOT.name,
        'case':label,
        'pid':os.getpid(),
        'step':step,
        'nsteps':nsteps,
        't_s':(step/nsteps*target) if nsteps else 0.0,
        't_end_s':target,
        'fraction':fraction,
        'steps_per_s':None,
        'eta_s':0 if phase=='готово' else None,
        'wall_s':wall_s,
        'phase':phase,
        'block_age_s':wall_s,
        'alive':True,
    }

def stamp_cases(cfg, phase, only_complete=False):
    for grid in cfg['grids']:
        case=RESULTS/'cases'/grid['name']
        path=case/'live_progress.json'
        if not path.exists():
            continue
        if only_complete and status_for(case, grid['name']).get('execution')!='COMPLETE':
            continue
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        data['phase']=phase
        data['pid']=os.getpid()
        data['alive']=True
        data['updated_unix']=time.time()
        atomic_write_json(path, data)

def load_config():
    d=json.loads(CFG.read_text(encoding='utf-8'))
    if not (0<d['physical_time_s']<=600): raise ValueError('physical_time_s must be in (0,600]')
    if d['save_interval_s']<=0: raise ValueError('save_interval_s must be positive')
    return d

def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def snapshots(case):
    result=[]
    for seg in sorted(case.glob('segment_*')):
        for p in seg.glob('snapshot_*.npz'):
            try: n=int(p.stem.split('_')[-1]); result.append((n,p))
            except (ValueError,IndexError): pass
    return sorted(result,key=lambda x:x[0])

def last_good_snapshot(case):
    
    import numpy as np
    for n,p in reversed(snapshots(case)):
        try:
            with np.load(p) as data:
                if all(k in data for k in ('t','ur_faces','utheta','uz_faces','p','c_water')):
                    t=float(data['t'])
                    if math.isfinite(t): return n,t,p
        except Exception: pass
    return None

def physical_fingerprint(config,grid,*,smoke=False):
    values=dict(config['parameters'])
    values.update({k:grid[k] for k in ('Nr','Nz','dt')})
    if smoke: values.update({'Nr':12,'Nz':24,'dt':2e-5,'z_refine_interface':1.0,
                              'z_refine_bottom':1.0,'r_axis_refine':1.0,'r_wall_refine':0.0})
    p=ROOT/values.get('drive_profile','')
    values['profile_sha256']=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'NOT_AVAILABLE'
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode('utf-8')).hexdigest()

def status_for(case, name):
    path=case/'status.json'
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return {'case':name,'execution':'NOT_STARTED'}

def metric_qc(case, conf):
    qc=conf['quality_thresholds']
    csvs=list(case.glob('segment_*/diagnostics.csv'))
    recs=[]
    for file in csvs:
        try:
            with file.open(encoding='utf-8-sig',newline='') as f: recs += list(csv.DictReader(f))
        except Exception: pass
    if not recs:
        return {'numerical_qc':'UNKNOWN','error':'No numerical diagnostics'}
    rows=[]
    for r in recs:
        try: rows.append({k:float(v) for k,v in r.items() if v not in ('',None)})
        except (ValueError,TypeError): continue
    if not rows:
        return {'numerical_qc':'UNKNOWN','error':'All diagnostics invalid'}

    phys=conf['parameters']
    analytic_initial_volume=math.pi*phys['R']**2*phys['H_water']
    max_vol=max(abs(r.get('water_volume_m3',float('nan'))/analytic_initial_volume-1.) for r in rows)
    max_div=max(abs(r.get('divergence_linf_1_s',float('nan'))) for r in rows)
    min_graph=min(r.get('graph_columns_fraction',float('nan')) for r in rows)
    moment_error=max(abs(r.get('budget_closure_residual_kg_m2_s',float('nan'))) for r in rows)
    moment_input=max(abs(r.get('integrated_input_kg_m2_s',0.)) for r in rows)
    relative_moment_error=moment_error/max(moment_input,1e-12)
    good=(math.isfinite(max_vol) and math.isfinite(max_div) and math.isfinite(min_graph)
          and math.isfinite(relative_moment_error)
          and max_vol<=qc['max_volume_drift'] and max_div<=qc['max_divergence_1_s']
          and min_graph>=qc['min_graph_columns_fraction'] and relative_moment_error<=1e-4)
    return {'numerical_qc':'PASS' if good else 'FAIL',
            'max_absolute_volume_drift':max_vol,'max_divergence_1_s':max_div,
            'max_relative_angular_balance_residual':relative_moment_error,
            'min_graph_columns_fraction':min_graph,'diagnostic_records':len(rows)}

def build_args(config,grid,case,segment,resume,smoke):
    p=dict(config['parameters'])
    if smoke:
        p.update({'Nr':12,'Nz':24,'dt':2e-5,
                  'z_refine_interface':1.0,'z_refine_bottom':1.0,
                  'r_axis_refine':1.0,'r_wall_refine':0.0})
        t_end=0.001
        interval=0.0005
    else:
        p.update({'Nr':grid['Nr'],'Nz':grid['Nz'],'dt':grid['dt']})
        t_end=float(config['physical_time_s'])
        interval=float(config['save_interval_s'])
    p['t_end']=t_end
    p['save_every']=max(1,int(round(interval/p['dt'])))
    if abs(p['save_every']*p['dt']-interval)>1e-9:
        raise ValueError('save_interval_s must be multiple of dt')
    p['block_steps']=25
    p['out']=str(segment.resolve())
    if 'drive_profile' in p:
        p['drive_profile']=str((ROOT/p['drive_profile']).resolve())
    if resume: p['resume_from']=str(resume.resolve())
    args=[sys.executable,'-u',str(SOLVER)]
    for k,v in p.items():
        args.extend(('--'+k.replace('_','-'),str(v)))
    return args,p

def run_case(config, grid, *, smoke=False, retry_failed=False):
    label='smoke' if smoke else grid['name']
    base=RESULTS/'smoke_test' if smoke else RESULTS/'cases'
    case=base/label
    case.mkdir(parents=True,exist_ok=True)
    old=status_for(case,label)
    fp=physical_fingerprint(config,grid,smoke=smoke)
    manifest=case/'physics_fingerprint.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8')).get('sha256')
        if previous!=fp:
            status={'case':label,'execution':'FAILED','reason':'CONFIGURATION_CHANGED',
                    'detail':'Cannot reuse CFD checkpoints after modifying physics, grid or input profile.',
                    'numerical_qc':'UNKNOWN'}
            write_json(case/'status.json',status)
            print('[FAILED]',label,': physical inputs changed; use a new results directory.',flush=True)
            return status
    else:
        
        if snapshots(case):
            status={'case':label,'execution':'FAILED','reason':'CHECKPOINT_WITHOUT_FINGERPRINT',
                    'numerical_qc':'UNKNOWN'}
            write_json(case/'status.json',status)
            print('[FAILED]',label,': found existing checkpoints without a matching run manifest',flush=True)
            return status
        write_json(manifest,{'sha256':fp,'case':label,'physics_and_grid_fixed':True})
    target=0.001 if smoke else float(config['physical_time_s'])
    dt_case=2e-5 if smoke else float(grid['dt'])
    nsteps_case=max(1, math.ceil(target/dt_case))
    last=last_good_snapshot(case)
    if last and last[1]>=target-1e-10:
        qc=metric_qc(case,config)
        status={'case':label,'execution':'COMPLETE','physical_time_reached_s':last[1],**qc,
                'experimental_validation':'NOT_AVAILABLE','spatial_convergence':'NOT_EVALUATED'}
        write_json(case/'status.json',status)
        publish_progress(case, progress_payload(label, target, nsteps_case, nsteps_case, 'готово'))
        print(f'[OK] {label}: existing saved solution reached t={last[1]:.4f}s, QC={qc["numerical_qc"]}',flush=True)
        return status
    if old.get('execution')=='FAILED' and not retry_failed:
        publish_progress(case, progress_payload(label, target, nsteps_case, 0, 'ошибка, предыдущий запуск'))
        print(f'[WARNING] {label}: prior failure. Skipping; use --retry-failed after fixing numerical cause.',flush=True)
        return old
    segidx=max([int(p.name.split('_')[-1]) for p in case.glob('segment_*') if p.name.split('_')[-1].isdigit()] or [0])+1
    segment=case/f'segment_{segidx:03d}'
    segment.mkdir(parents=True,exist_ok=True)
    resume=last[2] if last else None
    args,_=build_args(config,grid,case,segment,resume,smoke)
    print(f'\n[RUN] {label}: t={last[1] if last else 0:.3f} -> {target:.3f}s, '
          f'grid={12 if smoke else grid["Nr"]}x{24 if smoke else grid["Nz"]}, '
          f'dt={2e-5 if smoke else grid["dt"]}, snapshot interval={0.0005 if smoke else config["save_interval_s"]}s',flush=True)
    log=segment/'solver_stdout.log'
    begin=time.perf_counter()
    rc=90
    def notify_wait():
        age=time.perf_counter()-begin
        publish_progress(case, progress_payload(
            label, target, nsteps_case, 0,
            f'ждёт свободный процессор, уже {format_duration(age)}', age))
    notify_wait()
    slot=wait_for_slot('solver', worker_count(), notify_wait, 15)
    try:
        publish_progress(case, progress_payload(label, target, nsteps_case, 0, 'запуск решателя', time.perf_counter()-begin))
        env=solver_env()
        with log.open('w',encoding='utf-8',errors='replace') as f:
            proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace',bufsize=1,env=env,cwd=str(ROOT))
            stop_heartbeat=threading.Event()
            def heartbeat():
                while not stop_heartbeat.wait(20):
                    wall=time.perf_counter()-begin
                    live=case/'live_progress.json'
                    try:
                        data=json.loads(live.read_text(encoding='utf-8'))
                    except Exception:
                        data=None
                    if data and data.get('pid')==os.getpid() and str(data.get('phase','')).startswith('запуск'):
                        data['updated_unix']=time.time()
                        data['wall_s']=wall
                        data['block_age_s']=wall
                        data['phase']=f'запуск решателя, уже {format_duration(wall)}'
                        atomic_write_json(live, data)
                    cur=last_good_snapshot(case)
                    tcur=cur[1] if cur else (data.get('t_s') if data else 0.0)
                    if not isinstance(tcur,(int,float)):
                        tcur=0.0
                    progress={'case':label,'execution':'RUNNING','physical_time_reached_s':tcur,
                              'physical_time_target_s':target,'wall_elapsed_s':wall,
                              'last_snapshot':str(cur[2]) if cur else None}
                    write_json(case/'progress.json',progress)
                    if os.environ.get('VOF_DASHBOARD')=='1':
                        continue
                    eta=format_duration(data.get('eta_s')) if data else 'ещё нет оценки'
                    phase=data.get('phase') if data else 'запуск'
                    print(f'[HEARTBEAT] {label}: {phase}, wall={wall/3600:.2f} ч, t={tcur:.3f}/{target:.3f} с, осталось {eta}. Расчёт жив.',flush=True)
            beat=threading.Thread(target=heartbeat,daemon=True)
            beat.start()
            try:
                for line in proc.stdout:
                    f.write(line);f.flush()
                    echo_solver_line(line)
                rc=proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
            finally:
                stop_heartbeat.set()
                beat.join(timeout=1)
    except KeyboardInterrupt:
        rc=130
        print('[WARNING] Interrupted. Existing snapshots remain. Rerun to resume.',flush=True)
    except Exception as e:
        rc=90
        with log.open('a',encoding='utf-8') as f:f.write('\nMANAGER ERROR '+repr(e)+'\n'+traceback.format_exc())
        print('[FAILED] '+label+': '+repr(e),flush=True)
    finally:
        _unlock(slot)
    last=last_good_snapshot(case)
    reached=last[1] if last else 0.0
    qc=metric_qc(case,config)
    execution='COMPLETE' if rc==0 and reached>=target-1e-10 else ('PARTIAL' if rc in (0,130) else 'FAILED')
    status={'case':label,'execution':execution,'physical_time_target_s':target,
            'physical_time_reached_s':reached,'last_checkpoint':str(last[2].relative_to(ROOT)) if last else None,
            'return_code':rc,'wall_elapsed_s':round(time.perf_counter()-begin,3),
            'log':str(log.relative_to(ROOT)),**qc,
            'experimental_validation':'NOT_AVAILABLE', 'spatial_convergence':'NOT_EVALUATED'}
    write_json(case/'status.json',status)
    end_phase='готово' if execution=='COMPLETE' else ('остановлено' if execution=='PARTIAL' else 'ошибка')
    end_step=nsteps_case if execution=='COMPLETE' else min(nsteps_case, int(round(reached/dt_case)))
    end_payload=progress_payload(label, target, nsteps_case, end_step, end_phase, time.perf_counter()-begin)
    end_payload['t_s']=reached
    try:
        previous=json.loads((case/'live_progress.json').read_text(encoding='utf-8'))
        for key in ('divergence_linf_1_s','relative_water_volume_change','u_max_m_s','steps_per_s'):
            if key in previous:
                end_payload[key]=previous[key]
    except Exception:
        pass
    publish_progress(case, end_payload)
    print(f'[{'OK' if execution=="COMPLETE" and qc["numerical_qc"]=="PASS" else "FAILED" if execution=="FAILED" else "WARNING"}] '
          f'{label}: execution={execution}, t={reached:.4f}/{target:.4f}s, numerical_QC={qc["numerical_qc"]}.',flush=True)
    return status

def collect_summary(config):
    entries=[]
    for g in config['grids']:
        entries.append(status_for(RESULTS/'cases'/g['name'],g['name']))
    complete=all(x.get('execution')=='COMPLETE' for x in entries)
    passed=complete and all(x.get('numerical_qc')=='PASS' for x in entries)
    summary={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
             'physical_target_s':config['physical_time_s'],
             'cases':entries,'all_grid_executions_complete':complete,
             'all_grid_numerical_checks_pass':passed,
             'spatial_convergence':'NOT_EVALUATED',
             'experimental_validation':'NOT_ESTABLISHED',
             'notes':['Running 60 seconds is not evidence of spatial convergence.',
                      'Speed history above 30.433 s is assumed constant, not measured.',
                      'Magnet-fluid coupling coefficient is uncalibrated.']}
    return summary

def run_analysis():
    cmd=[sys.executable,str(ROOT/'tools/postprocess.py'),'--all']
    print('[ANALYZE] Generating scientific charts and visualization from available snapshots',flush=True)
    return subprocess.run(cmd,cwd=ROOT).returncode

def package_results():
    outfile=RESULTS/'SEND_TO_CHATGPT_v42.zip'
    outfile.parent.mkdir(parents=True,exist_ok=True)
    selected=set()
    
    for case in (RESULTS/'cases').glob('*') if (RESULTS/'cases').exists() else []:
        ss=snapshots(case)
        if ss:
            ix={0,len(ss)-1,len(ss)//3,2*len(ss)//3}
            selected.update(ss[i][1].resolve() for i in ix)
    with zipfile.ZipFile(outfile,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(RESULTS.rglob('*')):
            if not p.is_file() or p==outfile: continue
            if 'smoke_test' in p.parts: continue
            if p.suffix=='.npz' and p.resolve() not in selected:continue
            z.write(p,arcname=str(p.relative_to(RESULTS)))
        for extra in ('config.json','INPUT_PROVENANCE.md','inputs/omega_with_assumed_hold_to_60s.meta.json'):
            p=ROOT/extra
            if p.exists():z.write(p,arcname=extra)
    print('[PACKAGE] '+str(outfile),flush=True)

def write_report(summary):
    lines=['# Контроль вычислений до 60 секунд', '',
           'Эта таблица фиксирует реальные данные, а не заявленную функциональность программы.', '',
           '| Сетка | Исполнение | Время t, с | Численные проверки |',
           '|---|---|---:|---|']
    for item in summary['cases']:
        lines.append(f"| {item['case']} | {item.get('execution','NOT_STARTED')} | "
                     f"{item.get('physical_time_reached_s',0):.4f} | {item.get('numerical_qc','UNKNOWN')} |")
    lines += ['',f"**Достигнута 1 минута на всех сетках:** {'ДА' if summary['all_grid_executions_complete'] else 'НЕТ'}",
              f"**Численные балансы:** {'PASS' if summary['all_grid_numerical_checks_pass'] else 'НЕ ПОДТВЕРЖДЕНЫ'}",
              f"**Пространственная сходимость:** {summary['spatial_convergence']}",
              '**Экспериментальная валидация:** НЕ ПРОЙДЕНА: нет измерений h(r,t) и поля скорости воды.',
              '', 'Роторная скорость на отрезке 30.433–60 с экстраполирована, а коэффициент передачи импульса не калиброван.',
              'Даже полностью завершённая минута не означает физически валидированную модель.', '']
    (RESULTS/'QUALITY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def _restore_case_phases(cfg):
    for grid in cfg['grids']:
        case=RESULTS/'cases'/grid['name']
        path=case/'live_progress.json'
        if not path.exists():
            continue
        execution=status_for(case, grid['name']).get('execution')
        if execution=='COMPLETE':
            phase='готово'
        elif execution=='FAILED':
            phase='ошибка'
        else:
            phase='остановлено'
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        data['phase']=phase
        data['updated_unix']=time.time()
        atomic_write_json(path, data)

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--mode',choices=['full','smoke','analyze'],default='full')
    arg.add_argument('--retry-failed',action='store_true');a=arg.parse_args()
    os.environ.setdefault('VOF_WORKERS', str(os.cpu_count() or 1))
    os.environ.setdefault('VOF_SLOT_DIR', str(ROOT/'.vof_slots'))
    if blas_thread_count()==1:
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ.setdefault(key, '1')
    cfg=load_config()
    import numpy,scipy,jax
    print(f'[ENV] Python {sys.version.split()[0]} NumPy={numpy.__version__} SciPy={scipy.__version__} JAX={jax.__version__}',flush=True)
    if a.mode=='smoke':
        s=run_case(cfg,cfg['grids'][0],smoke=True,retry_failed=True)
        if s.get('execution')=='COMPLETE' and s.get('numerical_qc')=='PASS':
            print('[OK] SMOKE TEST: real two-phase CFD, save and pressure solve successful',flush=True)
            return 0
        print('[FAILED] SMOKE TEST',flush=True);return 2
    if a.mode=='full':
        from concurrent.futures import ThreadPoolExecutor
        grids=cfg['grids']
        print(f'[RUN] Сетки этой серии считаются параллельно. На машине одновременно до {worker_count()} расчётов. Схема, сетки и dt не меняются.',flush=True)
        if len(grids)==1:
            run_case(cfg,grids[0],retry_failed=a.retry_failed)
        else:
            with ThreadPoolExecutor(max_workers=len(grids)) as pool:
                futures=[pool.submit(run_case, cfg, g, smoke=False, retry_failed=a.retry_failed) for g in grids]
                for fut in futures:
                    fut.result()
    def notify_analysis():
        stamp_cases(cfg, 'ждёт постобработку, процесс жив')
    notify_analysis()
    analysis_slot=wait_for_slot('analysis', 1, notify_analysis, 15)
    stop_pulse=threading.Event()
    def pulse():
        while not stop_pulse.wait(15):
            stamp_cases(cfg, 'постобработка, процесс жив')
    pulse_thread=threading.Thread(target=pulse, daemon=True)
    stamp_cases(cfg, 'постобработка, процесс жив')
    pulse_thread.start()
    try:
        analyze_rc=run_analysis()
    finally:
        stop_pulse.set()
        pulse_thread.join(timeout=1)
        _unlock(analysis_slot)
        if a.mode=='full':
            _restore_case_phases(cfg)
    summary=collect_summary(cfg)
    
    conv=RESULTS/'analysis/convergence_status.json'
    if conv.exists():
        try: summary['spatial_convergence']=json.loads(conv.read_text(encoding='utf-8')).get('spatial_convergence','NOT_EVALUATED')
        except Exception: pass
    write_json(RESULTS/'STATUS.json',summary)
    write_report(summary)
    package_results()
    print('\n========== FINAL STATUS ==========',flush=True)
    for item in summary['cases']:
        print(f"[{item.get('execution','NOT_STARTED')}] {item['case']}: t={item.get('physical_time_reached_s',0):.4f}s QC={item.get('numerical_qc','UNKNOWN')}",flush=True)
    print('SPATIAL CONVERGENCE:',summary['spatial_convergence'],flush=True)
    print('EXPERIMENTAL VALIDATION: NOT ESTABLISHED (experimental surface and liquid velocities missing)',flush=True)
    if not summary['all_grid_executions_complete'] or not summary['all_grid_numerical_checks_pass']:
        print('[FAILED] The 60-second campaign is incomplete or violates numerical QC.',flush=True);return 2
    if analyze_rc!=0:print('[FAILED] Scientific postprocessing did not finish.',flush=True);return 3
    if summary['spatial_convergence']!='PASS':
        print('[WARNING] Computations complete; spatial convergence is not established.',flush=True)
        return 4
    print('[OK] EXECUTION AND GRID AGREEMENT PASS. This does NOT mean experimental validation.',flush=True)
    return 0
if __name__=='__main__':sys.exit(main())
