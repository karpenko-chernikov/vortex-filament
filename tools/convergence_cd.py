from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['grid.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['font.family'] = 'DejaVu Sans'

ROOT = Path(__file__).resolve().parents[1]
NR = (64, 80, 96)
DT = (20, 10, 5)


def json_read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def snapshots(case):
    paths = list((case/'results_v42'/'cases').glob('*/segment_*/snapshot_*.npz'))
    d = {}
    for p in paths:
        try:
            n = int(p.stem.split('_')[-1])
        except ValueError:
            continue
        if p.parent.parent.name == case.name:
            continue
        d[(p.parent.parent.name, n)] = p
    return d


def fetch_state(base, label, nr, target_t):
    grid = base/label/'results_v42'/'cases'/f'grid_{nr}x{nr*2}'
    if not grid.exists():
        grid = base/label/'results_v42'/'cases'/f'{nr}x{nr*2}'
    paths = sorted(grid.glob('segment_*/snapshot_*.npz'))
    best = None
    for p in paths:
        try:
            with np.load(p) as a:
                t = float(a['t'])
        except Exception:
            continue
        if t <= target_t + 1e-5 and (best is None or t > best[0]):
            best = (t, p)
    if best is None or target_t-best[0] > 0.13:
        return None
    return state_from_npz(best[1])


def state_from_npz(path):
    with np.load(path) as a:
        d = {k: np.asarray(a[k]) for k in ('t','r','rf','z','zf','ur_faces','utheta','uz_faces','p','c_water')}
    d['t'] = float(d['t'])
    d['ur'] = 0.5*(d['ur_faces'][:,1:]+d['ur_faces'][:,:-1])
    d['uz'] = 0.5*(d['uz_faces'][1:,:]+d['uz_faces'][:-1,:])
    d['h'] = (d['c_water']*np.diff(d['zf'])[:,None]).sum(axis=0)
    return d


def metrics(d):
    if d is None:
        return None
    r,h=d['r'],d['h']
    rim = (r>=0.72*r[-1]/0.985) & (r<=0.90*r[-1]/0.985)
    if not np.any(rim):
        rim = r>=0.7*r[-1]
    w = r[rim]*np.diff(d['rf'])[rim]
    href = float(np.average(h[rim],weights=w))
    n=min(4,len(r))
    axis = float(np.polyfit(r[:n]**2,h[:n],1)[-1])
    j=int(np.nanargmin(h))
    z0=0.006
    k=int(np.argmin(abs(d['z']-z0)))
    rr=d['r']
    core=(rr<=.005)
    c=d['c_water'][k]
    wet = np.clip(c[core],0,1)
    ff=d['uz'][k,core]
    flux = float(np.sum(ff*wet*np.pi*(d['rf'][1:][core]**2-d['rf'][:-1][core]**2)))
    smask=(d['c_water'][k]>=.95)
    if np.any(smask):
        vmax=float(np.max(np.abs(d['utheta'][k,smask])))
        rpeak=float(d['r'][smask][np.argmax(np.abs(d['utheta'][k,smask]))])
    else:
        vmax=float('nan');rpeak=float('nan')
    return {'time_s':d['t'],'center_height_m':axis,'edge_height_m':href,
            'center_depth_m':href-axis,'min_depth_m':href-float(h[j]),
            'ring_radius_m':float(r[j]),'core_flux_5mm_z6mm_m3_s':flux,
            'utheta_peak_z6mm_m_s':vmax,'utheta_peak_radius_z6mm_m':rpeak}


def interp_error(a,b,field):
    ra,za=a['r'],a['z']; rb,zb=b['r'],b['z']
    rlo=max(ra[0],rb[0]);rhi=min(ra[-1],rb[-1]);zlo=max(za[0],zb[0]);zhi=min(za[-1],zb[-1])
    r = np.linspace(rlo,rhi,55)
    z = np.linspace(zlo,zhi,90)
    R,Z=np.meshgrid(r,z)
    points=np.stack((Z.ravel(),R.ravel()),axis=-1)
    def sample(s,key):
        u=s[key]
        if key=='p':
            m=s['c_water']>=.95
            u=u-float(np.mean(u[m])) if np.any(m) else u
        return RegularGridInterpolator((s['z'],s['r']),u,bounds_error=False,fill_value=np.nan)(points).reshape(Z.shape)
    x=sample(a,field);y=sample(b,field)
    wa=sample(a,'c_water');wb=sample(b,'c_water')
    wet=(wa>.95)&(wb>.95)&np.isfinite(x)&np.isfinite(y)
    if not np.any(wet):return None
    ww=np.maximum(R[wet],rlo)
    rmse=math.sqrt(float(np.average((x[wet]-y[wet])**2,weights=ww)))
    base=math.sqrt(float(np.average(y[wet]**2,weights=ww)))
    floor=10.0 if field=='p' else .01
    return {'relative':rmse/max(base,floor),'rms_diff':rmse,'reference_rms':base,'water_overlap_points':int(wet.sum())}


def compare(a,b):
    if a is None or b is None:return {'available':False}
    ma,mb=metrics(a),metrics(b)
    rim=np.linspace(max(a['r'][0],b['r'][0]),min(a['r'][-1],b['r'][-1]),120)
    ha=np.interp(rim,a['r'],a['h']);hb=np.interp(rim,b['r'],b['h'])
    profile_rmse=float(np.sqrt(np.average((ha-hb)**2,weights=rim)))
    pscale=max(float(np.sqrt(np.average(hb**2,weights=rim))),.002)
    dp=abs(ma['center_depth_m']-mb['center_depth_m'])
    dscale=max(abs(mb['center_depth_m']),.002)
    qs=abs(ma['core_flux_5mm_z6mm_m3_s']-mb['core_flux_5mm_z6mm_m3_s'])
    qscale=max(abs(ma['core_flux_5mm_z6mm_m3_s']),abs(mb['core_flux_5mm_z6mm_m3_s']),1e-8)
    result={'available':True,'center_depth_relative':dp/dscale,'surface_profile_relative':profile_rmse/pscale,
            'core_flux_relative':qs/qscale,'center_depth_diff_mm':dp*1e3,
            'surface_profile_rms_mm':profile_rmse*1e3,
            'time_a':float(a['t']),'time_b':float(b['t'])}
    for key in ('utheta','ur','uz','p'):
        result[f'{key}_error']=interp_error(a,b,key)
    return result


def scenarios(participant):
    src=json_read(ROOT/'series_CD.json')
    return src[participant]


def state_lookup(participant,target):
    root=ROOT/f'results_{participant}'
    out={}
    for s in scenarios(participant):
        dt=int(s['dt_us']);label=s['name']
        for nr in NR:
            out[(dt,nr)]=fetch_state(root,label,nr,target)
    return out


def evaluation(participant,target):
    data=state_lookup(participant,target)
    values={f'dt{dt}_grid{n}':metrics(data[(dt,n)]) for dt in DT for n in NR}
    temporal={}
    for n in NR:
        temporal[str(n)]={'20_vs_10':compare(data[(20,n)],data[(10,n)]),
                          '10_vs_5':compare(data[(10,n)],data[(5,n)])}
    spatial={}
    for dt in DT:
        spatial[str(dt)]={'64_vs_80':compare(data[(dt,64)],data[(dt,80)]),
                          '80_vs_96':compare(data[(dt,80)],data[(dt,96)])}
    return {'participant':participant,'target_time_s':target,'metrics':values,
            'temporal':temporal,'spatial':spatial}


def acceptable_time(c):
    return bool(c.get('available') and c.get('center_depth_relative',1)>-1 and
      c.get('center_depth_relative',1)<=.025 and
      c.get('surface_profile_relative',1)<=.04 and
      (c.get('utheta_error') or {}).get('relative',1)<=.06 and
      (c.get('ur_error') or {}).get('relative',1)<=.12 and
      (c.get('uz_error') or {}).get('relative',1)<=.12)


def pick_dt(e):
    t=e['temporal']
    if all(acceptable_time(t[str(n)]['10_vs_5']) for n in (64,80,96)):
        return 10, '10 и 5 мкс согласуются на всех трёх сетках 64×128, 80×160 и 96×192'
    if all(t[str(n)]['10_vs_5'].get('available') for n in (64,80,96)):
        return 5, 'Расхождение 10 и 5 мкс превышает допуски; для продолжения выбран более малый шаг 5 мкс (сходимость по времени ещё не доказана)'
    return None, 'Нет полного сопоставления 10 и 5 мкс по всем трём сеткам; дорогое продолжение остановлено до получения численной проверки'


def spatial_choice(e,dt):
    s=e['spatial'][str(dt)]
    if not s['80_vs_96'].get('available') or not s['64_vs_80'].get('available'):
        return {'status':'INSUFFICIENT_DATA','recommended_grid':None,'reason':'Нет полной тройки сеток 64×128, 80×160, 96×192 при одинаковом Δt'}
    def check(c):
        return c.get('available') and c.get('center_depth_relative',1)<=.05 and c.get('surface_profile_relative',1)<=.05 and (
          (c.get('utheta_error') or {}).get('relative',1)<=.08) and (
          (c.get('ur_error') or {}).get('relative',1)<=.12) and (
          (c.get('uz_error') or {}).get('relative',1)<=.12) and (
          (c.get('p_error') or {}).get('relative',1)<=.10) and c.get('core_flux_relative',1)<=.20
    low=check(s['64_vs_80'])
    high=check(s['80_vs_96'])
    if low and high:
        return {'status':'PROVISIONAL_CONVERGED','recommended_grid':'64×128',
                'reason':'Оба последовательных сравнения 64→80 и 80→96 прошли заданные допуски; 64×128 — экономичный кандидат, а не доказательство асимптотической сходимости'}
    if high:
        return {'status':'PROVISIONAL_CONVERGED','recommended_grid':'80×160',
                'reason':'80×160 и 96×192 согласованы; 64×128 — нет. 80×160 — экономичный кандидат; экстраполяционная и временная погрешность требуют отдельной проверки'}
    return {'status':'NOT_CONVERGED','recommended_grid':None,
            'best_available_tested_grid':'96×192',
            'reason':'Расхождение 80×160 и 96×192 выше принятых допусков: оптимальная сетка НЕ установлена'}


def plot(e,dt,out):
    root=ROOT/f'results_{e["participant"]}'
    t=e['target_time_s']
    data=state_lookup(e['participant'],t)
    fig,axs=plt.subplots(1,2,figsize=(14,5.8),constrained_layout=True)
    for n in NR:
        s=data[(dt,n)]
        if s is None:continue
        axs[0].plot(s['r']*1e3,s['h']*1e3,label=f'{n}×{2*n}')
    axs[0].set(title=f'Поверхность, t={t:.1f} с, Δt={dt} мкс',xlabel='Радиус, мм',ylabel='Высота, мм')
    for n in NR:
        s=data[(dt,n)]
        if s is None:continue
        z_target=.010
        iz=int(np.argmin(abs(s['z']-z_target)))
        good=s['c_water'][iz]>=.95
        axs[1].plot(s['r'][good]*1e3,s['utheta'][iz,good],label=f'{n}×{2*n}')
    axs[1].set(title='Окружная скорость при z≈10 мм',xlabel='Радиус, мм',ylabel='uθ, м/с')
    for ax in axs:ax.grid(alpha=.3);ax.legend(fontsize=10)
    fig.savefig(out,dpi=160);plt.close(fig)


def plot_temporal(e,out):
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    for n in NR:
        d=[];y=[]
        for dt in DT:
            x=e['metrics'].get(f'dt{dt}_grid{n}')
            if x is not None: d.append(dt);y.append(x['center_depth_m']*1000)
        if d:ax.plot(d,y,'o-',label=f'{n}×{n*2}')
    ax.set(xlabel='Шаг времени, мкс',ylabel='Глубина воронки, мм',title=f'Временная сходимость, t={e["target_time_s"]:.1f} с')
    ax.set_xticks(DT);ax.grid(alpha=.3);ax.legend()
    fig.savefig(out,dpi=160);plt.close(fig)


def write_report(e,selection,stage,out):
    lines=['# Независимые проверки пространственной и временной сходимости', '',
           f'Очередь: {e["participant"]}; физическое время: {e["target_time_s"]:.1f} с; этап: {stage}.', '',
           'Метки выполнения, геометрия и численная пригодность проверяются отдельно.', '',
           '| Шаг, мкс | Сетка | Глубина, мм | Вертикальный расход через r<5 мм при z≈6 мм, мм³/с |',
           '|---:|---|---:|---:|']
    for dt in DT:
        for n in NR:
            x=e['metrics'][f'dt{dt}_grid{n}']
            if x is None:
                lines.append(f'| {dt} | {n}×{n*2} | — | — |')
            else:
                lines.append(f'| {dt} | {n}×{n*2} | {x["center_depth_m"]*1000:.3f} | {x["core_flux_5mm_z6mm_m3_s"]*1e9:.3f} |')
    lines+=['','## Пространственные различия при одинаковом Δt','',
            '| Δt, мкс | Сетки | Глубина | Профиль h(r) | uθ | ur | uz | p |',
            '|---:|---|---:|---:|---:|---:|---:|---:|']
    for dt in DT:
        for tag in ('64_vs_80','80_vs_96'):
            c=e['spatial'][str(dt)][tag]
            def fmt(key):
                v=c.get(key)
                return f'{v*100:.1f}%' if isinstance(v,(int,float)) else '—'
            def ffield(key):
                v=(c.get(key+'_error') or {}).get('relative')
                return f'{v*100:.1f}%' if v is not None else '—'
            lines.append(f'| {dt} | {tag.replace("_vs_"," → ")} | {fmt("center_depth_relative")} | '
                         f'{fmt("surface_profile_relative")} | {ffield("utheta")} | {ffield("ur")} | {ffield("uz")} | {ffield("p")} |')
    lines+=['','## Выбор сетки','',f'Статус: **{selection.get("status","NOT_EVALUATED")}**.',
            f'Рекомендация: **{selection.get("recommended_grid") or "не установлена"}**.',
            selection.get('reason',''),'',
            'Нормированные RMS-расхождения вычисляются в области совместно заполненных водой ячеек (C>0.95).',
            'Давление сравнивается после независимого удаления произвольной аддитивной константы.',
            'Расход через центральный диск чувствителен к смене направления потока и должен проверяться отдельно.',
            'Оценка на трёх сетках не доказывает математическую асимптотическую сходимость, особенно при сложной геометрии VOF.',
            'Экспериментальная валидация по h(r,t) и полям скорости не проведена.']
    out.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def late_time_comparison(participant,e,selected):
    result={'time_s':e['target_time_s'],'selected_dt_us':selected,
            'same_grid_64x128':{'available':False},'baseline_A_64x128':{'available':False}}
    if selected is None or e['target_time_s'] < 59.9:
        return result
    root=ROOT/f'results_{participant}'
    bydt={i['dt_us']:i['name'] for i in scenarios(participant)}
    if selected==10:
        a=fetch_state(root,bydt[10],64,60.)
        b=fetch_state(root,bydt[5],64,60.)
        result['same_grid_64x128']=compare(a,b)
    elif selected==5:
        a=fetch_state(root,bydt[5],64,60.)
        ref=ROOT/'references_A'/f'{participant}_A_64x128_60s.npz'
        if ref.is_file():
            b=state_from_npz(ref)
            result['same_grid_64x128']=compare(a,b)
    reference=ROOT/'references_A'/f'{participant}_A_64x128_60s.npz'
    current=fetch_state(root,bydt[selected],64,60.)
    if reference.is_file() and current is not None:
        result['baseline_A_64x128']=compare(current,state_from_npz(reference))
    c=result['same_grid_64x128']
    result['temporal_reference_available']=bool(c.get('available'))
    result['temporal_consistency_at_60s']=('PASS' if acceptable_time(c) else 'NOT_CONFIRMED') if c.get('available') else 'NO_DATA'
    result['reference_note']='Вспомогательная NPZ 64×128 взята из ранее присланной очереди A; это исторический контроль, не новый независимый эксперимент.'
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--participant',choices=['C','D'],required=True)
    p.add_argument('--time',type=float,required=True)
    p.add_argument('--stage',choices=['pilot','final'],required=True)
    args=p.parse_args()
    out=ROOT/f'results_{args.participant}'/'CD_ANALYSIS'
    out.mkdir(parents=True,exist_ok=True)
    e=evaluation(args.participant,args.time)
    selected,reason=pick_dt(e)
    prev=json_read(out/'PILOT_DECISION.json')
    if args.stage=='pilot':
        rec={'selected_dt_us':selected,'reason':reason,'pilot_time_s':args.time,
             'all_nine_available':all(v is not None for v in e['metrics'].values())}
        (out/'PILOT_DECISION.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        selected=prev.get('selected_dt_us')
    if selected is not None:
        selection=spatial_choice(e,selected)
        plot(e,selected,out/f'{args.stage}_surface_and_azimuth.png')
    else:
        selection={'status':'INSUFFICIENT_DATA','recommended_grid':None,'reason':'Не удалось определить подходящую тройку сеток'}
    if args.stage=='final':
        reference=late_time_comparison(args.participant,e,selected)
        (out/'final_temporal_reference.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        if not reference.get('temporal_reference_available'):
            selection['temporal_at_60s']='NO_DATA'
        else:
            selection['temporal_at_60s']=reference['temporal_consistency_at_60s']
        if selection.get('status')=='PROVISIONAL_CONVERGED' and selection.get('temporal_at_60s')!='PASS':
            selection['reason']+='; однако временная согласованность на 60 с не подтверждена'
    plot_temporal(e,out/f'{args.stage}_temporal_convergence.png')
    (out/f'{args.stage}_comparisons.json').write_text(json.dumps(e,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/f'{args.stage}_grid_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(e,selection,args.stage,out/f'{args.stage}_REPORT.md')
    if args.stage=='final':
        with (out/f'{args.stage}_REPORT.md').open('a',encoding='utf-8') as fp:
            fp.write('\n## Временная проверка на 60 с\n\nСтатус: '+str(selection.get('temporal_at_60s','NO_DATA'))+'\n')
    print(f'[{args.stage.upper()}] dt={selected} mks; grid={selection.get("recommended_grid")}, status={selection["status"]}',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
