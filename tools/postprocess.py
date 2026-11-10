

from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image

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

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results_v42'
CONF=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
H=float(CONF['parameters']['H_water']);R=float(CONF['parameters']['R'])

def save(fig,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
 plt.close(fig)

def get_state(path):
 with np.load(path) as f:
  d={k:np.array(f[k]) for k in ('t','r','rf','z','zf','ur_faces','utheta','uz_faces','p','c_water')}
 d['t']=float(d['t'])
 d['ur']=0.5*(d['ur_faces'][:,:-1]+d['ur_faces'][:,1:])
 d['uz']=0.5*(d['uz_faces'][:-1]+d['uz_faces'][1:])
 d['vm']=np.hypot(d['ur'],d['uz'])
 d['us']=np.hypot(d['vm'],d['utheta'])
 d['h']=np.sum(d['c_water']*np.diff(d['zf'])[:,None],axis=0)
 c=d['c_water'];d['graph']=((c[0]>0.99)&(c[-1]<0.01)&(np.max(c[1:]-c[:-1],axis=0)<0.02))
 return d

def state_files(case):
 result={}
 for p in case.glob('segment_*/snapshot_*.npz'):
  try:result[int(p.stem.split('_')[-1])]=p
  except ValueError:pass
 return [result[n] for n in sorted(result)]

def load_rows(case):
 allrows={}
 for file in sorted(case.glob('segment_*/diagnostics.csv')):
  try:
   with file.open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
     t=float(row['t']); allrows[t]={k:float(v) if v not in ('',None) else np.nan for k,v in row.items()}
  except (ValueError,OSError) as e: print('[WARNING] bad diagnostics',file,str(e))
 return [allrows[t] for t in sorted(allrows)]

def radial_height_metrics(d):
 r=d['r'];h=d['h']
 edge=(r>=.72*R)&(r<=.9*R)
 if not edge.any():edge=r>=.7*R
 weights=r[edge]*np.diff(d['rf'])[edge]
 href=np.average(h[edge],weights=weights)
 
 s=min(4,len(r))
 h0=np.polyfit(r[:s]**2,h[:s],1)[-1]
 j=int(np.argmin(h))
 return dict(h0_m=float(h0),href_m=float(href),center_depth_m=float(href-h0),
             min_depth_m=float(href-h[j]),ring_depth_m=float(max(0.,h0-h[j])),
             ring_radius_m=float(r[j]),topology=float(d['graph'].mean()))

def symmetric(a):return np.concatenate((a[:,::-1],a),axis=1)
def edgex(d):return np.r_[-d['rf'][::-1],d['rf'][1:]]*1e3

def water_map(ax,d,field,*,cmap='turbo',vmin=None,vmax=None):
 c=symmetric(d['c_water'])
 z=symmetric(field)
 masked=np.ma.array(z,mask=(c<0.5)|(~np.isfinite(z)))
 ax.set_facecolor('#f0f1f4')
 im=ax.pcolormesh(edgex(d),d['zf']*1000,masked,cmap=cmap,shading='flat',
                   vmin=vmin,vmax=vmax,rasterized=True)
 x=d['r']*1000
 ax.plot(np.r_[-x[::-1],x],np.r_[d['h'][::-1],d['h']]*1000,color='#161b22',lw=1.6)
 ax.axvline(0,color='#808792',lw=.7,ls='--')
 ax.set_xlim(-R*1000,R*1000);ax.set_ylim(0,CONF['parameters']['H_box']*1000)
 ax.set_xlabel('x, мм');ax.set_ylabel('z, мм')
 return im

def plot_snapshot(path,output):
 d=get_state(path)
 c=d['c_water']
 
 pr=d['p']-np.mean(d['p'][:,int(.72*len(d['r'])):],axis=1)[:,None]
 utheta_abs=np.abs(d['utheta']);vm=d['vm']
 ell=d['utheta']*d['r'][None,:]
 omega_z=np.gradient(ell,d['r'],axis=1,edge_order=2)/np.maximum(d['r'][None,:],1e-6)
 fields=[(utheta_abs,'|uθ|, м/с','turbo'),(vm,'|uм|, м/с','viridis'),(pr,'p − pпериф(z), Па','coolwarm'),(omega_z,'ωz, с⁻¹','coolwarm')]
 fig,axs=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
 for ax,(f,title,cmap) in zip(axs.flat,fields):
  good=f[c>=.5]
  if len(good): lim=np.nanpercentile(np.abs(good),99)
  else:lim=1.
  if cmap=='coolwarm':im=water_map(ax,d,f,cmap=cmap,vmin=-max(lim,1e-10),vmax=max(lim,1e-10))
  else:im=water_map(ax,d,f,cmap=cmap,vmin=0,vmax=max(lim,1e-9))
  ax.set_title(title,fontsize=14)
  fig.colorbar(im,ax=ax,shrink=.74)
 fig.suptitle(f'Двухфазное течение | t={d["t"]:.3f} с | воздух не окрашен',fontsize=16)
 save(fig,output)

def plot_profiles(last,first,output):
 d=get_state(last);m=radial_height_metrics(d)
 r=d['r']*1e3
 fig,ax=plt.subplots(figsize=(10,5.5))
 ax.plot(r,d['h']*1e3,label=f't={d["t"]:.2f} с',color='tab:blue')
 if first!=last:
  b=get_state(first);ax.plot(b['r']*1e3,b['h']*1e3,label=f't={b["t"]:.2f} с',ls='--',color='tab:gray')
 ax.scatter([m['ring_radius_m']*1e3],[np.min(d['h'])*1e3],marker='x',s=80,color='tab:red',label='Минимум')
 ax.set(xlabel='r, мм',ylabel='h, мм',title='Реальная форма поверхности: вода–воздух')
 ax.grid(alpha=.25);ax.legend();save(fig,output)

def physical_timeline(files,output):
                                                                                 
    all_rows=[]
    for p in files:
        d=get_state(p);m=radial_height_metrics(d)
        radial=d['r'][None,:];dz=np.diff(d['zf'])[:,None]
        weights=2*np.pi*radial*np.diff(d['rf'])[None,:]*dz*np.clip(d['c_water'],0,1)
        er=0.5*CONF['parameters']['rho_water']*np.sum(weights*d['ur']**2)
        et=0.5*CONF['parameters']['rho_water']*np.sum(weights*d['utheta']**2)
        ez=0.5*CONF['parameters']['rho_water']*np.sum(weights*d['uz']**2)
        surface_band=(d['z'][:,None] <= d['h'][None,:])&(d['z'][:,None]>=d['h'][None,:]-.005)&(d['c_water']>=.5)
        include=(radial>.004)&(radial<.035)&surface_band
        near=float(np.sqrt(np.mean(d['utheta'][include]**2))) if include.any() else np.nan
        threshold=0.05
        flow=(np.abs(d['utheta'])>=threshold)&(d['c_water']>=.5)
        front=float(np.max(d['z'][np.where(flow)[0]])) if flow.any() else np.nan
        all_rows.append(dict(t_s=d['t'],**m,
                             energy_radial_J=float(er),energy_azimuth_J=float(et),energy_vertical_J=float(ez),
                             rms_near_surface_utheta_m_s=near,swirl_front_m=front))
    if all_rows:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    return all_rows

def plot_timeline(rows,output):
 if not rows:return
 t=np.array([x['t'] for x in rows]);keys=[('apparent_funnel_depth_m',1e3,'h(R)-h(0), мм'),
 ('utheta_max_m_s',1,'max |uθ|, м/с'),('kinetic_energy_J',1e3,'Кинетическая энергия, мДж'),
 ('graph_columns_fraction',1,'Доля однозначных столбцов')]
 fig,axs=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
 for ax,(k,scale,label) in zip(axs.flat,keys):
  y=np.array([x.get(k,np.nan) for x in rows])*scale
  ax.plot(t,y,color='tab:blue');ax.grid(alpha=.3)
  ax.set_xlabel('t, с');ax.set_ylabel(label)
 fig.suptitle('Динамика и контроль качества решения',fontsize=16)
 save(fig,output)

def plot_core(d,output):
 r=d['r'];z=d['z'];ut=d['utheta'];vm=d['vm'];c=d['c_water']
 ell=ut*r[None,:]
 wz=np.gradient(ell,r,axis=1,edge_order=2)/np.maximum(r[None,:],1e-7)
 frac=ut**2/(vm**2+ut**2+1e-8)
 frac[(c<.5)|((vm**2+ut**2)<.03**2)]=np.nan
 j=np.argmin(abs(z-.012))
 fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 im=axs[0].pcolormesh(r*1e3,z*1e3,np.ma.masked_where(c<.5,wz),shading='auto',cmap='coolwarm')
 fig.colorbar(im,ax=axs[0],shrink=.75);axs[0].set_title('Осевая завихрённость, с⁻¹')
 im=axs[1].pcolormesh(r*1e3,z*1e3,frac,shading='auto',cmap='viridis',vmin=0,vmax=1)
 fig.colorbar(im,ax=axs[1],shrink=.75);axs[1].set_title('Доля азимутальной скорости²')
 axs[2].plot(r*1e3,ut[j],label='uθ')
 axs[2].plot(r*1e3,vm[j],label='|uм|')
 axs[2].set(title=f'Профили при z≈{z[j]*1e3:.1f} мм',xlabel='r, мм',ylabel='Скорость, м/с')
 axs[2].grid(alpha=.3);axs[2].legend()
 for ax in axs[:2]:ax.set(xlabel='r, мм',ylabel='z, мм')
 fig.suptitle(f'Вихревое ядро — диагностическое определение | t={d["t"]:.2f} с')
 save(fig,output)

def generate_animation(files,out,*,max_frames=24,fps=10):
 if not files:return
 count=min(len(files),max_frames)
 ids=sorted(set(np.linspace(0,len(files)-1,count).astype(int).tolist()))
 out.parent.mkdir(parents=True,exist_ok=True)
 vmax_u=0.
 vmax_m=0.
 for i in ids:
  d=get_state(files[i]);mask=d['c_water']>=.5
  if np.any(mask):
   vmax_u=max(vmax_u,float(np.max(np.abs(d['utheta'][mask]))))
   vmax_m=max(vmax_m,float(np.max(d['vm'][mask])))
 vmax_u=max(vmax_u,1.e-6)
 vmax_m=max(vmax_m,1.e-6)
 for mode,vmax,title,cmap in [('azimuthal',vmax_u,'Азимутальная скорость |uθ|','turbo'),('meridional',vmax_m,'Меридиональная скорость |uₘ|','viridis')]:
  frames=[]
  for i in ids:
   d=get_state(files[i]);c=d['c_water'];x=d['r']*1e3;z=d['z']*1e3
   f=np.abs(d['utheta']) if mode=='azimuthal' else d['vm']
   fig,ax=plt.subplots(figsize=(6.5,7.6),dpi=95)
   im=water_map(ax,d,f,cmap=cmap,vmin=0,vmax=vmax)
   if mode=='meridional':
    stz=max(2,len(z)//20);strad=max(2,len(x)//13)
    rx=np.r_[-x[::-1],x]
    u=np.c_[-d['ur'][:,::-1],d['ur']]
    w=np.c_[d['uz'][:,::-1],d['uz']]
    water=symmetric(c)
    U=u[::stz,::strad].copy();W=w[::stz,::strad].copy();mask=water[::stz,::strad]<.5
    U[mask]=np.nan;W[mask]=np.nan
    ax.quiver(rx[::strad],z[::stz],U,W,color='white',scale_units='xy',scale=3,width=.003)
   ax.set_aspect('equal','box')
   ax.set_title(f'{title}   t={d["t"]:.2f} с',fontsize=12)
   fig.colorbar(im,ax=ax,pad=.02,fraction=.045,label='м/с')
   fig.tight_layout()
   fig.canvas.draw()
   w0,h0=fig.canvas.get_width_height()
   img=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(h0,w0,4)
   frames.append(Image.fromarray(img[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=160))
   plt.close(fig)
  dest=out.parent/(mode+'_velocity.gif')
  frames[0].save(dest,save_all=True,append_images=frames[1:],duration=int(1000/fps),loop=0,optimize=False,disposal=2)
  print('[VISUAL]',dest,'frames',len(frames),flush=True)

def quasi_stationary(case,target):
                                                                                 
    path=RES/'analysis'/case.name/'physical_timeline.csv'
    if not path.exists():return {'status':'NOT_EVALUATED','reason':'No physical_timeline.csv'}
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    t=np.array([float(r['t_s']) for r in rows]) if rows else np.array([])
    width=CONF['quality_thresholds']['quasi_steady_window_s']
    if len(t)<6 or np.max(t)<2*width:return {'status':'NOT_EVALUATED','reason':'Less than two averaging windows'}
    
    end=float(np.max(t));before=(t>=end-2*width)&(t<end-width);after=(t>=end-width)&(t<=end)
    if before.sum()<2 or after.sum()<2:return {'status':'NOT_EVALUATED','reason':'Too few snapshots per window'}
    metrics={}
    for key,scale in [('center_depth_m',1e-4),('energy_azimuth_J',1e-7),('energy_vertical_J',1e-7),('rms_near_surface_utheta_m_s',0.001)]:
        val=np.array([float(r[key]) for r in rows]);m0=float(np.nanmean(val[before]));m1=float(np.nanmean(val[after]))
        metrics[key]={'previous':m0,'recent':m1,'relative_drift':float(abs(m1-m0)/max(abs(m1),scale))}
    limit=CONF['quality_thresholds']['quasi_steady_relative_change']
    passed=all(v['relative_drift']<=limit for v in metrics.values())
    return {'status':'PASS' if passed else 'FAIL','threshold':limit,'window_s':width,'metrics':metrics}

def evaluate_convergence(cases,output):
 available=[]
 for c in cases:
  f=state_files(c)
  if f:
   states=[get_state(f[-1])]
   available.append((c.name,states[0],f))
 out={'spatial_convergence':'NOT_EVALUATED','reason':'Need 3 completed grids to compare','grid_metrics':{}}
 target=CONF['physical_time_s']
 for name,d,files in available:
  last=radial_height_metrics(d)
  case=next(c for c in cases if c.name==name)
  out['grid_metrics'][name]={'last_time_s':d['t'],**last,'quasi_stationary':quasi_stationary(case,target)}
 if len(available)<3 or any(d['t']<target-1e-7 for _,d,_ in available):
  output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');return out
 
 names,datasets,flists=zip(*available)
 reference=datasets[-1]
 rr=np.linspace(.002,R-.002,90);zz=np.linspace(.003,H-.012,90)
 ZZ,RR=np.meshgrid(zz,rr,indexing='ij');coords=np.c_[ZZ.ravel(),RR.ravel()]
 def sample(d,name):
  a=d[name]
  if name=='p':
   a=a-np.mean(a[:,-max(2,len(d['r'])//5):],axis=1)[:,None]
  vals=RegularGridInterpolator((d['z'],d['r']),a,bounds_error=False,fill_value=np.nan)(coords).reshape(ZZ.shape)
  return vals
 refvals={key:sample(reference,key) for key in ('utheta','ur','uz','p')}
 def relative_difference(d):
  metrics={}
  for key in refvals:
   val=sample(d,key);ref=refvals[key]
   mask=np.isfinite(val)&np.isfinite(ref)
   ww=RR[mask];a=val[mask]-ref[mask];b=ref[mask]
   denom=np.sqrt(np.sum(ww*b*b)/np.sum(ww))
   dif=np.sqrt(np.sum(ww*a*a)/np.sum(ww))/max(denom,1e-7)
   metrics[key]=float(dif)
  return metrics
 errors={names[i]:relative_difference(datasets[i]) for i in range(2)}
 ref=radial_height_metrics(reference)
 dprev=radial_height_metrics(datasets[-2]);
 depth_error=abs(dprev['center_depth_m']-ref['center_depth_m'])/max(abs(ref['center_depth_m']),1e-4)
 thresholds=CONF['quality_thresholds'];limit=thresholds['target_spatial_relative_difference']
 maxfield=max(errors[names[-2]].values());steady=all(out['grid_metrics'][n]['quasi_stationary']['status']=='PASS' for n in names)
 pass_flag=(depth_error<=limit and maxfield<=limit and steady)
 out.update({'spatial_convergence':'PASS' if pass_flag else 'FAIL','target':limit,
             'reference_grid':names[-1], 'relative_depth_difference':float(depth_error),
             'relative_field_differences':errors,'field_comparison_domain':'r=2..R-2mm, z=3..H-12mm',
             'quasi_stationary_all_grids':steady})
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 for name,d,_ in available:
  axs[0].plot(d['r']*1e3,d['h']*1e3,label=name)
 axs[0].set(xlabel='r, мм',ylabel='h, мм',title=f'Сходимость поверхности при t={target:g} с')
 axs[0].grid(alpha=.3);axs[0].legend(fontsize=8)
 fields=('utheta','ur','uz','p')
 xx=np.arange(4)
 axs[1].bar(xx,np.array([errors[names[-2]][k]*100 for k in fields]),width=.6)
 axs[1].axhline(limit*100,ls='--',color='tab:red')
 axs[1].set(xticks=xx,xticklabels=['uθ','ur','uz','p'],ylabel='Относительная RMS-разность, %',title='Средняя vs тонкая сетка')
 axs[1].grid(axis='y',alpha=.25)
 save(fig,output.parent/'spatial_convergence.png')
 output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');return out

def plot_rotation_input(output):
    p=ROOT/'inputs/omega_with_assumed_hold_to_60s.csv'
    d=np.genfromtxt(p,delimiter=',',names=True,encoding='utf-8-sig')
    t=np.atleast_1d(d['time_s']);omega=np.atleast_1d(d['omega_rad_s'])
    last_obs=float(t[-2])
    fig,ax=plt.subplots(figsize=(11,5.3))
    measured=t<=last_obs+1e-9
    ax.plot(t[measured],omega[measured],lw=1.8,color='tab:blue',label='Трекинг 30 FPS (неоднозначен при быстрых оборотах)')
    ax.plot(t[~measured | (t==last_obs)],omega[~measured | (t==last_obs)],lw=2.2,ls='--',color='tab:orange',label='Предположение: ω постоянна до 60 с')
    ax.axvline(last_obs,color='0.4',ls=':')
    ax.set(xlim=(0,60),xlabel='t, с',ylabel='ω якоря, рад/с',title='Измеренная часть и отдельно обозначенная экстраполяция')
    ax.grid(alpha=.3);ax.legend(fontsize=9);save(fig,output)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');a=ap.parse_args()
 analysis=RES/'analysis';analysis.mkdir(parents=True,exist_ok=True)
 plot_rotation_input(analysis/'input_omega_and_assumption.png')
 cases=[RES/'cases'/g['name'] for g in CONF['grids']]
 for case in cases:
  files=state_files(case)
  if not files:
   print('[VISUAL] No snapshots for',case.name);continue
  dest=analysis/case.name;dest.mkdir(parents=True,exist_ok=True)
  plot_snapshot(files[-1],dest/'final_fields.png')
  plot_profiles(files[-1],files[0],dest/'surface_profiles.png')
  plot_timeline(load_rows(case),dest/'timeseries.png')
  physical_timeline(files,dest/'physical_timeline.csv')
  plot_core(get_state(files[-1]),dest/'vortex_core.png')
  generate_animation(files,dest/'water_air_swirl.gif',
                     max_frames=min(24,CONF['visualization']['max_animation_frames']),fps=CONF['visualization']['fps'])
  print('[VISUAL] Finished case',case.name,'last t=',get_state(files[-1])['t'],flush=True)
 conv=evaluate_convergence(cases,analysis/'convergence_status.json')
 print('[QUALITY] spatial convergence:',conv['spatial_convergence'],flush=True)
 return 0
if __name__=='__main__':sys.exit(main())
