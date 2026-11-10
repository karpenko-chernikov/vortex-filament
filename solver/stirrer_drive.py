

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import numpy as np
import jax.numpy as jnp


def read_profile(path, *, allow_unreliable=False):
    path=Path(path)
    with path.open('r',encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    if not rows or not {'time_s','omega_rad_s'}.issubset(rows[0]):
        raise ValueError('CSV must have time_s and omega_rad_s columns')
    t=np.array([float(r['time_s']) for r in rows],dtype=float)
    w=np.array([float(r['omega_rad_s']) for r in rows],dtype=float)
    c=np.array([float(r.get('confidence','1') or 'nan') for r in rows],dtype=float)
    good=np.isfinite(t)&np.isfinite(w)
    if not np.all(np.diff(t[good])>0):
        raise ValueError('Times must be strictly increasing; no duplicates')
    if good.sum()<3:
        raise ValueError('Need at least three finite measurements')
    
    meta_path=path.with_suffix('.meta.json')
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    qual=str(meta.get('tracking_quality','UNKNOWN')).upper()
    if qual=='FAIL' and not allow_unreliable:
        raise ValueError('Source tracking is FAIL; use --allow-unreliable-profile only for diagnostic runs')
    t=t[good];w=w[good];c=c[good]
    if np.isfinite(c).any() and np.nanmedian(c)<0.30 and not allow_unreliable:
        raise ValueError('Low tracking confidence; profile rejected')
    return t,w,{'quality':qual, 'sampling_fps':meta.get('sampling_fps'),
                 'max_abs_omega':float(np.max(np.abs(w))),
                 'aliased_fraction':float(meta.get('aliased_fraction',np.nan)),
                 'source':str(path),'n':int(len(t))}


def make_driver(mesh, cfg):
                                                                    
    if cfg.drive_mode!='speed_slip':
        mesh.drive_times=jnp.array([0.,1.],dtype=jnp.float64)
        mesh.drive_omega=jnp.zeros((2,),dtype=jnp.float64)
        mesh.drive_metadata={'mode':'fixed_torque'}
        return mesh.drive_metadata
    if not cfg.drive_profile:
        raise ValueError('For --drive-mode speed_slip provide --drive-profile CSV')
    tt,ww,md=read_profile(cfg.drive_profile,allow_unreliable=cfg.allow_unreliable_profile)
    mesh.drive_times=jnp.asarray(tt,dtype=jnp.float64)
    mesh.drive_omega=jnp.asarray(ww*cfg.drive_sign,dtype=jnp.float64)
    mesh.drive_metadata=md
    return md


def omega_at(t,mesh,cfg):
    if cfg.drive_mode!='speed_slip':
        return jnp.asarray(0.,dtype=jnp.float64)
    return jnp.interp(t+cfg.drive_time_offset_s,mesh.drive_times,mesh.drive_omega,
                      left=mesh.drive_omega[0],right=mesh.drive_omega[-1])


def contact_weight(mesh,cfg):

    return jnp.exp(-0.5*((mesh.RC-cfg.bar_radius)/cfg.bar_sigma_r)**2
                   -0.5*((mesh.ZC-cfg.bar_center_z)/cfg.bar_sigma_z)**2)


def drive_terms(c,rho,ut,mesh,cfg,t):

    omg=omega_at(t,mesh,cfg)
    weight=contact_weight(mesh,cfg)
    ftheta=cfg.rho_water*c*cfg.coupling_rate_s*weight*(omg*mesh.RC-ut)
    accel=ftheta/rho
    moment=jnp.sum(mesh.RC*ftheta*mesh.cellvol)
    p_liquid=jnp.sum(ftheta*ut*mesh.cellvol)
    p_rotor=omg*moment
    p_slip=jnp.sum(cfg.rho_water*c*cfg.coupling_rate_s*weight*
                   (omg*mesh.RC-ut)**2*mesh.cellvol)
    return accel,moment,p_liquid,p_slip,omg,p_rotor


def effective_bar_reynolds(omega,bar_radius,nu):
    return abs(float(omega))*bar_radius**2/nu
