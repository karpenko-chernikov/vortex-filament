

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.sparse.linalg import cg
import geometric_vof
import height_curvature
import meridional_transport
import stirrer_drive
import pressure_sparse_direct


@dataclass(frozen=True)
class Config:
    R: float = 0.04
    H_water: float = 0.06
    H_box: float = 0.09
    rho_water: float = 998.0
    rho_air: float = 1.2
    mu_water: float = 1.0e-3
    mu_air: float = 1.81e-5
    sigma: float = 0.072
    g: float = 9.81
    torque: float = 1.0e-5
    
    torque_protocol: str = "constant"  
    torque_on_s: float = 0.0
    torque_off_s: float = 0.12    
    torque_ramp_s: float = 0.0    
    
    
    drive_mode: str = "fixed_torque"   
    drive_profile: str = ""             
    drive_time_offset_s: float = 0.0   
    drive_sign: float = 1.0
    allow_unreliable_profile: int = 0
    coupling_rate_s: float = 5.0       
    bar_radius: float = 0.012         
    bar_center_z: float = 0.003       
    bar_sigma_r: float = 0.003        
    bar_sigma_z: float = 0.002        
    force_r: float = 0.012
    force_z: float = 0.004
    Nr: int = 32
    Nz: int = 64
    radial_stretch: float = 1.3
    r_axis_refine: float = 0.0
    r_wall_refine: float = 0.0
    r_axis_width_m: float = 0.006
    r_wall_width_m: float = 0.006
    
    
    z_refine_interface: float = 0.0
    z_refine_bottom: float = 0.0
    z_refine_top: float = 0.0
    z_refine_width_m: float = 0.004
    z_refine_bottom_width_m: float = 0.006
    z_refine_top_width_m: float = 0.006
    dt: float = 1.0e-5
    t_end: float = 0.002
    save_every: int = 40
    block_steps: int = 1  
    pressure_solver: str = "direct"   
    poisson_tol: float = 1.0e-8
    poisson_maxiter: int = 350
    fail_divergence_above: float = 0.0   
    fail_graph_fraction_below: float = 0.0  
    allow_unverified_bottom_contact: int = 0  
    fail_orphan_fraction_above: float = 5.0e-4  
    curvature_smooth_passes: int = 2
    curvature_method: str = "variational"  
    contact_angle_deg: float = 90.0  
    slip_length: float = 1.0e-5   
    
    initial_flow: str = ""
    resume_from: str = ""  
    out: str = "two_phase_output"
    angular_advection: str = "conservative"  
    meridional_advection: str = "consistent"  
    phase_advection: str = "geometric"  

    def validate(self):
        if self.drive_mode == 'speed_slip' and self.torque_protocol != 'constant':
            raise ValueError('Prescribed torque protocol applies only to fixed_torque')
        if self.torque_protocol not in ('constant','pulse','ramp'):
            raise ValueError('torque_protocol: constant | pulse | ramp')
        if self.torque_on_s < 0 or self.torque_ramp_s < 0:
            raise ValueError('Torque start/ramp cannot be negative')
        if self.torque_protocol == 'pulse' and (self.torque_off_s <= self.torque_on_s or
                2*self.torque_ramp_s > self.torque_off_s-self.torque_on_s):
            raise ValueError('For pulse: torque_off_s > torque_on_s + 2*torque_ramp_s')
        if self.pressure_solver not in ("direct","cg"):
            raise ValueError("pressure_solver: direct or cg")
        if self.drive_mode not in ('fixed_torque','speed_slip'):
            raise ValueError('drive_mode must be fixed_torque or speed_slip')
        if self.drive_mode == 'speed_slip':
            if self.coupling_rate_s<=0 or self.bar_radius<=0 or self.bar_sigma_r<=0 or self.bar_sigma_z<=0:
                raise ValueError('Invalid surrogate-bar geometry or coupling rate')
            if self.bar_radius>=self.R or self.bar_center_z<0 or self.bar_center_z>=self.H_water:
                raise ValueError('Effective bar envelope must be near bottom and inside water')
        if self.drive_sign not in (-1.,1.):
            raise ValueError('drive_sign must be +1 or -1')
        if not (0.0 < self.H_water < self.H_box and self.R > 0):
            raise ValueError("0 < H_water < H_box и R > 0 обязательны")
        if self.block_steps<1 or self.save_every<1:
            raise ValueError("block_steps and save_every must be >=1")
        if self.Nr < 8 or self.Nz < 12:
            raise ValueError("Nr>=8, Nz>=12")
        if min(self.rho_water, self.rho_air, self.mu_water, self.mu_air) <= 0:
            raise ValueError("Плотности и вязкости должны быть положительными")
        if any(x < 0 or x > 30 for x in (self.z_refine_interface, self.z_refine_bottom, self.z_refine_top, self.r_axis_refine, self.r_wall_refine)):
            raise ValueError('z-grid refinement weights must be within [0,30]')
        if min(self.z_refine_width_m, self.z_refine_bottom_width_m, self.z_refine_top_width_m, self.r_axis_width_m, self.r_wall_width_m) <= 0:
            raise ValueError('z-grid widths must be positive')
        if self.radial_stretch < 0 or self.radial_stretch > 3:
            raise ValueError("radial_stretch в диапазоне [0, 3]")
        if not (0 < self.contact_angle_deg < 180) or self.slip_length < 0:
            raise ValueError("contact_angle_deg должен быть (0,180), slip_length >=0")
        if self.curvature_method not in ("height", "variational", "legacy", "adaptive_hf"):
            raise ValueError("curvature_method: variational, height, legacy, adaptive_hf")
        if self.phase_advection not in ("geometric", "donor"):
            raise ValueError("phase_advection: geometric либо donor")
        if self.meridional_advection not in ("consistent", "legacy"):
            raise ValueError("meridional_advection: consistent либо legacy")
        if self.angular_advection not in ("conservative", "legacy"):
            raise ValueError("angular_advection: conservative либо legacy")
        if self.fail_orphan_fraction_above < 0:
            raise ValueError('fail_orphan_fraction_above must be nonnegative')
        if self.fail_divergence_above < 0:
            raise ValueError("fail_divergence_above must be nonnegative")
        if not (0<=self.fail_graph_fraction_below<=1):
            raise ValueError('fail_graph_fraction_below must be in [0,1]')
        if self.dt <= 0 or self.t_end <= 0 or self.sigma < 0 or self.g < 0:
            raise ValueError("Проверьте dt, t_end, sigma и g")


class Mesh:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        s = np.linspace(0.0, 1.0, cfg.Nr + 1)
        a = cfg.radial_stretch
        rf = cfg.R * (np.expm1(a*s)/np.expm1(a) if a > 1e-12 else s)
        if cfg.r_axis_refine or cfg.r_wall_refine:
            dense_r = np.linspace(0.0, cfg.R, 20001)
            if a > 1e-12:
                
                e = np.expm1(a)
                monitor = e/(a*cfg.R*(1+e*dense_r/cfg.R))
            else:
                monitor = np.full_like(dense_r,1/cfg.R)
            if cfg.r_axis_refine:
                monitor *= (1+cfg.r_axis_refine*np.exp(-0.5*(dense_r/cfg.r_axis_width_m)**2))
            if cfg.r_wall_refine:
                monitor *= (1+cfg.r_wall_refine*np.exp(-0.5*((cfg.R-dense_r)/cfg.r_wall_width_m)**2))
            I=np.zeros_like(dense_r)
            I[1:]=np.cumsum(0.5*(monitor[1:]+monitor[:-1])*np.diff(dense_r))
            rf=np.interp(np.linspace(0.0,I[-1],cfg.Nr+1),I,dense_r)
            rf[0],rf[-1]=0.,cfg.R
        rc = 0.5*(rf[:-1]+rf[1:])
        self.rf = jnp.asarray(rf)
        self.rc = jnp.asarray(rc)
        self.dr = jnp.asarray(np.diff(rf))
        self.drc = jnp.asarray(np.diff(rc))
        self.vr = 0.5 * (self.rf[1:]**2 - self.rf[:-1]**2)
        if cfg.z_refine_interface == cfg.z_refine_bottom == cfg.z_refine_top == 0.0:
            zf = np.linspace(0.0, cfg.H_box, cfg.Nz+1)
        else:
            zsample = np.linspace(0.0, cfg.H_box, 20001)
            w = np.ones_like(zsample)
            for strength, center, width in (
                (cfg.z_refine_interface, cfg.H_water, cfg.z_refine_width_m),
                (cfg.z_refine_bottom, 0.0, cfg.z_refine_bottom_width_m),
                (cfg.z_refine_top, cfg.H_box, cfg.z_refine_top_width_m)):
                if strength:
                    w += strength*np.exp(-0.5*((zsample-center)/width)**2)
            integral = np.zeros_like(zsample)
            integral[1:] = np.cumsum(0.5*(w[1:]+w[:-1])*np.diff(zsample))
            zf = np.interp(np.linspace(0.0,integral[-1],cfg.Nz+1),integral,zsample)
            zf[0],zf[-1] = 0.0,cfg.H_box
        zc = 0.5*(zf[:-1]+zf[1:])
        self.zf = jnp.asarray(zf)
        self.zc = jnp.asarray(zc)
        self.dzv = jnp.asarray(np.diff(zf))     
        self.dzc = jnp.asarray(np.diff(zc))      
        self.dz = cfg.H_box/cfg.Nz               
        self.min_dz = float(np.min(np.diff(zf)))
        self.RC, self.ZC = jnp.meshgrid(self.rc, self.zc, indexing='xy')
        self.cellvol = 2*jnp.pi*self.vr[None,:]*self.dzv[:,None]
        self.min_dr = float(np.min(np.diff(rf)))
        self.root_vol = jnp.sqrt(self.cellvol)
        height_curvature.prepare(self)
        self.direct_pressure = pressure_sparse_direct.make_solver(self)[0]


class State(NamedTuple):
    ur: jax.Array            
    ut: jax.Array            
    uz: jax.Array            
    p: jax.Array             
    c: jax.Array             


def div_flux(fr, fz, m: Mesh):
                                                             
    radial = (m.rf[1:][None,:]*fr[:,1:] - m.rf[:-1][None,:]*fr[:,:-1]) / m.vr[None,:]
    axial = (fz[1:,:] - fz[:-1,:])/m.dzv[:,None]
    return radial + axial


def averages(ur, uz):
    return 0.5*(ur[:,:-1]+ur[:,1:]), 0.5*(uz[:-1,:]+uz[1:,:])


def materials(c, cfg: Config):
    rho = cfg.rho_air + (cfg.rho_water-cfg.rho_air)*c
    mu = cfg.mu_air + (cfg.mu_water-cfg.mu_air)*c
    return rho, mu


def face_inverse_density(rho, m: Mesh):
    
    ar = jnp.zeros((m.cfg.Nz, m.cfg.Nr+1), dtype=rho.dtype)
    az = jnp.zeros((m.cfg.Nz+1, m.cfg.Nr), dtype=rho.dtype)
    ar = ar.at[:,1:-1].set(2/(rho[:,1:]+rho[:,:-1]))
    az = az.at[1:-1,:].set(2/(rho[1:,:]+rho[:-1,:]))
    az = az.at[-1,:].set(1/rho[-1,:])
    return ar, az


def grad_pressure(p, m: Mesh):
    gr = jnp.zeros((m.cfg.Nz,m.cfg.Nr+1), dtype=p.dtype)
    gr = gr.at[:,1:-1].set((p[:,1:]-p[:,:-1])/m.drc[None,:])
    gz = jnp.zeros((m.cfg.Nz+1,m.cfg.Nr), dtype=p.dtype)
    gz = gz.at[1:-1,:].set((p[1:,:]-p[:-1,:])/m.dzc[:,None])
    
    gz = gz.at[-1,:].set(-2*p[-1,:]/m.dzv[-1])
    return gr, gz


def poisson_operator(q, invr, invz, m: Mesh):
    gr, gz = grad_pressure(q, m)
    return -div_flux(invr*gr, invz*gz, m)


def poisson_diag(invr, invz, m: Mesh):
    right = m.rf[1:][None,:] * invr[:,1:] / jnp.concatenate((m.drc,jnp.ones((1,))))[None,:]
    left  = m.rf[:-1][None,:] * invr[:,:-1] / jnp.concatenate((jnp.ones((1,)),m.drc))[None,:]
    
    drdiag = (right + left) / m.vr[None,:]
    dzlow = jnp.concatenate((jnp.ones((1,),dtype=m.dzc.dtype),m.dzc))
    dzhigh = jnp.concatenate((m.dzc,m.dzv[-1:]/2))
    dzdiag = (invz[:-1,:]/dzlow[:,None]+invz[1:,:]/dzhigh[:,None])/m.dzv[:,None]
    return drdiag + dzdiag


def solve_increment(rhs, invr, invz, m: Mesh, cfg: Config):
                                                                         
    if cfg.pressure_solver == "direct":
        return m.direct_pressure(rhs,invr,invz)
    root = m.root_vol
    diag = jnp.maximum(poisson_diag(invr, invz, m), 1e-16)
    def A(y):
        return (root*poisson_operator(y.reshape(rhs.shape)/root, invr, invz, m)).ravel()
    def precond(y):
        return y/diag.ravel()
    b = (root*rhs).ravel()
    sol, _info = cg(A, b, M=precond, tol=cfg.poisson_tol,
                    atol=0.0, maxiter=cfg.poisson_maxiter)
    return sol.reshape(rhs.shape)/root


def project(u_star_r, u_star_z, rho, m: Mesh, cfg: Config):
    invr, invz = face_inverse_density(rho,m)
    rhs = -div_flux(u_star_r,u_star_z,m)/cfg.dt
    q = solve_increment(rhs, invr, invz, m, cfg)
    gr, gz = grad_pressure(q,m)
    ur = u_star_r - cfg.dt*invr*gr
    uz = u_star_z - cfg.dt*invz*gz
    ur = ur.at[:,0].set(0.0).at[:,-1].set(0.0)
    uz = uz.at[0,:].set(0.0)
    return ur, uz, q


def phase_face_values(c, ur, uz):

    cr = jnp.zeros_like(ur)
    cz = jnp.zeros_like(uz)
    cr = cr.at[:, 1:-1].set(jnp.where(ur[:, 1:-1] >= 0, c[:, :-1], c[:, 1:]))
    cz = cz.at[1:-1, :].set(jnp.where(uz[1:-1, :] >= 0, c[:-1, :], c[1:, :]))
    cz = cz.at[-1, :].set(jnp.where(uz[-1, :] >= 0, c[-1, :], 0.0))
    return cr, cz


def advect_c(c, ur, uz, m: Mesh, dt, donor=None):
                                                                                     
    cr, cz = donor if donor is not None else phase_face_values(c, ur, uz)
    cnew = c-dt*div_flux(ur*cr, uz*cz, m)
    return jnp.clip(cnew, 0.0, 1.0)


def angular_advection_fluxes(ut, ur, uz, c_old, m: Mesh, cfg: Config, donor=None):

    cr, cz = donor if donor is not None else phase_face_values(c_old, ur, uz)
    rho_r = cfg.rho_air+(cfg.rho_water-cfg.rho_air)*cr
    rho_z = cfg.rho_air+(cfg.rho_water-cfg.rho_air)*cz
    ell = m.RC*ut
    ell_r = jnp.zeros_like(ur)
    ell_r = ell_r.at[:, 1:-1].set(jnp.where(ur[:, 1:-1]>=0, ell[:, :-1], ell[:, 1:]))
    ell_z = jnp.zeros_like(uz)
    ell_z = ell_z.at[1:-1, :].set(jnp.where(uz[1:-1, :]>=0, ell[:-1, :], ell[1:, :]))
    ell_z = ell_z.at[-1, :].set(jnp.where(uz[-1, :]>=0, ell[-1, :], 0.0))
    return ur*rho_r*ell_r, uz*rho_z*ell_z


def angular_momentum_total(s: State, m: Mesh, cfg: Config):
    rho, _ = materials(s.c, cfg)
    return jnp.sum(rho*m.RC*s.ut*m.cellvol)


def angular_boundary_budget(ut, mu, z_ang_flux, m: Mesh, cfg: Config):
                                                                             
    tb = mu[0, :]*ut[0, :]/(cfg.slip_length+m.dzv[0]/2)
    bottom = -jnp.sum(2*jnp.pi*(m.rc**2)*tb*m.dr)
    bf = cfg.slip_length/(cfg.slip_length+cfg.R-m.rc[-1])
    ts = mu[:, -1]*(-ut[:, -1]/(cfg.slip_length+cfg.R-m.rc[-1])-
                    bf*ut[:, -1]/cfg.R)
    side = jnp.sum(2*jnp.pi*cfg.R**2*ts*m.dzv)
    top = jnp.sum(z_ang_flux[-1, :]*2*jnp.pi*m.vr)
    return bottom, side, top


def smooth_fraction(c, n):
    
    def f(_i, a):
        l=jnp.concatenate((a[:,:1],a[:,:-1]),axis=1)
        r=jnp.concatenate((a[:,1:],a[:,-1:]),axis=1)
        b=jnp.concatenate((a[:1,:],a[:-1,:]),axis=0)
        t=jnp.concatenate((a[1:,:],a[-1:,:]),axis=0)
        return (4*a+l+r+b+t)/8
    return lax.fori_loop(0, n, f, c)


def curvature_and_capillary(c, m: Mesh, cfg: Config):

    sm = smooth_fraction(c, cfg.curvature_smooth_passes)
    
    gr = jnp.zeros((cfg.Nz,cfg.Nr+1),dtype=c.dtype)
    gr = gr.at[:,1:-1].set((sm[:,1:]-sm[:,:-1])/m.drc[None,:])
    gz_center = jnp.concatenate(((sm[1:2,:]-sm[:1,:])/m.dzc[0],
                                  (sm[2:,:]-sm[:-2,:])/(m.zc[2:]-m.zc[:-2])[:,None],
                                  (sm[-1:,:]-sm[-2:-1,:])/m.dzc[-1]),axis=0)
    gz_at_rf=jnp.zeros_like(gr)
    gz_at_rf=gz_at_rf.at[:,1:-1].set(0.5*(gz_center[:,:-1]+gz_center[:,1:]))
    
    gz = jnp.zeros((cfg.Nz+1,cfg.Nr),dtype=c.dtype)
    gz = gz.at[1:-1,:].set((sm[1:,:]-sm[:-1,:])/m.dzc[:,None])
    dr_center = jnp.zeros_like(sm)
    dr_center=dr_center.at[:,1:-1].set((sm[:,2:]-sm[:,:-2])/(m.rc[2:]-m.rc[:-2])[None,:])
    dr_center=dr_center.at[:,0].set(0.0) 
    dr_center=dr_center.at[:,-1].set(0.0) 
    gr_at_zf=jnp.zeros_like(gz)
    gr_at_zf=gr_at_zf.at[1:-1,:].set(0.5*(dr_center[:-1,:]+dr_center[1:,:]))
    eps=1e-10 / min(m.min_dr,m.min_dz)
    nr=gr/jnp.sqrt(gr**2+gz_at_rf**2+eps**2)
    nz=gz/jnp.sqrt(gz**2+gr_at_zf**2+eps**2)
    
    
    theta=jnp.deg2rad(cfg.contact_angle_deg)
    weight=jnp.abs(gz_center[:,-1])/jnp.sqrt(gz_center[:,-1]**2+eps**2)
    nr=nr.at[:,-1].set(jnp.cos(theta)*weight)
    kappa=-div_flux(nr,nz,m)
    if cfg.curvature_method == "height":
        kh, valid, _h = height_curvature.column_curvature(c,m)

        kappa=jnp.where(valid[None,:],kh[None,:],kappa)
    elif cfg.curvature_method == "variational":
        kv, valid, _h = height_curvature.variational_curvature(c,m)
        kappa=jnp.where(valid[None,:],kv[None,:],kappa)
    elif cfg.curvature_method == "adaptive_hf":
        kappa,_wet,_dry,_valid_r=height_curvature.adaptive_orientation_curvature(c,m,kappa)
    
    raw_r, raw_z=grad_pressure(c,m)
    raw_z=raw_z.at[-1,:].set(0.0) 
    kr=0.5*(kappa[:,:-1]+kappa[:,1:])
    kz=0.5*(kappa[:-1,:]+kappa[1:,:])
    fsr=jnp.zeros_like(gr).at[:,1:-1].set(cfg.sigma*kr*raw_r[:,1:-1])
    fsz=jnp.zeros_like(gz).at[1:-1,:].set(cfg.sigma*kz*raw_z[1:-1,:])
    return kappa,fsr,fsz


def upwind_theta(ut,urc,uzc,m:Mesh,cfg:Config):
    
    l=jnp.concatenate((-ut[:,:1],ut[:,:-1]),axis=1)
    gamma_r=(cfg.slip_length-(cfg.R-m.rc[-1]))/(cfg.slip_length+(cfg.R-m.rc[-1]))
    gamma_z=(cfg.slip_length-m.dzv[0]/2)/(cfg.slip_length+m.dzv[0]/2)
    r=jnp.concatenate((ut[:,1:],gamma_r*ut[:,-1:]),axis=1)
    dl=jnp.concatenate((2*m.rc[:1],m.drc))
    dr=jnp.concatenate((m.drc,2*(m.cfg.R-m.rc[-1:])))
    d_r=jnp.where(urc>=0,(ut-l)/dl[None,:],(r-ut)/dr[None,:])
    b=jnp.concatenate((gamma_z*ut[:1,:],ut[:-1,:]),axis=0)
    t=jnp.concatenate((ut[1:,:],ut[-1:,:]),axis=0)
    lower=jnp.concatenate((m.dzv[:1],m.dzc))[:,None]
    upper=jnp.concatenate((m.dzc,m.dzv[-1:]))[:,None]
    d_z=jnp.where(uzc>=0,(ut-b)/lower,(t-ut)/upper)
    return -(urc*d_r+uzc*d_z+urc*ut/m.RC)


def advect_velocity(ur,ut,uz,m:Mesh,cfg:Config):
    
    ar=jnp.zeros_like(ur)
    az=jnp.zeros_like(uz)
    ui=ur[:,1:-1]
    wl=0.25*(uz[:-1,:-1]+uz[1:,:-1]+uz[:-1,1:]+uz[1:,1:])
    d_lr=m.rf[1:-1]-m.rf[:-2]
    d_rr=m.rf[2:]-m.rf[1:-1]
    dr_u=jnp.where(ui>=0,(ui-ur[:,:-2])/d_lr[None,:],
                         (ur[:,2:]-ui)/d_rr[None,:])
    gamma_z=(cfg.slip_length-m.dzv[0]/2)/(cfg.slip_length+m.dzv[0]/2)
    up=jnp.concatenate((gamma_z*ur[:1,:],ur,ur[-1:,:]),axis=0)
    lower=jnp.concatenate((m.dzv[:1],m.dzc))[:,None]
    upper=jnp.concatenate((m.dzc,m.dzv[-1:]))[:,None]
    d_zu=jnp.where(wl>=0,(ui-up[:-2,1:-1])/lower,
                         (up[2:,1:-1]-ui)/upper)
    vtheta=0.5*(ut[:,:-1]+ut[:,1:])
    ar=ar.at[:,1:-1].set(-ui*dr_u-wl*d_zu+vtheta**2/m.rf[None,1:-1])

    wi=uz[1:-1,:]
    vf=0.25*(ur[:-1,:-1]+ur[:-1,1:]+ur[1:,:-1]+ur[1:,1:])
    vl=jnp.concatenate((uz[:,:1],uz[:,:-1]),axis=1)
    gamma_r=(cfg.slip_length-(cfg.R-m.rc[-1]))/(cfg.slip_length+(cfg.R-m.rc[-1]))
    vr=jnp.concatenate((uz[:,1:],gamma_r*uz[:,-1:]),axis=1)
    dl=jnp.concatenate((2*m.rc[:1],m.drc))
    dr=jnp.concatenate((m.drc,2*(m.cfg.R-m.rc[-1:])))
    dr_w=jnp.where(vf>=0,(wi-vl[1:-1,:])/dl[None,:],
                          (vr[1:-1,:]-wi)/dr[None,:])
    dz_w=jnp.where(wi>=0,(wi-uz[:-2,:])/m.dzv[:-1,None],
                          (uz[2:,:]-wi)/m.dzv[1:,None])
    az=az.at[1:-1,:].set(-vf*dr_w-wi*dz_w)
    urc,uzc=averages(ur,uz)
    at=upwind_theta(ut,urc,uzc,m,cfg)
    return ar,at,az


def theta_on_radial_faces(ut, m:Mesh):

    weight=(m.rf[1:-1]-m.rc[:-1])/m.drc
    return ut[:,:-1]+(ut[:,1:]-ut[:,:-1])*weight[None,:]


def stress_divergence(ur,ut,uz,mu,m:Mesh,cfg:Config):

    urc,_=averages(ur,uz)
    trr=2*mu*(ur[:,1:]-ur[:,:-1])/m.dr[None,:]
    ttt=2*mu*urc/m.RC
    tzz=2*mu*(uz[1:,:]-uz[:-1,:])/m.dzv[:,None]
    
    trz=jnp.zeros((m.cfg.Nz+1,m.cfg.Nr+1),dtype=mu.dtype)
    mu_corner=0.25*(mu[:-1,:-1]+mu[1:,:-1]+mu[:-1,1:]+mu[1:,1:])
    trz=trz.at[1:-1,1:-1].set(mu_corner*((ur[1:,1:-1]-ur[:-1,1:-1])/m.dzc[:,None]+
                                    (uz[1:-1,1:]-uz[1:-1,:-1])/m.drc[None,:]))
    
    mub=0.5*(mu[0,:-1]+mu[0,1:])
    trz=trz.at[0,1:-1].set(mub*(ur[0,1:-1]/(cfg.slip_length+m.dzv[0]/2)))
    
    mus=0.5*(mu[:-1,-1]+mu[1:,-1])
    trz=trz.at[1:-1,-1].set(mus*(-uz[1:-1,-1]/(cfg.slip_length+m.cfg.R-m.rc[-1])))
    
    vr=jnp.zeros_like(ur)
    a=(trr[:,1:]-trr[:,:-1])/m.drc[None,:]
    b=(0.5*(trr[:,1:]+trr[:,:-1])-0.5*(ttt[:,1:]+ttt[:,:-1]))/m.rf[None,1:-1]
    c=(trz[1:,1:-1]-trz[:-1,1:-1])/m.dzv[:,None]
    vr=vr.at[:,1:-1].set(a+b+c)

    vz=jnp.zeros_like(uz)
    radial=(m.rf[None,1:]*trz[1:-1,1:]-m.rf[None,:-1]*trz[1:-1,:-1])/m.vr[None,:]
    dzz=(tzz[1:,:]-tzz[:-1,:])/m.dzc[:,None]
    vz=vz.at[1:-1,:].set(radial+dzz)

    
    trt=jnp.zeros_like(ur)
    mu_r=0.5*(mu[:,:-1]+mu[:,1:])
    ut_r=theta_on_radial_faces(ut,m)
    trt=trt.at[:,1:-1].set(mu_r*((ut[:,1:]-ut[:,:-1])/m.drc[None,:]-
                                    ut_r/m.rf[None,1:-1]))
    wall_fraction=cfg.slip_length/(cfg.slip_length+m.cfg.R-m.rc[-1])
    trt=trt.at[:,-1].set(mu[:,-1]*(-ut[:,-1]/(cfg.slip_length+m.cfg.R-m.rc[-1])
                                         - wall_fraction*ut[:,-1]/m.cfg.R))
    tzt=jnp.zeros_like(uz)
    mu_z=0.5*(mu[:-1,:]+mu[1:,:])
    tzt=tzt.at[1:-1,:].set(mu_z*(ut[1:,:]-ut[:-1,:])/m.dzc[:,None])
    tzt=tzt.at[0,:].set(mu[0,:]*ut[0,:]/(cfg.slip_length+m.dzv[0]/2))
    vt=(m.rf[None,1:]**2*trt[:,1:]-m.rf[None,:-1]**2*trt[:,:-1])/(m.RC**2*m.dr[None,:])
    vt=vt+(tzt[1:,:]-tzt[:-1,:])/m.dzv[:,None]
    return vr,vt,vz


def torque_factor(t, cfg: Config):

    tt=jnp.asarray(t,dtype=jnp.float64)
    if cfg.torque_protocol == 'constant':
        return jnp.ones_like(tt)
    if cfg.torque_ramp_s > 0:
        x=jnp.clip((tt-cfg.torque_on_s)/cfg.torque_ramp_s,0.,1.)
        onset=0.5*(1-jnp.cos(jnp.pi*x))
    else:
        onset=jnp.where(tt>=cfg.torque_on_s,1.,0.)
    if cfg.torque_protocol == 'ramp':
        return onset
    if cfg.torque_ramp_s > 0:
        x=jnp.clip((cfg.torque_off_s-tt)/cfg.torque_ramp_s,0.,1.)
        cutoff=0.5*(1-jnp.cos(jnp.pi*x))
    else:
        cutoff=jnp.where(tt<cfg.torque_off_s,1.,0.)
    return onset*cutoff


def torque_acceleration(c, rho, m:Mesh, cfg:Config, t=0.0):
    sr=(m.RC/cfg.force_r)*jnp.exp(-0.5*(m.RC/cfg.force_r)**2)
    sz=jnp.exp(-0.5*(m.ZC/cfg.force_z)**2)
    shape=c*sr*sz
    
    norm=jnp.sum(2*jnp.pi*(m.RC**2)*shape*m.dr[None,:]*m.dzv[:,None])
    return (cfg.torque*torque_factor(t,cfg))*shape / jnp.maximum(norm,1e-30) / rho


def euler_step_with_budget(s: State, m: Mesh, cfg: Config, t=0.0):

    donor = phase_face_values(s.c, s.ur, s.uz)
    if cfg.phase_advection == "geometric":
        donor = geometric_vof.geometric_face_values(s.c,s.ur,s.uz,m,cfg.dt,donor)
    cn = advect_c(s.c, s.ur, s.uz, m, cfg.dt, donor=donor)
    rho_old, _ = materials(s.c, cfg)
    rho, mu = materials(cn, cfg)
    invr, invz = face_inverse_density(rho, m)
    avr, avt, avz = advect_velocity(s.ur, s.ut, s.uz, m, cfg)
    vr, vt, vz = stress_divergence(s.ur, s.ut, s.uz, mu, m, cfg)
    _kappa, fr, fz = curvature_and_capillary(cn, m, cfg)
    pr, pz = grad_pressure(s.p, m)
    if cfg.meridional_advection == 'consistent':

        frmass, fzmass = meridional_transport.primal_mass_flux(
            s.ur,s.uz,donor[0],donor[1],m,cfg)
        ur_adv,uz_adv = meridional_transport.advect_meridional(
            s.ur,s.uz,rho_old,rho,frmass,fzmass,m,cfg.dt)
        theta_face=theta_on_radial_faces(s.ut,m)
        centrifugal=jnp.zeros_like(s.ur).at[:,1:-1].set(theta_face**2/m.rf[None,1:-1])
        ur_star=ur_adv+cfg.dt*(centrifugal+invr*(vr+fr-pr))
        uz_star=uz_adv+cfg.dt*(invz*(vz+fz-pz)-cfg.g)
    else:
        
        ur_star = s.ur+cfg.dt*(avr+invr*(vr+fr-pr))
        uz_star = s.uz+cfg.dt*(avz+invz*(vz+fz-pz)-cfg.g)
    ur_star = ur_star.at[:, 0].set(0.0).at[:, -1].set(0.0)
    uz_star = uz_star.at[0, :].set(0.0)
    uz_star = uz_star.at[-1, :].set(uz_star[-2, :])
    if cfg.drive_mode == 'speed_slip':
        accel, _moment, _power_liquid, _diss_slip, _omega, _power_rotor = (
            stirrer_drive.drive_terms(cn,rho,s.ut,m,cfg,t))
    else:
        accel = torque_acceleration(cn, rho, m, cfg, t)
    fr_ang, fz_ang = angular_advection_fluxes(s.ut, s.ur, s.uz, s.c, m, cfg, donor)
    if cfg.angular_advection == 'conservative':
        q_old = rho_old*m.RC*s.ut
        dq_advection = -div_flux(fr_ang, fz_ang, m)
        q_new = q_old+cfg.dt*(dq_advection+m.RC*(vt+rho*accel))
        ut_star = q_new/(rho*m.RC)
    else:
        
        ut_star = s.ut+cfg.dt*(avt+vt/rho+accel)
    unr, unz, dp = project(ur_star, uz_star, rho, m, cfg)
    sn = State(unr, ut_star, unz, s.p+dp, cn)
    bottom, side, outflux = angular_boundary_budget(s.ut, mu, fz_ang, m, cfg)
    applied = jnp.sum(rho*m.RC*accel*m.cellvol)
    L0 = jnp.sum(rho_old*m.RC*s.ut*m.cellvol)
    L1 = jnp.sum(rho*m.RC*ut_star*m.cellvol)
    integrated = cfg.dt*jnp.stack([applied, bottom, side, outflux])
    residual = (L1-L0) - (integrated[0]+integrated[1]+integrated[2]-integrated[3])
    return sn, jnp.concatenate((integrated, jnp.array([residual])))


def euler_step(s: State, m: Mesh, cfg: Config, t=0.0):
    return euler_step_with_budget(s, m, cfg, t)[0]


def initial_state(m:Mesh,cfg:Config):
    
    c=jnp.clip((cfg.H_water-m.zf[:-1,None])/m.dzv[:,None],0.0,1.0)*jnp.ones((1,cfg.Nr))
    rho,_=materials(c,cfg)
    
    p=cfg.g*(jnp.cumsum((rho*m.dzv[:,None])[::-1,:],axis=0)[::-1,:]-0.5*rho*m.dzv[:,None])
    ur=jnp.zeros((cfg.Nz,cfg.Nr+1),dtype=jnp.float64)
    ut=jnp.zeros((cfg.Nz,cfg.Nr),dtype=jnp.float64)
    uz=jnp.zeros((cfg.Nz+1,cfg.Nr),dtype=jnp.float64)
    return State(ur,ut,uz,p,c)


def initial_from_old(s:State,m:Mesh,cfg:Config):
    if not cfg.initial_flow:
        return s
    filepath=Path(cfg.initial_flow)
    data=np.load(filepath)
    idx=-1 if 't' in data.files and data['utheta'].ndim==3 else None
    r=np.asarray(data['r']);z=np.asarray(data['z'])
    from scipy.interpolate import RegularGridInterpolator
    def interpol(key,rf,zf):
        src=data[key][idx] if idx is not None else data[key]
        f=RegularGridInterpolator((z,r),src,bounds_error=False,fill_value=0.0)
        Z,R=np.meshgrid(zf,rf,indexing='ij')
        value=f(np.column_stack([Z.ravel(),R.ravel()])).reshape(Z.shape)
        return value*(Z<cfg.H_water)
    rr=np.asarray(m.rc); rf=np.asarray(m.rf)
    zc=np.asarray(m.zc); zf=np.asarray(m.zf)
    ur=jnp.asarray(interpol('ur',rf,zc))
    ut=jnp.asarray(interpol('utheta',rr,zc))
    uz=jnp.asarray(interpol('uz',rr,zf))
    ur=ur.at[:,0].set(0).at[:,-1].set(0)
    uz=uz.at[0,:].set(0).at[-1,:].set(0)
    rho,_=materials(s.c,cfg)
    ur,uz,_=project(ur,uz,rho,m,cfg)
    return State(ur,ut,uz,s.p,s.c)


def initial_from_checkpoint(m:Mesh,cfg:Config):
    if not cfg.resume_from:
        return None
    file=Path(cfg.resume_from)
    previous=file.parent/'parameters.json'
    if not previous.exists():
        raise FileNotFoundError(f'Нет исходного parameters.json рядом с {file}')
    earlier=json.loads(previous.read_text(encoding='utf-8'))
    immutable=('torque','R','H_water','H_box','rho_water','rho_air','mu_water',
               'mu_air','sigma','g','Nr','Nz','radial_stretch','dt',
               'z_refine_interface','z_refine_bottom','z_refine_top',
               'r_axis_refine','r_wall_refine','r_axis_width_m','r_wall_width_m',
               'z_refine_width_m','z_refine_bottom_width_m','z_refine_top_width_m',
               'contact_angle_deg','slip_length','force_r','force_z','angular_advection','phase_advection','meridional_advection','curvature_method',
               'drive_mode','drive_profile','drive_sign','coupling_rate_s',
               'bar_radius','bar_center_z','bar_sigma_r','bar_sigma_z',
               'drive_time_offset_s', 'torque_protocol','torque_on_s','torque_off_s','torque_ramp_s')
    for key in immutable:
        if key not in earlier:
            if key=='torque_protocol' and cfg.torque_protocol!='constant':
                raise ValueError('Old checkpoint lacks requested torque protocol')
            if key=='drive_mode' and cfg.drive_mode!='fixed_torque':
                raise ValueError('Old checkpoint has no speed_slip drive state')
            
            
            if key in ('drive_mode','drive_profile','drive_sign','coupling_rate_s',
                       'bar_radius','bar_center_z','bar_sigma_r','bar_sigma_z','drive_time_offset_s', 'torque_protocol','torque_on_s','torque_off_s','torque_ramp_s'):
                continue
            if key=='angular_advection' and cfg.angular_advection=='legacy':
                continue
            raise ValueError(f'Missing parameter in previous run: {key}')
        if earlier[key]!=getattr(cfg,key):
            
            
            if key=='curvature_method' and earlier[key]=='variational' and cfg.curvature_method=='adaptive_hf':
                print('[WARNING] CURVATURE METHOD CHANGE: variational -> adaptive_hf at this checkpoint')
                continue
            
            
            if key=='drive_profile' and Path(str(earlier[key]).replace('\\','/')).name==Path(str(getattr(cfg,key)).replace('\\','/')).name:
                print('[WARNING] Relinked drive CSV by basename; input checksum must be verified by manager.')
                continue
            raise ValueError(f'Несовместимые параметры для продолжения: {key}')
    with np.load(file) as d:
        if not (np.allclose(d['r'],np.asarray(m.rc)) and
                np.allclose(d['z'],np.asarray(m.zc))):
            raise ValueError('Несовместимая сетка checkpoint')
        t=float(d['t'])
        start=int(round(t/cfg.dt))
        if abs(start*cfg.dt-t)>max(1e-12,1e-9*cfg.dt):
            raise ValueError('Время checkpoint не кратно dt')
        st=State(jnp.asarray(d['ur_faces']),jnp.asarray(d['utheta']),
                 jnp.asarray(d['uz_faces']),jnp.asarray(d['p']),
                 jnp.asarray(d['c_water']))
        cumulative = jnp.asarray(d['budget_integrated']) if 'budget_integrated' in d.files else jnp.zeros(5,dtype=jnp.float64)
    print(f'Продолжение из {file} при t = {t} с; M нового расчёта = {cfg.torque}')
    return st,start,cumulative


def calc_metrics(s:State,m:Mesh,cfg:Config,t:float):
    urc,uzc=averages(s.ur,s.uz)
    vol=float(jnp.sum(s.c*m.cellvol))
    div=div_flux(s.ur,s.uz,m)
    rho,_=materials(s.c,cfg)
    v2=urc**2+uzc**2+s.ut**2
    
    Lz=float(jnp.sum(rho*m.RC*s.ut*m.cellvol))
    mub=materials(s.c,cfg)[1][0,:]
    mus=materials(s.c,cfg)[1][:,-1]
    tb=mub*s.ut[0,:]/(cfg.slip_length+m.dzv[0]/2)
    bf=cfg.slip_length/(cfg.slip_length+cfg.R-m.rc[-1])
    ts=mus*(-s.ut[:,-1]/(cfg.slip_length+cfg.R-m.rc[-1])
            -bf*s.ut[:,-1]/cfg.R)
    Mwbottom=float(jnp.sum(-2*jnp.pi*m.rc**2*tb*m.dr))
    Mwside=float(jnp.sum(2*jnp.pi*cfg.R**2*ts*m.dzv))
    hcol=jnp.sum(s.c*m.dzv[:,None],axis=0)
    graph_strict=(s.c[0,:]>.99)&(s.c[-1,:]<.01)&(jnp.max(s.c[1:,:]-s.c[:-1,:],axis=0)<.02)
    if cfg.curvature_method=='adaptive_hf':
        graph,dry=height_curvature.principal_interface_qc(s.c)
        graph=graph|dry
    else:
        graph=graph_strict
        dry=jnp.zeros_like(graph)

    first_air=jnp.argmax(s.c<0.5,axis=0)
    high=jnp.arange(cfg.Nz)[:,None]>first_air[None,:]+2
    orphan_frac=jnp.sum(jnp.where(high,s.c*m.cellvol,0.0))/jnp.maximum(vol,1e-30)
    if cfg.drive_mode == 'speed_slip':
        aa,Mdrive,Pwater,Pslip,omega,Prot = stirrer_drive.drive_terms(s.c,rho,s.ut,m,cfg,t)
        Mdrive=float(Mdrive);Pwater=float(Pwater);Pslip=float(Pslip)
        omega=float(omega);Prot=float(Prot)
    else:
        Mdrive=float(cfg.torque*torque_factor(t,cfg));Pwater=float('nan');Pslip=float('nan')
        omega=float('nan');Prot=float('nan')
    return dict(t=t,
       bar_omega_rad_s=omega,bar_frequency_Hz=omega/(2*math.pi),
       bar_drive_torque_N_m=Mdrive,bar_drive_power_W=Prot,
       bar_power_to_fluid_W=Pwater,bar_slip_dissipation_W=Pslip,
       water_volume_m3=vol,
       apparent_funnel_depth_m=float(hcol[-1]-hcol[0]),
       graph_columns_fraction=float(jnp.mean(graph)),
       legacy_strict_graph_fraction=float(jnp.mean(graph_strict)),
       dry_bottom_columns_fraction=float(jnp.mean(dry)),
       satellite_water_screening_fraction=float(orphan_frac),
       water_column_center_m=float(hcol[0]),
       water_column_wall_m=float(hcol[-1]),
       angular_momentum_kg_m2_s=Lz,
       torque_input_N_m=Mdrive,
       torque_bottom_N_m=Mwbottom,
       torque_side_N_m=Mwside,
       u_max_m_s=float(jnp.sqrt(jnp.max(v2))),
       utheta_max_m_s=float(jnp.max(jnp.abs(s.ut))),
       divergence_linf_1_s=float(jnp.max(jnp.abs(div))),
       pressure_min_Pa=float(jnp.min(s.p)),pressure_max_Pa=float(jnp.max(s.p)),
       kinetic_energy_J=float(jnp.sum(0.5*rho*v2*m.cellvol)),
       water_fraction_min=float(jnp.min(s.c)),water_fraction_max=float(jnp.max(s.c)))


def numerical_limits(s:State,m:Mesh,cfg:Config):
    maxu=float(max(jnp.max(jnp.abs(s.ur)), jnp.max(jnp.abs(s.uz)), 1e-10))
    dx=min(m.min_dr,m.min_dz)
    advection=0.25*dx/maxu
    nu=max(cfg.mu_water/cfg.rho_water,cfg.mu_air/cfg.rho_air)
    diffusion=0.10*dx**2/nu
    
    capillary=(0.25*math.sqrt((cfg.rho_water+cfg.rho_air)*dx**3/
                                (2*math.pi*cfg.sigma))) if cfg.sigma>0 else float('inf')
    return dict(advection=advection, diffusion=diffusion, capillary=capillary)


PROGRESS_EVERY_S=20


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


def progress_bar(fraction, width=20):
    fraction=0.0 if fraction<0 else 1.0 if fraction>1 else fraction
    filled=int(fraction*width)
    if 0<fraction<1 and filled==0:
        filled=1
    if fraction<1 and filled==width:
        filled=width-1
    return '█'*filled+'░'*(width-filled)


def atomic_write_json(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def format_progress_line(payload):
    rate=payload.get('steps_per_s')
    rate_txt=f'{rate:.1f} шаг/с' if isinstance(rate,(int,float)) and rate else 'скорость уточняется'
    divv=payload.get('divergence_linf_1_s')
    health=f"  div={divv:.1e}" if isinstance(divv,(int,float)) and math.isfinite(divv) else ''
    return (f"[ПРОГРЕСС] {payload['series']} {payload['case']} "
            f"[{progress_bar(payload['fraction'])}] {payload['fraction']*100:5.1f}%  "
            f"t={payload['t_s']:.2f}/{payload['t_end_s']:.0f}с  "
            f"шаг {payload['step']}/{payload['nsteps']}  {rate_txt}  "
            f"прошло {format_duration(payload['wall_s'])}  "
            f"осталось {format_duration(payload.get('eta_s'))}  "
            f"{payload['phase']}  блок {payload['block_age_s']:.0f} с{health}")


class ProgressReporter:

    def __init__(self, cfg, path, nsteps, nstart):
        self.cfg=cfg
        self.path=Path(path)
        self.nsteps=int(nsteps)
        self.nstart=int(nstart)
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.wall0=time.perf_counter()
        self.block_started=self.wall0
        self.step=int(nstart)
        self.phase='подготовка'
        self.metrics={}
        self.series='?'
        self.case=self.path.parent.name
        for part in self.path.parts:
            if len(part)>=3 and part[0] in 'ABCD' and part[1:3].isdigit():
                self.series=part
        self.thread=threading.Thread(target=self._loop, name='vof-progress', daemon=True)

    def _payload(self):
        step=self.step
        wall=time.perf_counter()-self.wall0
        done=step-self.nstart
        rate=(done/wall) if done>0 and wall>0 else None
        left=self.nsteps-step
        eta=(left/rate) if rate else None
        frac=0.0 if self.nsteps<=0 else min(1.0, step/self.nsteps)
        payload={
            'series':self.series,
            'case':self.case,
            'pid':os.getpid(),
            'step':step,
            'nsteps':self.nsteps,
            't_s':step*self.cfg.dt,
            't_end_s':self.cfg.t_end,
            'fraction':frac,
            'steps_per_s':rate if rate is not None and math.isfinite(rate) else None,
            'eta_s':eta if eta is not None and math.isfinite(eta) else None,
            'wall_s':wall,
            'phase':self.phase,
            'block_age_s':time.perf_counter()-self.block_started,
            'alive':True,
            'updated_unix':time.time(),
        }
        payload.update(self.metrics)
        return payload

    def publish(self, print_line=True):
        with self.lock:
            payload=self._payload()
            text=format_progress_line(payload)
            for target in (self.path/'live_progress.json', self.path.parent/'live_progress.json'):
                try:
                    atomic_write_json(target, payload)
                except Exception:
                    pass
        if print_line:
            print(text, flush=True)

    def set_step(self, step):
        with self.lock:
            self.step=int(step)

    def set_phase(self, phase):
        with self.lock:
            self.phase=phase
            self.block_started=time.perf_counter()

    def note_metrics(self, metrics):
        kept={}
        for key in ('divergence_linf_1_s','relative_water_volume_change','u_max_m_s'):
            if key in metrics:
                kept[key]=metrics[key]
        with self.lock:
            self.metrics=kept

    def start(self):
        self.publish(print_line=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.wait(PROGRESS_EVERY_S):
            self.publish(print_line=True)

    def close(self, phase):
        self.set_phase(phase)
        if phase=='готово':
            with self.lock:
                self.step=self.nsteps
        self.publish(print_line=True)
        self.stop.set()
        self.thread.join(timeout=1)


def save_snapshot(path:Path,s:State,m:Mesh,t:float,budget=None):
    np.savez_compressed(path,t=np.asarray(t),r=np.asarray(m.rc),rf=np.asarray(m.rf),
       z=np.asarray(m.zc),zf=np.asarray(m.zf),
       ur_faces=np.asarray(s.ur),utheta=np.asarray(s.ut),uz_faces=np.asarray(s.uz),
       p=np.asarray(s.p),c_water=np.asarray(s.c),
       budget_integrated=np.asarray(budget if budget is not None else jnp.zeros(5)))


def run(cfg:Config):
    cfg.validate()
    path=Path(cfg.out)
    path.mkdir(parents=True,exist_ok=True)

    m=Mesh(cfg)
    md=stirrer_drive.make_driver(m,cfg)
    (path/'drive_metadata.json').write_text(json.dumps(md,ensure_ascii=False,indent=2),encoding='utf-8')
    if cfg.drive_mode=='speed_slip':
        print('EXPERIMENTAL omega profile:',md)
        print('WARNING: coupling_rate_s is NOT independently identified; predicted funnel is conditional.')
        if np.isfinite(md.get('aliased_fraction',np.nan)) and md['aliased_fraction']>0:
            print('WARNING: rotational sampling alias possible (Nyquist).')
    if cfg.resume_from and cfg.initial_flow:
        raise ValueError('Используйте либо --resume-from, либо --initial-flow')
    saved=initial_from_checkpoint(m,cfg)
    metadata_file = path/'parameters.json'
    if cfg.resume_from and Path(cfg.resume_from).resolve().parent == path.resolve():
        
        
        archive = path/'parameters_previous.json'
        if metadata_file.exists() and not archive.exists():
            archive.write_bytes(metadata_file.read_bytes())
    metadata_file.write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding='utf-8')
    if saved is None:
        state=initial_from_old(initial_state(m,cfg),m,cfg)
        nstart=0
        budget=jnp.zeros(5,dtype=jnp.float64)
    else:
        state,nstart,budget=saved
    nsteps=math.ceil(cfg.t_end/cfg.dt)
    if nsteps<nstart:
        raise ValueError('t_end должен быть больше или равен времени продолжения')
    dt_limits=numerical_limits(state,m,cfg)
    if cfg.dt>min(dt_limits.values()):
        raise ValueError(f"dt={cfg.dt:g} превышает предварительную границу устойчивости: {dt_limits}. "
                         "Уменьшите --dt; это НЕ автоматический подбор dt.")
    print(f"backend={jax.default_backend()} | grid={cfg.Nz}x{cfg.Nr} | "
          f"min dr={m.min_dr:.3g} m | dz_min={m.min_dz:.3g} m, "
          f"dz_max={float(jnp.max(m.dzv)):.3g} m")
    print('dt guide:',dt_limits,'| chosen:',cfg.dt)
    print('ВНИМАНИЕ: это исследовательский прототип; точность межфазной динамики не подтверждена.')
    if cfg.curvature_method=='adaptive_hf':
        print('[WARNING] HYBRID H(r)/A(z) CAPILLARITY IS AN EXPERIMENTAL NUMERICAL BRANCH')
    if cfg.allow_unverified_bottom_contact:
        print('[WARNING] THREE-PHASE CONTACT DYNAMICS NOT VALIDATED; RESULTS BEYOND CONTACT ARE RESEARCH ONLY')
    
    
    @jit
    def step(state_budget, t):
        state0, old_budget = state_budget
        state1, delta_budget = euler_step_with_budget(state0,m,cfg,t)
        return state1, old_budget+delta_budget
    @jit
    def advance_block(state_budget, start_step):
        
        def one(i, pair):
            return step(pair, (start_step+i)*cfg.dt)
        return lax.fori_loop(0, cfg.block_steps, one, state_budget)
    metrics=[]
    initial_volume=float(jnp.sum(state.c*m.cellvol))
    def capture(n):
        t=n*cfg.dt
        jax.block_until_ready(state.ur)
        d=calc_metrics(state,m,cfg,t)
        b=np.asarray(budget)
        b_names=('input','bottom','side','top_outflow','local_residual')
        for i,name in enumerate(b_names):
            d['integrated_'+name+'_kg_m2_s']=float(b[i])
        d['budget_expected_dL_kg_m2_s']=float(b[0]+b[1]+b[2]-b[3])
        d['budget_closure_residual_kg_m2_s']=float(b[4])
        d['relative_water_volume_change']=(d['water_volume_m3']/initial_volume-1)
        metrics.append(d)
        reporter.note_metrics(d)
        if cfg.fail_divergence_above and d['divergence_linf_1_s'] > cfg.fail_divergence_above:
            raise FloatingPointError(
                f"Pressure projection failed QC: div={d['divergence_linf_1_s']:.3e} 1/s "
                f"> {cfg.fail_divergence_above:.3e} 1/s at step {n}. "
                'No checkpoint of the failed state was saved. Inspect the pressure solve.')

        if (cfg.curvature_method=='adaptive_hf' and d['dry_bottom_columns_fraction']>0 and
                not cfg.allow_unverified_bottom_contact):
            save_snapshot(path/f'FAILED_BOTTOM_CONTACT_{n:07d}.npz',state,m,t,budget=budget)
            raise FloatingPointError(
                f'BOTTOM CONTACT: {d["dry_bottom_columns_fraction"]:.3f} radial columns have air at the bottom. '
                'This requires an independently validated wetting/contact-line model. '
                'Last verified checkpoint is preserved; no unverified state is promoted.')
        if (cfg.curvature_method=='adaptive_hf' and cfg.fail_orphan_fraction_above and
                d['satellite_water_screening_fraction'] > cfg.fail_orphan_fraction_above):
            save_snapshot(path/f'FAILED_SATELLITES_{n:07d}.npz',state,m,t,budget=budget)
            raise FloatingPointError(
                f'SATELLITE WATER QC: fraction {d["satellite_water_screening_fraction"]:.3e} '
                f'> {cfg.fail_orphan_fraction_above:.3e}. Candidate saved separately.')
        if cfg.fail_graph_fraction_below and d['graph_columns_fraction'] < cfg.fail_graph_fraction_below:
            save_snapshot(path/f'FAILED_INTERFACE_{n:07d}.npz',state,m,t,budget=budget)
            raise FloatingPointError(
                f'Interface-topology QC: only {d["graph_columns_fraction"]:.3f} of radial columns remain a single h(r). '
                f'Required >= {cfg.fail_graph_fraction_below:.3f}. This curvature model is no longer verified.')
        save_snapshot(path/f'snapshot_{n:07d}.npz',state,m,t,budget=budget)
        print(f"{n:6d}/{nsteps} t={t:.5g} | u_max={d['u_max_m_s']:.4g} | "
              f"div={d['divergence_linf_1_s']:.3g} | dV/V={d['relative_water_volume_change']:.2e} | "
              f"L_res={d['budget_closure_residual_kg_m2_s']:.2e}", flush=True)
        if not all(math.isfinite(float(x)) for x in (d['u_max_m_s'],d['divergence_linf_1_s'],d['pressure_max_Pa'])):
            raise FloatingPointError('В решении возникли NaN/Inf')
    reporter=ProgressReporter(cfg, path, nsteps, nstart)
    reporter.start()
    phase_end='готово'
    try:
        reporter.set_phase('снимок и проверка')
        capture(nstart)
        n=nstart
        first_block=True
        while n<nsteps:
            next_capture=min(((n//cfg.save_every)+1)*cfg.save_every,nsteps)
            reporter.set_phase('компиляция и счёт шагов' if first_block else 'счёт шагов')
            if cfg.block_steps>1 and n+cfg.block_steps<=next_capture:
                state,budget=advance_block((state,budget),jnp.asarray(n))
                n+=cfg.block_steps
            else:
                state,budget=step((state,budget),jnp.asarray(n*cfg.dt))
                n+=1
            first_block=False
            reporter.set_step(n)
            if n==next_capture:
                reporter.set_phase('снимок и проверка')
                capture(n)
                lim=numerical_limits(state,m,cfg)
                if cfg.dt>min(lim.values()):
                    raise RuntimeError(f"На шаге {n} нарушено ограничение шага по времени {lim}.")
    except KeyboardInterrupt:
        phase_end='прервано'
        print('Прервано пользователем; последние checkpoint уже сохранены.')
    except Exception:
        phase_end='ошибка'
        raise
    finally:
        reporter.close(phase_end)
        if metrics:
            with (path/'diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=metrics[0].keys());writer.writeheader();writer.writerows(metrics)
    return path


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name,field in Config.__dataclass_fields__.items():
        default=field.default
        if isinstance(default,int): tp=int
        elif isinstance(default,float): tp=float
        else: tp=str
        ap.add_argument('--'+name.replace('_','-'),type=tp,default=default)
    return Config(**vars(ap.parse_args()))


if __name__=='__main__':
    run(parse_args())
