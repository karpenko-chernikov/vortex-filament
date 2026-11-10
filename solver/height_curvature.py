

from __future__ import annotations
import numpy as np
import jax.numpy as jnp


def prepare(m):
    rf = np.asarray(m.rf, dtype=float)
    nr = len(rf)-1
    mean1 = (2./3.)*(rf[1:]**3-rf[:-1]**3)/(rf[1:]**2-rf[:-1]**2)
    mean2 = .5*(rf[1:]**2+rf[:-1]**2)
    mean4 = (rf[1:]**6-rf[:-1]**6)/(3*(rf[1:]**2-rf[:-1]**2))
    ids=np.empty((nr,3),dtype=np.int32)
    w1=np.zeros((nr,3));w2=np.zeros((nr,3))
    for i in range(nr):
        if i >= nr-2:
            ix=np.array([nr-2,nr-1,-1]); x=mean1[i]
            R=rf[-1]
            aidx=ix[:2]
            mat=np.stack([np.ones(2), mean1[aidx]-R, mean2[aidx]-2*R*mean1[aidx]+R*R],axis=1)
            mat=np.concatenate([mat, np.array([[0.,1.,0.]])],axis=0)
            inv=np.linalg.inv(mat)
            w1[i]=np.array([0.,1.,2*(x-R)])@inv
            w2[i]=np.array([0.,0.,2.])@inv
        elif i <= 1:
            ix=np.array([0,1,2]); x=mean1[i]
            mat=np.stack([np.ones(3),mean2[ix],mean4[ix]],axis=1)
            inv=np.linalg.inv(mat)
            w1[i]=(np.array([0.,2*x,4*x**3])@inv)
            w2[i]=(np.array([0.,2.,12*x*x])@inv)
        else:
            ic=max(1,min(i,nr-2)); ix=np.array([ic-1,ic,ic+1]);x=mean1[i]
            mat=np.stack([np.ones(3),mean1[ix]-x,mean2[ix]-2*x*mean1[ix]+x*x],axis=1)
            inv=np.linalg.inv(mat)
            w1[i]=np.array([0.,1.,0.])@inv
            w2[i]=np.array([0.,0.,2.])@inv
        ids[i]=ix
    
    m.hf_indices=jnp.asarray(np.maximum(ids,0))
    m.hf_use_wall=jnp.asarray(ids[:,2]==-1)
    m.hf_w1=jnp.asarray(w1)
    m.hf_w2=jnp.asarray(w2)
    m.hf_r=jnp.asarray(mean1)

    zc=np.asarray(m.zc,dtype=float)
    nz=len(zc)
    ix_z=np.empty((nz,3),dtype=np.int32)
    w_z1=np.empty((nz,3),dtype=float)
    w_z2=np.empty((nz,3),dtype=float)
    for j in range(nz):
        start=max(0,min(j-1,nz-3))
        inds=np.arange(start,start+3,dtype=np.int32)
        x=zc[j]
        mat=np.stack([np.ones(3), zc[inds]-x,(zc[inds]-x)**2],axis=1)
        inv=np.linalg.inv(mat)
        ix_z[j]=inds
        w_z1[j]=np.array([0.,1.,0.])@inv
        w_z2[j]=np.array([0.,0.,2.])@inv
    m.hf_z_indices=jnp.asarray(ix_z)
    m.hf_z_w1=jnp.asarray(w_z1)
    m.hf_z_w2=jnp.asarray(w_z2)


def column_curvature(c,m):

    h=jnp.sum(c*m.dzv[:,None],axis=0)
    wall_gradient=1./jnp.tan(jnp.deg2rad(m.cfg.contact_angle_deg))
    h3=h[m.hf_indices]
    h3=h3.at[:,2].set(jnp.where(m.hf_use_wall,wall_gradient,h3[:,2]))
    hp=jnp.sum(m.hf_w1*h3,axis=1)
    hpp=jnp.sum(m.hf_w2*h3,axis=1)
    curvature=-hpp/(1+hp**2)**1.5-hp/(m.hf_r*jnp.sqrt(1+hp**2))
    
    valid=(c[0,:]>.99)&(c[-1,:]<.01)&(jnp.max(c[1:,:]-c[:-1,:],axis=0)<.02)
    valid=valid & (h>2*m.min_dz) & (h<m.cfg.H_box-2*m.min_dz)
    return curvature,valid,h


def variational_curvature(c,m):

    h=jnp.sum(c*m.dzv[:,None],axis=0)
    slope=(h[1:]-h[:-1])/m.drc
    fmid=m.rf[1:-1]*slope/jnp.sqrt(1+slope*slope)
    
    
    theta=jnp.deg2rad(m.cfg.contact_angle_deg)
    fwall=m.cfg.R*jnp.cos(theta)
    flux=jnp.concatenate((jnp.zeros((1,),dtype=h.dtype),fmid,
                         jnp.asarray([fwall],dtype=h.dtype)),axis=0)
    k=-(flux[1:]-flux[:-1])/m.vr
    valid=(c[0,:]>.99)&(c[-1,:]<.01)&(jnp.max(c[1:,:]-c[:-1,:],axis=0)<.02)
    valid=valid&(h>2*m.min_dz)&(h<m.cfg.H_box-2*m.min_dz)
    return k,valid,h


def discrete_surface_energy(c,m,sigma):

    h=jnp.sum(c*m.dzv[:,None],axis=0)
    slope=(h[1:]-h[:-1])/m.drc
    theta=jnp.deg2rad(m.cfg.contact_angle_deg)
    wetting=-2*jnp.pi*sigma*m.cfg.R*jnp.cos(theta)*(h[-1]-m.cfg.H_water)
    return 2*jnp.pi*sigma*jnp.sum(m.rf[1:-1]*m.drc*(jnp.sqrt(1+slope*slope)-1))+wetting


def principal_interface_qc(c):

    down=(c[:-1,:]>=0.5)&(c[1:,:]<0.5)
    up=(c[:-1,:]<0.5)&(c[1:,:]>=0.5)
    dn=jnp.sum(down,axis=0)
    un=jnp.sum(up,axis=0)
    wet=(c[0,:]>=0.5)&(c[-1,:]<0.5)&(dn==1)&(un==0)
    dry=(c[0,:]<0.5)&(c[-1,:]<0.5)&(dn==0)&(un==0)
    return wet,dry


def radial_air_core_curvature(c,m):

    water_area=jnp.sum(c*m.vr[None,:],axis=1)
    air_area=jnp.clip(0.5*m.cfg.R**2-water_area,0,0.5*m.cfg.R**2)
    a=jnp.sqrt(2*air_area)
    three=a[m.hf_z_indices]
    ap=jnp.sum(m.hf_z_w1*three,axis=1)
    app=jnp.sum(m.hf_z_w2*three,axis=1)
    k=app/(1+ap*ap)**1.5-1/(jnp.maximum(a,1e-12)*jnp.sqrt(1+ap*ap))
    
    
    monotone=jnp.max(c[:,:-1]-c[:,1:],axis=1)<0.03
    valid=monotone & (a > 1.5*m.min_dr) & (a < m.cfg.R-1.5*m.min_dr)
    return k,valid,a


def adaptive_orientation_curvature(c,m,legacy_kappa):

    kh,old_valid,h=variational_curvature(c,m)
    wet,dry=principal_interface_qc(c)
    
    valid_h=wet & (h > 2*m.min_dz)
    kr,valid_r,a=radial_air_core_curvature(c,m)

    dcr=jnp.zeros_like(c)
    dcr=dcr.at[:,1:-1].set((c[:,2:]-c[:,:-2])/(m.rc[2:]-m.rc[:-2])[None,:])
    dcr=dcr.at[:,0].set((c[:,1]-c[:,0])/(m.rc[1]-m.rc[0]))
    dcr=dcr.at[:,-1].set((c[:,-1]-c[:,-2])/(m.rc[-1]-m.rc[-2]))
    dcz=jnp.zeros_like(c)
    dcz=dcz.at[1:-1,:].set((c[2:,:]-c[:-2,:])/(m.zc[2:]-m.zc[:-2])[:,None])
    dcz=dcz.at[0,:].set((c[1,:]-c[0,:])/m.dzc[0])
    dcz=dcz.at[-1,:].set((c[-1,:]-c[-2,:])/m.dzc[-1])
    wr=dcr*dcr/(dcr*dcr+dcz*dcz+1e-22)
    
    
    wr=(2*wr+jnp.pad(wr[:,1:],((0,0),(0,1)),mode='edge')+
        jnp.pad(wr[:,:-1],((0,0),(1,0)),mode='edge')+
        jnp.pad(wr[1:,:],((0,1),(0,0)),mode='edge')+
        jnp.pad(wr[:-1,:],((1,0),(0,0)),mode='edge'))/6
    vh=valid_h[None,:]
    vr=valid_r[:,None]
    k_h=jnp.broadcast_to(kh[None,:],c.shape)
    k_r=jnp.broadcast_to(kr[:,None],c.shape)
    
    both=vh&vr
    out=jnp.where(both,(1-wr)*k_h+wr*k_r,
          jnp.where(vh,k_h,jnp.where(vr,k_r,legacy_kappa)))
    return out,wet,dry,valid_r
