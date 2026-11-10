

from __future__ import annotations
import jax
import jax.numpy as jnp


def plane_fraction(rlo, rhi, zlo, zhi, rc, zc, nr, nz, alpha):

    dr_area = jnp.maximum(0.5*(rhi**2-rlo**2), 1e-35)
    dz = jnp.maximum(zhi-zlo, 1e-20)
    nzabs=jnp.maximum(jnp.abs(nz),1e-14)
    nre=jnp.where(nz>=0,nr,-nr)
    ae=jnp.where(nz>=0,alpha,-alpha)
    
    a=(ae+nre*rc+nzabs*(zc-zlo))/(nzabs*dz)
    b=-nre/(nzabs*dz)
    b_safe=jnp.where(jnp.abs(b)>1e-13,b,1.0)
    cross0=jnp.clip(-a/b_safe,rlo,rhi)
    cross1=jnp.clip((1.0-a)/b_safe,rlo,rhi)
    edges=jnp.sort(jnp.stack(jnp.broadcast_arrays(rlo,cross0,cross1,rhi),axis=0),axis=0)
    l,h=edges[:-1],edges[1:]
    mid=(l+h)*0.5
    fmid=a+b*mid
    linear=0.5*a*(h**2-l**2)+(b/3.0)*(h**3-l**3)
    filled=0.5*(h**2-l**2)
    measure=jnp.sum(jnp.where(fmid<=0,0.0,jnp.where(fmid>=1,filled,linear)),axis=0)
    fill_z=jnp.where(nz>=0,measure/dr_area,1-measure/dr_area)
    
    nr_safe=jnp.where(jnp.abs(nr)>1e-14,nr,1.0)
    cutoff=jnp.clip(rc+alpha/nr_safe,rlo,rhi)
    fracr=(cutoff**2-rlo**2)/(2*dr_area)
    fill_r=jnp.where(nr>=0,fracr,1-fracr)
    return jnp.clip(jnp.where(jnp.abs(nz)>1e-8,fill_z,fill_r),0.0,1.0)


def normals(c,m):
                                                                 
    grad_r=jnp.zeros_like(c)
    grad_r=grad_r.at[:,1:-1].set((c[:,2:]-c[:,:-2])/(m.rc[2:]-m.rc[:-2])[None,:])
    grad_r=grad_r.at[:,-1].set((c[:,-1]-c[:,-2])/(m.rc[-1]-m.rc[-2]))
    grad_z=jnp.zeros_like(c)
    grad_z=grad_z.at[1:-1,:].set((c[2:,:]-c[:-2,:])/(m.zc[2:]-m.zc[:-2])[:,None])
    grad_z=grad_z.at[0,:].set((c[1,:]-c[0,:])/m.dzc[0])
    grad_z=grad_z.at[-1,:].set((c[-1,:]-c[-2,:])/m.dzc[-1])
    nr=-grad_r
    nz=-grad_z
    length=jnp.sqrt(nr**2+nz**2)
    nr=jnp.where(length>1e-15,nr/jnp.maximum(length,1e-15),0.0)
    nz=jnp.where(length>1e-15,nz/jnp.maximum(length,1e-15),1.0)
    return nr,nz


def reconstruct_dense(c,m,iterations=33):
                                                                                
    nr,nz=normals(c,m)
    xlo=m.rf[:-1][None,:]-m.RC
    xhi=m.rf[1:][None,:]-m.RC
    zlo=m.zf[:-1,None]-m.ZC
    zhi=m.zf[1:,None]-m.ZC
    corners=jnp.stack(jnp.broadcast_arrays(nr*xlo+nz*zlo,nr*xlo+nz*zhi,
                  nr*xhi+nz*zlo,nr*xhi+nz*zhi),axis=0)
    lo=jnp.min(corners,axis=0)-1e-12
    hi=jnp.max(corners,axis=0)+1e-12

    def bisect(_i, bounds):
        lower, upper = bounds
        mid = (lower + upper)*0.5
        f = plane_fraction(m.rf[:-1][None,:],m.rf[1:][None,:],
                           m.zf[:-1,None],m.zf[1:,None],m.RC,m.ZC,nr,nz,mid)
        lower = jnp.where(f<c, mid, lower)
        upper = jnp.where(f>=c, mid, upper)
        return lower, upper
    lo, hi = jax.lax.fori_loop(0, iterations, bisect, (lo, hi))
    alpha=(lo+hi)*0.5
    return nr,nz,alpha


def reconstruct_sparse(c, m, iterations=33, oversubscription=6):

    nr,nz=normals(c,m)
    nrows,ncols=c.shape
    capacity=min(nrows*ncols,oversubscription*ncols)
    active=(c>1e-12)&(c<1-1e-12)
    n_active=jnp.sum(active,dtype=jnp.int32)

    def sparse(_):
        flat_ids=jnp.nonzero(active.reshape(-1), size=capacity, fill_value=0)[0]
        zz=flat_ids//ncols
        rr=flat_ids%ncols
        nr_s=nr[zz,rr]
        nz_s=nz[zz,rr]
        rc_s=m.rc[rr]; zc_s=m.zc[zz]
        rlo=m.rf[rr];rhi=m.rf[rr+1]
        zlo=m.zf[zz];zhi=m.zf[zz+1]
        xlo=rlo-rc_s;xhi=rhi-rc_s
        zlow=zlo-zc_s;zhigh=zhi-zc_s
        corners=jnp.stack((nr_s*xlo+nz_s*zlow,
                            nr_s*xlo+nz_s*zhigh,
                            nr_s*xhi+nz_s*zlow,
                            nr_s*xhi+nz_s*zhigh),axis=0)
        lower=jnp.min(corners,axis=0)-1e-12
        upper=jnp.max(corners,axis=0)+1e-12
        target=c[zz,rr]
        def bisect(_i,bounds):
            lo,hi=bounds
            mid=0.5*(lo+hi)
            f=plane_fraction(rlo,rhi,zlo,zhi,rc_s,zc_s,nr_s,nz_s,mid)
            return jnp.where(f<target,mid,lo), jnp.where(f>=target,mid,hi)
        lo,hi=jax.lax.fori_loop(0,iterations,bisect,(lower,upper))
        a_s=0.5*(lo+hi)
        alpha=jnp.zeros_like(c).reshape(-1).at[flat_ids].set(a_s)
        return alpha.reshape(c.shape)

    def dense(_):
        return reconstruct_dense(c,m,iterations)[2]

    alpha=jax.lax.cond(n_active>capacity,dense,sparse,operand=None)
    return nr,nz,alpha


def reconstruct(c,m,iterations=33):
    return reconstruct_sparse(c,m,iterations)


def geometric_donors(c,ur,uz,m,dt):
                                                                                      
    nr,nz,alpha=reconstruct(c,m)
    
    u=ur[:,1:-1]
    rf=m.rf[None,1:-1]
    positive=u>=0
    donor_r=jnp.where(positive,c[:,:-1],c[:,1:])
    nr_r=jnp.where(positive,nr[:,:-1],nr[:,1:])
    nz_r=jnp.where(positive,nz[:,:-1],nz[:,1:])
    a_r=jnp.where(positive,alpha[:,:-1],alpha[:,1:])
    rc_r=jnp.where(positive,m.RC[:,:-1],m.RC[:,1:])
    zc_r=jnp.where(positive,m.ZC[:,:-1],m.ZC[:,1:])
    
    swept_r=jnp.sqrt(jnp.maximum(rf**2-2*rf*u*dt,1e-35))
    rlo=jnp.where(positive,swept_r,rf)
    rhi=jnp.where(positive,rf,swept_r)
    rlo=jnp.maximum(rlo,jnp.where(positive,m.rf[None,:-2],m.rf[None,1:-1]))
    rhi=jnp.minimum(rhi,jnp.where(positive,m.rf[None,1:-1],m.rf[None,2:]))
    fracr=plane_fraction(rlo,jnp.maximum(rhi,rlo+1e-16),
              m.zf[:-1,None],m.zf[1:,None],rc_r,zc_r,nr_r,nz_r,a_r)
    fracr=jnp.where((donor_r<=1e-12)|(jnp.abs(u)<1e-15),
                     jnp.where(donor_r<=1e-12,0.0,donor_r),fracr)
    fracr=jnp.where(donor_r>=1-1e-12,1.0,fracr)
    cr=jnp.zeros_like(ur).at[:,1:-1].set(fracr)

    
    u=uz[1:-1,:]
    zf=m.zf[1:-1,None]
    positive=u>=0
    donor_z=jnp.where(positive,c[:-1,:],c[1:,:])
    nr_z=jnp.where(positive,nr[:-1,:],nr[1:,:])
    nz_z=jnp.where(positive,nz[:-1,:],nz[1:,:])
    a_z=jnp.where(positive,alpha[:-1,:],alpha[1:,:])
    rc_z=jnp.where(positive,m.RC[:-1,:],m.RC[1:,:])
    zc_z=jnp.where(positive,m.ZC[:-1,:],m.ZC[1:,:])
    zlo=jnp.where(positive,zf-u*dt,zf)
    zhi=jnp.where(positive,zf,zf-u*dt)
    zlo=jnp.maximum(zlo,jnp.where(positive,m.zf[:-2,None],m.zf[1:-1,None]))
    zhi=jnp.minimum(zhi,jnp.where(positive,m.zf[1:-1,None],m.zf[2:,None]))
    fracz=plane_fraction(m.rf[None,:-1],m.rf[None,1:],zlo,
              jnp.maximum(zhi,zlo+1e-16),rc_z,zc_z,nr_z,nz_z,a_z)
    fracz=jnp.where((donor_z<=1e-12)|(jnp.abs(u)<1e-15),
                     jnp.where(donor_z<=1e-12,0.0,donor_z),fracz)
    fracz=jnp.where(donor_z>=1-1e-12,1.0,fracz)
    cz=jnp.zeros_like(uz).at[1:-1,:].set(fracz)
    
    ut=uz[-1,:]
    zlo_top=m.zf[-1]-jnp.maximum(ut,0)*dt
    ftop=plane_fraction(m.rf[:-1],m.rf[1:],zlo_top,
          m.zf[-1]+jnp.zeros_like(ut),m.RC[-1],m.ZC[-1],nr[-1],nz[-1],alpha[-1])
    ftop=jnp.where(ut>1e-15,ftop,0.0)
    ftop=jnp.where(c[-1,:]<=1e-12,0.0,ftop)
    ftop=jnp.where(c[-1,:]>=1-1e-12,1.0,ftop)
    cz=cz.at[-1,:].set(ftop)
    return cr,cz


def bound_fluxes(c,ur,uz,m,dt,donor,high):

    cl_r,cl_z=donor
    hi_r,hi_z=high
    def divergence(fr,fz):
        rad=(m.rf[1:][None,:]*fr[:,1:]-m.rf[:-1][None,:]*fr[:,:-1])/m.vr[None,:]
        return rad+(fz[1:,:]-fz[:-1,:])/m.dzv[:,None]
    clow=c-dt*divergence(ur*cl_r,uz*cl_z)
    
    a_r=(2*jnp.pi*dt*m.dzv[:,None])*m.rf[None,:]*ur*(hi_r-cl_r)
    a_z=(2*jnp.pi*dt)*m.vr[None,:]*uz*(hi_z-cl_z)
    rl=a_r[:,:-1];rr=a_r[:,1:]
    zb=a_z[:-1,:];zt=a_z[1:,:]
    plus=jnp.maximum(rl,0)+jnp.maximum(-rr,0)+jnp.maximum(zb,0)+jnp.maximum(-zt,0)
    minus=jnp.maximum(-rl,0)+jnp.maximum(rr,0)+jnp.maximum(-zb,0)+jnp.maximum(zt,0)
    vol=m.cellvol
    rplus=jnp.clip(jnp.maximum(1-clow,0)*vol/jnp.maximum(plus,1e-40),0,1)
    rminus=jnp.clip(jnp.maximum(clow,0)*vol/jnp.maximum(minus,1e-40),0,1)
    theta_r=jnp.where(a_r[:,1:-1]>=0,
          jnp.minimum(rminus[:,:-1],rplus[:,1:]),
          jnp.minimum(rplus[:,:-1],rminus[:,1:]))
    theta_z=jnp.where(a_z[1:-1,:]>=0,
          jnp.minimum(rminus[:-1,:],rplus[1:,:]),
          jnp.minimum(rplus[:-1,:],rminus[1:,:]))
    fr=cl_r.at[:,1:-1].set(cl_r[:,1:-1]+theta_r*(hi_r[:,1:-1]-cl_r[:,1:-1]))
    fz=cl_z.at[1:-1,:].set(cl_z[1:-1,:]+theta_z*(hi_z[1:-1,:]-cl_z[1:-1,:]))
    
    theta_top=jnp.where(a_z[-1,:]>=0,rminus[-1,:],rplus[-1,:])
    fz=fz.at[-1,:].set(cl_z[-1,:]+theta_top*(hi_z[-1,:]-cl_z[-1,:]))
    return fr,fz


def geometric_face_values(c,ur,uz,m,dt,donor):
    high=geometric_donors(c,ur,uz,m,dt)
    return bound_fluxes(c,ur,uz,m,dt,donor,high)
