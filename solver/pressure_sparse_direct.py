

import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def make_solver(mesh):
    rc=np.asarray(mesh.rc)
    rf=np.asarray(mesh.rf)
    zf=np.asarray(mesh.zf)
    zc=np.asarray(mesh.zc)
    vr=np.asarray(mesh.vr)
    dz=np.diff(zf)
    drc=np.diff(rc)
    dzc=np.diff(zc)
    Nr=len(rc); Nz=len(zc)
    vol=2*np.pi*vr[None,:]*dz[:,None]
    root=np.sqrt(vol)
    radial_factor=2*np.pi*rf[1:-1][None,:]*dz[:,None]/drc[None,:]
    vertical_factor=2*np.pi*vr[None,:]/dzc[:,None]
    top_factor=2*np.pi*vr/(dz[-1]/2)
    N=Nr*Nz

    def matrix_coefficients(invr,invz):
        gr=radial_factor*invr[:,1:-1]   
        gz=vertical_factor*invz[1:-1,:] 
        gt=top_factor*invz[-1,:]
        diag=np.zeros((Nz,Nr),dtype=np.float64)
        diag[:,:-1]+=gr
        diag[:,1:]+=gr
        diag[:-1,:]+=gz
        diag[1:,:]+=gz
        diag[-1,:]+=gt
        diag=diag/vol
        off_r=-gr/(root[:,:-1]*root[:,1:])
        off_z=-gz/(root[:-1,:]*root[1:,:])
        upper_r=np.zeros((Nz,Nr),dtype=np.float64)
        upper_r[:,:-1]=off_r
        return diag.reshape(-1),upper_r.reshape(-1)[:-1],off_z.reshape(-1)

    def callback(rhs,invr,invz):
        rhs=np.asarray(rhs,dtype=np.float64)
        ir=np.asarray(invr,dtype=np.float64)
        iz=np.asarray(invz,dtype=np.float64)
        diag,up_r,up_z=matrix_coefficients(ir,iz)
        A=diags((up_z,up_r,diag,up_r,up_z),
                offsets=(-Nr,-1,0,1,Nr),shape=(N,N),format='csc')
        b=(rhs*root).reshape(-1)
        y=spsolve(A,b,permc_spec='COLAMD')
        if not np.all(np.isfinite(y)):
            raise FloatingPointError('Sparse direct pressure produced NaN/Inf')
        return (y.reshape(Nz,Nr)/root).astype(np.float64)

    def solve(rhs,invr,invz):
        shape=jax.ShapeDtypeStruct((Nz,Nr),jnp.float64)
        return jax.pure_callback(callback,shape,rhs,invr,invz,vmap_method='sequential')
    return solve, callback, matrix_coefficients
