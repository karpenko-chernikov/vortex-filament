

import jax.numpy as jnp


def primal_mass_flux(ur, uz, cr, cz, mesh, cfg):
                                                                           
    rho_r = cfg.rho_air + (cfg.rho_water - cfg.rho_air)*cr
    rho_z = cfg.rho_air + (cfg.rho_water - cfg.rho_air)*cz
    fr = (2*jnp.pi*mesh.rf[None,:]*mesh.dzv[:,None])*ur*rho_r
    fz = (2*jnp.pi*mesh.vr[None,:])*uz*rho_z
    return fr, fz


def primal_mass_residual(rho0, rho1, fr, fz, mesh, dt):
    mass0 = rho0*mesh.cellvol
    mass1 = rho1*mesh.cellvol
    return (mass1-mass0) + dt*(fr[:,1:]-fr[:,:-1]+fz[1:,:]-fz[:-1,:])


def dual_masses(rho, mesh):
    m = rho * mesh.cellvol
    return 0.5*(m[:,:-1]+m[:,1:]), 0.5*(m[:-1,:]+m[1:,:])


def advect_meridional(ur, uz, rho0, rho1, fr, fz, mesh, dt):

    mr0, mz0 = dual_masses(rho0,mesh)
    mr1, mz1 = dual_masses(rho1,mesh)
    
    frr_left = 0.5*(fr[:,:-2]+fr[:,1:-1])
    frr_right = 0.5*(fr[:,1:-1]+fr[:,2:])
    ur_here = ur[:,1:-1]
    vr_left = jnp.where(frr_left >= 0, ur[:,:-2], ur_here)
    vr_right = jnp.where(frr_right >= 0, ur_here, ur[:,2:])
    rz = 0.5*(fz[:,:-1]+fz[:,1:])
    u_below = jnp.concatenate((ur_here[:1,:],ur_here),axis=0)
    u_above = jnp.concatenate((ur_here,ur_here[-1:,:]),axis=0)
    vz_face = jnp.where(rz >= 0, u_below,u_above)
    vz_face = vz_face.at[-1,:].set(jnp.where(rz[-1,:]>=0,ur_here[-1,:],0.0))
    adv_r = (frr_right*vr_right-frr_left*vr_left +
             rz[1:,:]*vz_face[1:,:]-rz[:-1,:]*vz_face[:-1,:])
    pred_ur = (mr0*ur_here-dt*adv_r)/jnp.maximum(mr1,1e-40)
    next_ur = jnp.zeros_like(ur).at[:,1:-1].set(pred_ur)

    
    fzz_low = 0.5*(fz[:-2,:]+fz[1:-1,:])
    fzz_high = 0.5*(fz[1:-1,:]+fz[2:,:])
    uz_here = uz[1:-1,:]
    wz_low = jnp.where(fzz_low>=0,uz[:-2,:],uz_here)
    wz_high = jnp.where(fzz_high>=0,uz_here,uz[2:,:])
    zr = 0.5*(fr[:-1,:]+fr[1:,:])
    wleft = jnp.concatenate((uz_here[:,:1],uz_here),axis=1)
    wright = jnp.concatenate((uz_here,uz_here[:,-1:]),axis=1)
    wr_face = jnp.where(zr>=0,wleft,wright)
    adv_z = (fzz_high*wz_high-fzz_low*wz_low +
             zr[:,1:]*wr_face[:,1:]-zr[:,:-1]*wr_face[:,:-1])
    pred_uz = (mz0*uz_here-dt*adv_z)/jnp.maximum(mz1,1e-40)
    next_uz = jnp.zeros_like(uz).at[1:-1,:].set(pred_uz)
    
    next_uz = next_uz.at[-1,:].set(uz[-1,:])
    return next_ur, next_uz


def dual_transport_residuals(rho0,rho1,fr,fz,mesh,dt):
                                                                          
    er = primal_mass_residual(rho0,rho1,fr,fz,mesh,dt)
    return 0.5*(er[:,:-1]+er[:,1:]), 0.5*(er[:-1,:]+er[1:,:])


def discrete_meridional_kinetic(rho,ur,uz,mesh):
    mr,mz=dual_masses(rho,mesh)
    return 0.5*(jnp.sum(mr*ur[:,1:-1]**2)+jnp.sum(mz*uz[1:-1,:]**2))
