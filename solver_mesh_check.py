from __future__ import annotations
import math, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'solver'))
from two_phase_axisymmetric import Config, Mesh

def check_mesh_stability(height,grid_sizes=(64,80,96),dt_us_values=(20,10,5)):
    p=dict(R=.04,H_box=.09,r_axis_refine=3.,r_wall_refine=2.,r_axis_width_m=.006,
           r_wall_width_m=.005,z_refine_interface=4.,z_refine_width_m=.014,
           z_refine_bottom=3.,z_refine_bottom_width_m=.006,radial_stretch=.5)
    for nr in grid_sizes:
        cfg=Config(Nr=nr,Nz=2*nr,H_water=height,**p)
        m=Mesh(cfg)
        dx=min(m.min_dr,m.min_dz)
        capillary=0.25*math.sqrt((cfg.rho_water+cfg.rho_air)*dx**3/(2*math.pi*cfg.sigma))
        allowed=max(dt_us_values)*1e-6
        print(f'[MESH] {nr}x{2*nr}, min dx={dx*1e3:.4f} mm, capillary dt limit={capillary*1e6:.3f} us',flush=True)
        if allowed>capillary:
            raise RuntimeError(f'Unsafe nominal dt {allowed*1e6:g} us for {nr}x{2*nr}')
    print('[OK] Mesh resolution and preliminary capillary dt limit',flush=True)

if __name__=='__main__':
    check_mesh_stability(.06)
    check_mesh_stability(.045)
