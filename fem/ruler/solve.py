"""Ejecutar el mismo caso físico eligiendo P1 o Q4, sin modelos aprendidos."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fem_regla import Scene,solve
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--elemento',choices=['P1','Q4'],default='P1')
p.add_argument('--sin-regla',action='store_true')
p.add_argument('--x-cm',type=float,default=11)
p.add_argument('--y-cm',type=float,default=14)
p.add_argument('--angulo',type=float,default=90)
p.add_argument('--malla-mm',type=float,default=1.5)
p.add_argument('--dominio',type=float,default=16)
a=p.parse_args()
s=solve(Scene(x_cm=a.x_cm,y_cm=a.y_cm,angle_deg=a.angulo,mode='none' if a.sin_regla else 'ground',mesh_mm=a.malla_mm,domain_scale=a.dominio,element_type=a.elemento))
out=Path(__file__).resolve().parent/'resultados';out.mkdir(exist_ok=True)
name=f'{a.elemento}_{s.scene.mode}'
np.savez_compressed(out/f'{name}.npz',nodes=s.nodes,elements=s.elements,phi=s.phi,element_type=a.elemento)
from dataclasses import asdict
(out/f'{name}.json').write_text(json.dumps(dict(scene=asdict(s.scene),metrics=s.metrics),indent=2))
x,y=np.meshgrid(np.linspace(-.18,.30,321),np.linspace(-.08,.38,301))
u=s.interpolator(x,y)
fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
im=ax.pcolormesh(x*100,y*100,u,vmin=0,vmax=1,cmap='turbo',shading='auto')
ax.set(xlabel='x (cm)',ylabel='y (cm)',title=f'FEM {a.elemento} — potencial normalizado',aspect='equal')
fig.colorbar(im,ax=ax,label='V / V fuente')
fig.savefig(out/f'{name}.png',dpi=170)
print(out/f'{name}.png')
print(json.dumps(s.metrics,indent=2))
