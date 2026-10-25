"""Verificación independiente, regresión P1 y sensibilidad P1/Q4 del caso con regla."""
import importlib.util,json,hashlib,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from fem_regla import Scene,solve,stiffness,stiffness_q4,dirichlet,Q4Interpolator
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'resultados';OUT.mkdir(exist_ok=True)
old=ROOT/'reference_p1.py'
spec=importlib.util.spec_from_file_location('original',old)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
report={'original_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'manufactured':[],'scenes':[]}
for kind in ('P1','Q4'):
    errors=[]
    for n in (8,16,32):
        x,y=np.meshgrid(np.linspace(0,1,n+1),np.linspace(0,1,n+1))
        nodes=np.column_stack([x.ravel(),y.ravel()])
        # Distorsión interior suave: prueba también el mapeo isoparamétrico.
        nodes[:,0]+=.025*np.sin(np.pi*nodes[:,0])*np.sin(2*np.pi*nodes[:,1])
        cells=np.array([[j*(n+1)+i,j*(n+1)+i+1,(j+1)*(n+1)+i+1,(j+1)*(n+1)+i] for j in range(n) for i in range(n)])
        triangles=np.concatenate([cells[:,[0,1,2]],cells[:,[0,2,3]]])
        elements=triangles if kind=='P1' else cells
        K,_=(stiffness if kind=='P1' else stiffness_q4)(nodes,elements)
        fixed=np.flatnonzero((x.ravel()==0)|(x.ravel()==1)|(y.ravel()==0)|(y.ravel()==1))
        affine=1+2*nodes[:,0]-3*nodes[:,1]
        u,_=dirichlet(K,fixed,affine[fixed])
        patch=float(np.max(np.abs(u-affine)));assert patch<1e-10
        exact=lambda x,y:np.exp(x)*np.cos(y) # Laplace(u)=0
        u,free=dirichlet(K,fixed,exact(*nodes[fixed].T))
        triang=mtri.Triangulation(*nodes.T,triangles)
        interpolator=mtri.LinearTriInterpolator(triang,u) if kind=='P1' else Q4Interpolator(nodes,cells,u,triang)
        xx,yy=np.meshgrid(np.linspace(.001,.999,121),np.linspace(.001,.999,121))
        err=float(np.sqrt(np.mean((interpolator(xx,yy)-exact(xx,yy))**2)))
        symmetry=K-K.T
        assert np.max(np.abs(symmetry.data),initial=0)<1e-12
        assert np.max(np.abs((K@u)[free]))<1e-10
        # Valores y gradientes afines evaluados fuera de nodos, incluidos puntos externos.
        if kind=='Q4':
            interp=Q4Interpolator(nodes,cells,affine,triang)
            gx,gy=interp.gradient(xx,yy)
            assert np.max(np.abs(interp(xx,yy)-(1+2*xx-3*yy)))<1e-10
            assert np.max(np.abs(gx-2))<1e-9 and np.max(np.abs(gy+3))<1e-9
            assert np.ma.is_masked(interp(-1,-1))
        errors.append(err)
        report['manufactured'].append(dict(element=kind,n=n,nodes=len(nodes),rms=err,affine_max=patch))
    assert errors[2]<errors[1]<errors[0]
    assert errors[0]/errors[1]>3 and errors[1]/errors[2]>3
print('Pruebas analíticas aprobadas',flush=True)
xx,yy=np.meshgrid(np.linspace(-.12,.24,181),np.linspace(-.04,.30,171))
for mode in ('none','ground'):
    maps={}
    for h in (3.,1.5,1.):
        for kind in ('P1','Q4'):
            scene=Scene(mode=mode,domain_scale=16,mesh_mm=h,element_type=kind)
            sol=solve(scene)
            m=sol.interpolator(xx,yy)
            maps[kind,h]=m
            assert sol.metrics['air_residual_max']<1e-10
            assert np.all(sol.phi[sol.boundary['source']]==1)
            assert np.all(sol.phi[sol.boundary['outer']]==0)
            assert np.all(sol.phi[sol.boundary['ruler']]==0)
            row=dict(mode=mode,mesh_mm=h,**sol.metrics)
            report['scenes'].append(row)
            print(mode,h,kind,sol.metrics,flush=True)
            if kind=='P1' and h==1.5:
                baseline=original.solve(original.Scene(mode=mode,domain_scale=16,mesh_mm=h))
                assert np.array_equal(sol.nodes,baseline.nodes)
                assert np.array_equal(sol.elements,baseline.triangles)
                difference=float(np.max(np.abs(sol.phi-baseline.phi)))
                assert difference==0
                row['original_p1_max_difference']=difference
    for row in report['scenes']:
        if row['mode']!=mode:continue
        kind,h=row['element_type'],row['mesh_mm']
        delta=maps[kind,h]-maps[kind,1.]
        row['rms_vs_1mm_same_family']=float(np.sqrt(np.ma.mean(delta**2)))
        delta=maps['P1',h]-maps['Q4',h]
        row['p1_q4_rms_same_requested_h']=float(np.sqrt(np.ma.mean(delta**2)))
        row['p1_q4_max_same_requested_h']=float(np.ma.max(np.abs(delta)))
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,kind in zip(axs[:2],('P1','Q4')):
        im=ax.pcolormesh(xx*100,yy*100,maps[kind,1.5],vmin=0,vmax=1,cmap='turbo',shading='auto')
        ax.set_title(kind+' — potencial normalizado');ax.set_aspect('equal');ax.set_xlabel('x (cm)');ax.set_ylabel('y (cm)')
    fig.colorbar(im,ax=axs[:2],label='V / V fuente')
    delta=maps['Q4',1.5]-maps['P1',1.5]
    lim=float(np.ma.max(np.abs(delta)))
    im=axs[2].pcolormesh(xx*100,yy*100,delta,cmap='coolwarm',vmin=-lim,vmax=lim,shading='auto')
    axs[2].set_title('Q4 − P1');axs[2].set_aspect('equal');axs[2].set_xlabel('x (cm)')
    fig.colorbar(im,ax=axs[2]);fig.savefig(OUT/f'comparacion_{mode}.png',dpi=170);plt.close(fig)
report['notes']=['Q4: subdivisión completa Gmsh; el mismo tamaño solicitado no implica los mismos nodos ni coste.', 'Convergencia analítica verificada; sensibilidad del caso real limitada a dos escenas y tres tamaños locales, no equivalencia universal.', 'No se modificó la calibración LED ni se entrenaron modelos.']
report['status']='passed'
(OUT/'verificacion.json').write_text(json.dumps(report,indent=2))
print('VERIFICACION COMPLETA',flush=True)
