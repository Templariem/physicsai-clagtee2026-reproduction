from pathlib import Path
import sys,json,time,hashlib,importlib.util
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--element',choices=['P1','Q4'],default='P1');args=parser.parse_args()
R=Path(__file__).resolve().parent;SRC=R;OUT=R/'results'/args.element;FIG=OUT
(OUT/'data').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R))
import solver_fem_2d as solver

def triangles(q):
    return np.vstack([q[:,[0,1,2]],q[:,[0,2,3]]]) if q.shape[1]==4 else q

assembly=solver.ensamblar_sistema_p1

def loadmesh(name):
    a=np.load(SRC/'gmsh_meshes'/name);q=a['elements'];e=triangles(q) if args.element=='P1' else q;m=np.tile(a['materials'],2) if len(e)==2*len(q) else a['materials']
    return a['nodes'],e,m,a

def check():
    n=np.array([[0.,0.],[1,0],[0,1]]);e=np.array([[0,1,2]]);K,C,M,F=assembly(n,e,[1],{1:1},{1:2},{1:3},{1:6})
    assert np.allclose(K.toarray(),[[1,-.5,-.5],[-.5,.5,0],[-.5,0,.5]])
    assert np.allclose(M.toarray(),(np.ones((3,3))+np.eye(3))/8)
    assert np.allclose(F,1)
check()
report={'element_type':args.element,'mesh_policy':'Each existing conformal quadrilateral split along local diagonal 0-2; unchanged vertices, material interfaces and boundary nodes. P1 assembly performed anew; not Q4 interpolation.','checks':'element stiffness, consistent mass and source integral verified','verification':[]}
if args.element=='Q4':report['mesh_policy']='Original conformal quadrilateral connectivity; explicit Q4 2x2 Gauss assembly.'
fields=None
for name in ['subdivision','blossom','transfinite']:
    n,e,m,a=loadmesh('exp_1_mesh_'+name+'.npz');bn=np.unique(np.concatenate([a[k] for k in ['bobina','esfera','izq_sup','superior','derecho','inferior']])).astype(int)
    x,y=n.T;r=np.hypot(x,y-.5);wave=lambda t:np.where(t>0,np.sin(5*t)**3,0)
    exacts=[1000*np.log(5/r)/np.log(25),x*x-y*y,np.exp(-(1+1j)*x),wave(.8-x)]
    vals=[];errors=[];times=[]
    for idx in range(4):
        bc={int(j):exacts[idx][j] for j in bn}
        kind=['static','static','harmonic','transient'][idx];kw={}
        if idx==2:kw={'omega':1}
        if idx==3:bc={int(j):(lambda t,xj=x[j]:float(wave(t-xj))) for j in bn};kw={'dt':.02,'num_pasos':40}
        start=time.perf_counter();v=solver.resolver_sistema_maestro(n,e,m,{1:1},{1:2} if idx==2 else {},{1:1} if idx==3 else {},{},bc,kind,kw)
        if idx==3:v=v[-1]
        times.append(time.perf_counter()-start);errors.append(float(np.linalg.norm(v-exacts[idx])/np.linalg.norm(exacts[idx])));vals.append(v)
    report['verification'].append({'mesh':name,'nodes':len(n),'elements':len(e),'element_type':args.element,'relative_errors':errors,'assembly_and_solve_s':times})
    if name=='transfinite':fields=(n,e,vals);np.savez_compressed(OUT/'verification_fields.npz',nodes=n,elements=e,values=np.stack(vals))
# Original Fig1 slots, palettes and typographic format, now showing actual P1 solutions.
spec=importlib.util.spec_from_file_location('fig1',SRC/'plots/figure1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.OUT=FIG;mod.triangles=lambda e:e if e.shape[1]==3 else triangles(e)
mod.plot_verification(*fields)
# Coil: identical geometry/materials/BC, new P1 assembly.
import exp_3
n,e,m,a=loadmesh('exp_3_mesh_calibration.npz');bc,ground,active=exp_3.boundary_conditions(n)
t0=time.perf_counter();v=exp_3.solve_prior(n,e,m,bc)
np.savez_compressed(OUT/'data/fem_prior.npz',nodes=n,elements=e,materials=m,U_abs=np.abs(v),U_complex=v)
from matplotlib.tri import Triangulation,LinearTriInterpolator
interp=LinearTriInterpolator(Triangulation(n[:,0],n[:,1],triangles(e)),np.abs(v));ds=exp_3.DISTANCE_CM;sample=np.asarray(interp(.0225+ds/100,np.full(5,.1225)))
report['coil']={'nodes':len(n),'elements':len(e),'element_type':args.element,'elapsed_s':time.perf_counter()-t0,'led_FEM_V':sample.tolist(),'attenuation':(exp_3.ASSIGNED_V/sample).tolist(),'source_nodes':int(active.sum()),'ground_nodes':int(ground.sum())}
(OUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
