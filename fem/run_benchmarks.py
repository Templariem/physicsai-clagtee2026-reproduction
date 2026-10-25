"""Recompute all four Figure 2 cases using explicit P1 assembly.
Run from any directory with Python + NumPy/SciPy/Matplotlib.
Inputs and numerical outputs live in ../experimentos/figura2_P1.
No archived manuscript, model or release is modified.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
R=Path(__file__).resolve().parent
DATA=R/'results/benchmarks';DATA.mkdir(parents=True,exist_ok=True)
FIG=DATA
RECORDS=[]
CURRENT=''

def assembly(nodes,elements,materials,pk,pc,pm,source,tipo_analisis='static'):
    e=np.asarray(elements);assert e.shape[1]==3
    p=nodes[e];x=p[:,:,0];y=p[:,:,1]
    det=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0]);assert np.all(det>0)
    area=det/2
    grad=np.stack([np.stack([y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]],axis=1),np.stack([x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]],axis=1)],axis=2)/det[:,None,None]
    stiff=area[:,None,None]*np.einsum('eik,ejk->eij',grad,grad)
    mass=area[:,None,None]/12*(np.ones((3,3))+np.eye(3))
    ii=np.repeat(e,3,axis=1).ravel();jj=np.tile(e,(1,3)).ravel();shape=(len(nodes),len(nodes))
    dtype=complex if tipo_analisis=='harmonic' or any(np.iscomplexobj(v) for d in [pk,pc,pm,source] for v in d.values()) else float
    def coeff(d):return np.array([d.get(int(m),0) for m in materials],dtype=dtype)
    def mat(local,d):return sp.coo_matrix(((local*coeff(d)[:,None,None]).ravel(),(ii,jj)),shape=shape).tocsr()
    f=np.zeros(len(nodes),dtype=dtype);np.add.at(f,e.ravel(),np.repeat(area*coeff(source)/3,3))
    return mat(stiff,pk),mat(mass,pc),mat(mass,pm),f

def resolver_sistema_maestro(nodes,elements,materials,pk,pc,pm,source,dir_borde,kind,kwargs=None):
    K,C,M,F=assembly(nodes,elements,materials,pk,pc,pm,source,kind)
    omega=(kwargs or {}).get('omega',0.)
    A=K if kind=='static' else K+1j*omega*C-omega**2*M
    b=np.array(sorted(dir_borde));values=np.array([dir_borde[int(i)] for i in b]);rhs=F.copy()
    penalized=A.copy();penalized.setdiag(penalized.diagonal()+np.bincount(b,minlength=len(nodes))*1e15)
    rhs[b]+=1e15*values
    u=spsolve(penalized,rhs)
    free=np.ones(len(nodes),bool);free[b]=False
    # Relative rowwise algebraic residual, excluding penalty equations.
    residual=A@u-F
    scale=abs(A)@abs(u)+abs(F)
    rel=float(np.max(abs(residual[free])/np.maximum(scale[free],1e-30)))
    assert np.isfinite(u).all() and rel<1e-10,rel
    bcerror=float(np.max(abs(u[b]-values)))
    assert bcerror<1e-8,bcerror
    np.savez_compressed(DATA/f'{CURRENT}_solution.npz',nodes=nodes,elements=elements,materials=materials,U=u,boundary_nodes=b,boundary_values=values)
    RECORDS.append(dict(case=CURRENT,element='P1',nodes=len(nodes),triangles=len(elements),nodes_per_element=elements.shape[1],free_row_relative_residual=rel,boundary_max_error=bcerror,kind=kind,omega=omega,coefficients=dict(K=pk,C=pc,M=pm,source=source),mesh_sha256=hashlib.sha256((R/'benchmark_meshes'/f'{CURRENT}.npz').read_bytes()).hexdigest()))
    return u

sigma_al=3.5e7

def solve_dipolo(nodes, elements, materials):
    print("Resolviendo Dipolo Eléctrico...")
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98:
            dir_borde[i] = 0.0
        elif np.sqrt((x + 0.5)**2 + y**2) < 0.21:
            dir_borde[i] = 10.0
        elif np.sqrt((x - 0.5)**2 + y**2) < 0.21:
            dir_borde[i] = -10.0
    prop_K = {1: 1.0}
    U = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, {}, dir_borde, 'static')
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.append(el.tolist())
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, U)
    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, U, -dx, -dy, X, Y,
            "Electric Dipole", 'RdBu_r', "Potential V (V)", 'electric')

def solve_iman(nodes, elements, materials):
    print("Resolviendo Imán Horizontal...")
    for i, el in enumerate(elements):
        pts = nodes[el]
        xc, yc = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        if abs(xc) < 0.601 and abs(yc) < 0.201:
            if yc >= 0.0: materials[i] = 2
            else: materials[i] = 3
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98: dir_borde[i] = 0.0
    prop_K = {1: 1.0, 2: 1.0, 3: 1.0}
    fuentes = {1: 0.0, 2: 500.0, 3: -500.0}
    A = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde, 'static')
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.append(el.tolist())
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, A)
    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, A, dy, -dx, X, Y,
            "Permanent Magnet", 'viridis', "Potential Az (Wb/m)", 'magnetic')

def solve_cables(nodes, elements, materials):
    print("Resolviendo Cables de Corriente...")
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98: dir_borde[i] = 0.0
    prop_K = {1: 1.0, 2: 1.0, 3: 1.0}
    fuentes = {1: 0.0, 2: 800.0, 3: -800.0}
    A = resolver_sistema_maestro(nodes, elements, materials, prop_K, {}, {}, fuentes, dir_borde, 'static')
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.append(el.tolist())
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp = mtri.LinearTriInterpolator(tri_mesh, A)
    Z = interp(X, Y)
    Z = np.nan_to_num(Z, nan=0.0)
    dy, dx = np.gradient(Z, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, A, dy, -dx, X, Y,
            "Parallel Wires", 'coolwarm', "Potential Az (Wb/m)", 'magnetic')

def solve_faraday(nodes, elements, materials):
    print("Resolviendo Ley de Faraday...")
    dir_borde = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) > 1.98 or abs(y) > 1.98: dir_borde[i] = 0.0
    w_ac = 2 * np.pi * 50.0
    mu0 = 4 * np.pi * 1e-7
    prop_K = {1: 1.0/mu0, 2: 1.0/mu0, 3: 1.0/mu0, 4: 1.0/mu0}
    prop_C = {1: 0.0, 2: 0.0, 3: 0.0, 4: sigma_al}
    prop_M = {k: 0.0 for k in [1, 2, 3, 4]}
    fuentes = {1: 0.0, 2: 2000.0, 3: -2000.0, 4: 0.0}
    A_complex = resolver_sistema_maestro(nodes, elements, materials, prop_K, prop_C, prop_M, fuentes, dir_borde, 'harmonic', kwargs={'omega': w_ac})
    xg = np.linspace(-1.95, 1.95, 150); yg = np.linspace(-1.95, 1.95, 150)
    X, Y = np.meshgrid(xg, yg)
    triangles = []
    for el in elements: triangles.append(el.tolist())
    tri_mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interp_A = mtri.LinearTriInterpolator(tri_mesh, np.real(A_complex))
    A_grid = interp_A(X, Y)
    A_grid = np.nan_to_num(A_grid, nan=0.0)
    dy, dx = np.gradient(A_grid, yg[1] - yg[0], xg[1] - xg[0])
    return (nodes, elements, materials, triangles, np.real(A_complex), dy, -dx, X, Y,
            "Faraday Induction", 'coolwarm', "Real Potential Az", 'magnetic')


def main():
    global CURRENT
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':14,'axes.titlesize':21,'axes.titleweight':'bold','pdf.fonttype':42,'mathtext.fontset':'dejavuserif'})
    cases=[]
    for name,fn in [('dipole',solve_dipolo),('magnet',solve_iman),('cables',solve_cables),('faraday',solve_faraday)]:
        CURRENT=name
        with np.load(R/'benchmark_meshes'/f'{name}.npz') as a:
            assert a['elements'].shape[1]==3
            cases.append(fn(a['nodes'].copy(),a['elements'].copy(),a['materials'].copy()))
    fig=plt.figure(figsize=(20,9.5),facecolor='white')
    height=.181*20/9.5
    for col,d in enumerate(cases):
        nodes,elements,materials,triangles,u,vx,vy,X,Y,title,cmap,clabel,kind=d
        tri=mtri.Triangulation(nodes[:,0],nodes[:,1],elements)
        selected=elements[materials>1]
        edges=np.concatenate([nodes[selected[:,[0,1]]],nodes[selected[:,[1,2]]],nodes[selected[:,[2,0]]]]) if len(selected) else np.empty((0,2,2))
        x=.012+col*.243
        for row,y in enumerate([.555,.065]):
            ax=fig.add_axes([x,y,.181,height]);ax.set_aspect('equal');ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.2,1.2);ax.axis('off')
            cf=ax.tricontourf(tri,u,levels=50 if row==0 else 20,cmap=cmap,alpha=1 if row==0 else .15)
            if row==0:
                ax.set_title(title,pad=10)
                cax=fig.add_axes([x+.188,y,.009,height])
                cb=fig.colorbar(cf,cax=cax)
                cb.ax.tick_params(labelsize=13,width=.6,length=3,pad=3)
                label=r'Potential $V$ (V)' if kind=='electric' else (r'Real potential $A_z$ (Wb/m)' if col==3 else r'Potential $A_z$ (Wb/m)')
                cb.set_label(label,fontsize=16,fontweight='bold',labelpad=6)
            else:
                ax.set_title('Electric Field' if kind=='electric' else 'Magnetic Field',fontsize=18,pad=8)
                if kind=='electric':
                    # Do not extend field streamlines through the electrode holes.
                    holes=(np.hypot(X+.5,Y)<=.21)|(np.hypot(X-.5,Y)<=.21)
                    ax.streamplot(X,Y,np.ma.masked_where(holes,vx),np.ma.masked_where(holes,vy),color='black',density=1.5,linewidth=1,arrowsize=1.2)
                else:
                    z=mtri.LinearTriInterpolator(tri,u)(X,Y)
                    ax.contour(X,Y,z,levels=25,colors='black',linewidths=1)
            if len(edges):ax.add_collection(LineCollection(edges,colors='black',linewidths=.3,alpha=.3))
            fig.text(x+.0905,y-.018,f'({chr(97+4*row+col)})',ha='center',va='top',fontsize=17,fontweight='bold')
    fig.canvas.draw()
    bounds=fig.get_tightbbox(fig.canvas.get_renderer()).bounds
    assert bounds[0]>=0 and bounds[0]+bounds[2]<=20 and bounds[1]>=0 and bounds[1]+bounds[3]<=9.5,bounds
    fig.savefig(FIG/'fig2_benchmarks_P1.pdf')
    fig.savefig(FIG/'fig2_benchmarks_P1.png',dpi=210)
    plt.close(fig)
    record=dict(assembly='Explicit P1 constant-gradient stiffness and consistent mass; no Q4 field interpolation',boundary='Same benchmark boundary values, diagonal penalty 1e15',cases=RECORDS,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (DATA/'procedencia.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
