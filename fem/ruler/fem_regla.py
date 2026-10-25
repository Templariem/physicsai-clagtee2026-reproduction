"""FEM P1/Q4 planar: electrodo efectivo y regla únicamente a tierra.

No importa ni modifica el motor de la release estable. Los potenciales son
normalizados V/V_fuente; no son las etiquetas de voltaje de los LED del paper.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import threading
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

GMSH_LOCK = threading.Lock()
SPHERE = np.array([0., .1225])
RADIUS = .0225
COIL = np.array([[-.015,0],[.015,0],[.015,.104],[-.015,.104]])
PROBES = {'frente': [.045,.1225], 'detras': [.18,.1225]}

@dataclass(frozen=True)
class Scene:
    x_cm: float = 11.
    y_cm: float = 14.
    angle_deg: float = 90.
    length_cm: float = 17.
    width_cm: float = 1.7
    mode: str = 'ground'
    mesh_mm: float = 3.
    domain_scale: float = 1.
    element_type: str = "P1"

    def checked(self):
        if self.element_type not in ("P1", "Q4"):raise ValueError("Elemento permitido: P1 o Q4.")
        numeric=[self.x_cm,self.y_cm,self.angle_deg,self.length_cm,self.width_cm,self.mesh_mm,self.domain_scale]
        if not np.isfinite(numeric).all():raise ValueError('Se requieren valores finitos.')
        if self.mode not in ['none','ground']:raise ValueError('Condición eléctrica desconocida.')
        if not 5 <= self.length_cm <= 35 or not 1 <= self.width_cm <= 5:
            raise ValueError('Longitud permitida: 5–35 cm; ancho: 1–5 cm.')
        if not 1 <= self.mesh_mm <= 8 or not 1 <= self.domain_scale <= 64:
            raise ValueError('Malla 1–8 mm; factor de dominio 1–64.')
        if self.mode!='none':
            vertices=self.vertices();xmin,xmax,ymin,ymax=self.bounds()
            if (vertices[:,0].min()<xmin+.004 or vertices[:,0].max()>xmax-.004
                or vertices[:,1].min()<ymin+.004 or vertices[:,1].max()>ymax-.004):
                raise ValueError('La regla debe quedar dentro del dominio con al menos 4 mm de margen.')
            angle=np.deg2rad(self.angle_deg)
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            local=(SPHERE-np.array([self.x_cm,self.y_cm])*.01)@rotation
            nearest=np.clip(local,[-self.length_cm*.005,-self.width_cm*.005],[self.length_cm*.005,self.width_cm*.005])
            if np.linalg.norm(local-nearest)<RADIUS+.003:
                raise ValueError('Malla virtual: se requieren 3 mm de separación de la esfera; es un límite numérico, no una distancia de seguridad.')
            # Separating axis theorem, rectángulo de bobina ampliado 3 mm.
            expanded=COIL+np.array([[-.003,-.003],[.003,-.003],[.003,.003],[-.003,.003]])
            axes=[np.array([1.,0]),np.array([0.,1]),rotation[:,0],rotation[:,1]]
            separated=any((vertices@axis).max()<(expanded@axis).min() or
                          (expanded@axis).max()<(vertices@axis).min() for axis in axes)
            if not separated:raise ValueError('Malla virtual: se requieren 3 mm de separación de la bobina; es un límite numérico, no una distancia de seguridad.')
        return self

    def bounds(self):
        s=self.domain_scale
        return (-.35*s,.35*s,.195-.275*s,.195+.275*s)

    def vertices(self):
        a=np.deg2rad(self.angle_deg);c,s=np.cos(a),np.sin(a)
        points=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array([self.length_cm,self.width_cm])*.005
        return points@np.array([[c,s],[-s,c]])+np.array([self.x_cm,self.y_cm])*.01


def stiffness(nodes, triangles):
    """Ensamblado exacto de ∫ grad(N_i)·grad(N_j) dA para triángulos P1."""
    triangles=np.array(triangles,copy=True)
    p=nodes[triangles]
    twice=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1])
    negative=twice<0
    triangles[negative]=triangles[negative][:,[0,2,1]]
    p=nodes[triangles]
    twice=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1])
    if np.any(twice<=1e-16):raise ValueError('Malla con elementos degenerados.')
    bx=(p[:,[1,2,0],1]-p[:,[2,0,1],1])/twice[:,None]
    by=(p[:,[2,0,1],0]-p[:,[1,2,0],0])/twice[:,None]
    values=twice[:,None,None]*.5*(bx[:,:,None]*bx[:,None,:]+by[:,:,None]*by[:,None,:])
    rows=np.broadcast_to(triangles[:,:,None],values.shape).ravel()
    cols=np.broadcast_to(triangles[:,None,:],values.shape).ravel()
    return sparse.coo_matrix((values.ravel(),(rows,cols)),shape=(len(nodes),len(nodes))).tocsr(),triangles


def shape_q4(local):
    """Funciones bilineales y derivadas en coordenadas naturales, orden CCW."""
    local=np.asarray(local)
    signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    factors=1+local[...,None,:]*signs
    N=.25*factors.prod(axis=-1)
    dN=.25*np.stack([signs[:,0]*factors[...,1],signs[:,1]*factors[...,0]],axis=-2)
    return N,dN


def stiffness_q4(nodes,elements):
    """Q4 isoparamétrico, integración completa Gauss 2x2, coeficiente unitario."""
    p=nodes[elements]
    values=np.zeros((len(elements),4,4))
    for xi in (-1/np.sqrt(3),1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3),1/np.sqrt(3)):
            _,dN=shape_q4([xi,eta])
            J=np.einsum('ai,eij->eaj',dN,p)
            det=np.linalg.det(J)
            if np.any(det<=1e-16):raise ValueError('Q4 con Jacobiano no positivo.')
            grad=np.linalg.solve(J,np.broadcast_to(dN,(len(p),2,4)))
            values+=np.einsum('eai,eaj,e->eij',grad,grad,det)
    rows=np.broadcast_to(elements[:,:,None],values.shape).ravel()
    cols=np.broadcast_to(elements[:,None,:],values.shape).ravel()
    return sparse.coo_matrix((values.ravel(),(rows,cols)),shape=(len(nodes),len(nodes))).tocsr(),elements


class Q4Interpolator:
    """La triangulación solo localiza celdas; potencial y gradiente son Q4."""
    def __init__(self,nodes,elements,phi,triangulation):
        self.nodes,self.elements,self.phi=nodes,elements,phi
        self.finder=triangulation.get_trifinder()

    def evaluate(self,x,y):
        x,y=np.broadcast_arrays(x,y)
        shape=x.shape
        xy=np.column_stack([x.ravel(),y.ravel()])
        found=np.asarray(self.finder(x.ravel(),y.ravel()))
        valid=found>=0
        cell=found[valid]%len(self.elements)
        p=self.nodes[self.elements[cell]]
        local=np.zeros((len(p),2))
        for _ in range(20):
            N,dN=shape_q4(local)
            residual=np.einsum('ei,eij->ej',N,p)-xy[valid]
            J=np.einsum('eai,eij->eaj',dN,p)
            delta=np.linalg.solve(J.transpose(0,2,1),residual[...,None])[...,0]
            local-=delta
            if not len(delta) or np.max(np.abs(delta))<1e-12:break
        N,dN=shape_q4(local)
        residual=np.einsum('ei,eij->ej',N,p)-xy[valid]
        if np.any(np.abs(residual)>1e-9) or np.any(np.abs(local)>1+1e-7):
            raise RuntimeError('No converge la inversión isoparamétrica Q4.')
        J=np.einsum('eai,eij->eaj',dN,p)
        grad=np.linalg.solve(J,dN)
        nodal=self.phi[self.elements[cell]]
        outputs=[]
        for values in (np.einsum('ei,ei->e',N,nodal),*np.einsum('eai,ei->ae',grad,nodal)):
            out=np.zeros(len(xy));out[valid]=values
            outputs.append(np.ma.array(out.reshape(shape),mask=(~valid).reshape(shape)))
        return outputs

    def __call__(self,x,y):return self.evaluate(x,y)[0]
    def gradient(self,x,y):return self.evaluate(x,y)[1:]


def dirichlet(K,fixed,values):
    fixed=np.asarray(fixed,dtype=int);values=np.asarray(values,float)
    free=np.setdiff1d(np.arange(K.shape[0]),fixed)
    u=np.zeros(K.shape[0]);u[fixed]=values
    u[free]=spsolve(K[free][:,free],-K[free][:,fixed]@values)
    return u,free


def mesh(scene):
    # The camera app runs meshing in its own process, separate from Torch.
    import gmsh
    scene.checked()
    with GMSH_LOCK:
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber('General.Terminal',0)
            gmsh.option.setNumber('General.NumThreads',1)
            gmsh.model.add('regla_planar')
            occ=gmsh.model.occ
            xmin,xmax,ymin,ymax=scene.bounds()
            air=occ.addRectangle(xmin,ymin,0,xmax-xmin,ymax-ymin)
            disk=occ.addDisk(*SPHERE,0,RADIUS,RADIUS)
            coil=occ.addRectangle(-.015,0,0,.03,.104)
            source,_=occ.fuse([(2,disk)],[(2,coil)])
            tools=list(source)
            if scene.mode!='none':
                x,y=scene.x_cm*.01,scene.y_cm*.01
                length,width=scene.length_cm*.01,scene.width_cm*.01
                ruler=occ.addRectangle(x-length/2,y-width/2,0,length,width)
                occ.rotate([(2,ruler)],x,y,0,0,0,1,np.deg2rad(scene.angle_deg))
                tools.append((2,ruler))
            surfaces,_=occ.cut([(2,air)],tools)
            occ.synchronize()
            curves=gmsh.model.getBoundary(surfaces,combined=True,oriented=False)
            gmsh.option.setNumber('Mesh.Algorithm',6)
            gmsh.option.setNumber('Mesh.MeshSizeMin',scene.mesh_mm*.001)
            gmsh.option.setNumber('Mesh.MeshSizeMax',.025*scene.domain_scale)
            gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
            gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
            gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
            conductor_curves=[]
            for dim,tag in curves:
                cx,cy,_=occ.getCenterOfMass(dim,tag)
                outer=(abs(cx-xmin)<1e-8 or abs(cx-xmax)<1e-8 or abs(cy-ymin)<1e-8 or abs(cy-ymax)<1e-8)
                if not outer:conductor_curves.append(tag)
            gmsh.model.mesh.field.add('Distance',1)
            gmsh.model.mesh.field.setNumbers(1,'CurvesList',conductor_curves)
            gmsh.model.mesh.field.setNumber(1,'Sampling',100)
            gmsh.model.mesh.field.add('Threshold',2)
            for key,value in {'InField':1,'SizeMin':scene.mesh_mm*.001,'SizeMax':.0125,'DistMin':.012,'DistMax':.10}.items():
                gmsh.model.mesh.field.setNumber(2,key,value)
            gmsh.model.mesh.field.add('Threshold',3)
            for key,value in {'InField':1,'SizeMin':.0125,'SizeMax':.025*scene.domain_scale,'DistMin':.35,'DistMax':2.}.items():
                gmsh.model.mesh.field.setNumber(3,key,value)
            gmsh.model.mesh.field.add('MathEval',4)
            gmsh.model.mesh.field.setString(4,'F','F2 + F3 - 0.0125')
            gmsh.model.mesh.field.setAsBackgroundMesh(4)
            gmsh.option.setNumber('Mesh.ElementOrder',1)
            gmsh.option.setNumber('Mesh.RecombineAll',0)
            gmsh.option.setNumber('Mesh.SubdivisionAlgorithm',1 if scene.element_type=='Q4' else 0)
            gmsh.model.mesh.generate(2)
            tags,coords,_=gmsh.model.mesh.getNodes()
            points=np.asarray(coords).reshape(-1,3)[:,:2]
            index={int(t):i for i,t in enumerate(tags)}
            kind, width = (2,3) if scene.element_type=='P1' else (3,4)
            types, _, _ = gmsh.model.mesh.getElements(2)
            if set(types) != {kind}:raise RuntimeError(f'Malla no homogénea: {types}')
            _,connectivity=gmsh.model.mesh.getElementsByType(kind)
            triangles=np.array([index[int(t)] for t in connectivity],int).reshape(-1,width)
            active=np.unique(triangles);mapping=np.full(len(points),-1,int);mapping[active]=np.arange(len(active))
            boundaries={'outer':set(),'source':set(),'ruler':set()}
            a=np.deg2rad(scene.angle_deg);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            for _,tag in curves:
                ct,_,_=gmsh.model.mesh.getNodes(1,tag,includeBoundary=True)
                ids=np.unique([index[int(t)] for t in ct]);p=points[ids]
                outer=any(np.allclose(p[:,axis],value,atol=1e-8,rtol=0) for axis,value in [(0,xmin),(0,xmax),(1,ymin),(1,ymax)])
                local=(p-np.array([scene.x_cm,scene.y_cm])*.01)@rotation
                on_ruler=(np.isclose(np.abs(local[:,0]),scene.length_cm*.005,atol=1e-8)
                          | np.isclose(np.abs(local[:,1]),scene.width_cm*.005,atol=1e-8)).all()
                name='outer' if outer else 'ruler' if scene.mode!='none' and on_ruler else 'source'
                boundaries[name].update(mapping[ids][mapping[ids]>=0].tolist())
            boundaries={key:np.array(sorted(value),int) for key,value in boundaries.items()}
            if len(boundaries['source'])<10 or (scene.mode!='none' and len(boundaries['ruler'])<4):
                raise RuntimeError('No se identificaron correctamente las fronteras conductoras.')
            return points[active],mapping[triangles],boundaries
        finally:gmsh.finalize()


class Solution:
    def __init__(self,scene,nodes,triangles,boundary,K,phi,seconds):
        self.scene,self.nodes,self.elements=scene,nodes,triangles
        self.triangles = triangles if scene.element_type=='P1' else np.concatenate([triangles[:,[0,1,2]],triangles[:,[0,2,3]]])
        self.boundary,self.phi=boundary,phi
        self.triangulation=mtri.Triangulation(nodes[:,0],nodes[:,1],self.triangles)
        self.interpolator=(mtri.LinearTriInterpolator(self.triangulation,phi) if scene.element_type=='P1' else Q4Interpolator(nodes,triangles,phi,self.triangulation))
        reactions=K@phi
        air=np.setdiff1d(np.arange(len(nodes)),np.unique(np.concatenate(list(boundary.values()))))
        ruler=boundary['ruler']
        self.metrics=dict(nodes=len(nodes),element_type=scene.element_type,elements=len(triangles),triangles=len(triangles) if scene.element_type=="P1" else 0,solve_total_ms=round(seconds*1000,2),
                          phi_min=float(phi.min()),phi_max=float(phi.max()),
                          air_residual_max=float(np.max(np.abs(reactions[air]))),
                          ruler_charge_normalized=float(reactions[ruler].sum()) if len(ruler) else None,
                          ruler_phi=float(phi[ruler].mean()) if len(ruler) else None,
                          ruler_equipotential_spread=float(np.ptp(phi[ruler])) if len(ruler) else None,
                          total_charge_balance=float(reactions.sum()))

    def probe(self,xy):
        value=self.interpolator(*xy)
        return None if np.ma.is_masked(value) else float(value)

    def grid(self,nx=121,ny=97):
        xmin,xmax,ymin,ymax=self.scene.bounds()
        x=np.linspace(xmin,xmax,nx);y=np.linspace(ymin,ymax,ny)
        xx,yy=np.meshgrid(x,y)
        phi=self.interpolator(xx,yy);gx,gy=self.interpolator.gradient(xx,yy)
        valid=~np.ma.getmaskarray(phi)
        def packed(array):
            return np.where(valid,np.round(np.ma.filled(array,0),6),None).tolist()
        return dict(nx=nx,ny=ny,bounds_cm=[xmin*100,xmax*100,ymin*100,ymax*100],
                    phi=packed(phi),ex=packed(-gx),ey=packed(-gy))


@lru_cache(maxsize=12)
def solve(scene=Scene()):
    start=time.perf_counter()
    nodes,triangles,boundary=mesh(scene)
    K,triangles=(stiffness if scene.element_type=="P1" else stiffness_q4)(nodes,triangles)
    ruler=boundary['ruler'];source=boundary['source'];outer=boundary['outer']
    fixed=np.r_[source,outer,ruler]
    values=np.r_[np.ones(len(source)),np.zeros(len(outer)+len(ruler))]
    phi,_=dirichlet(K,fixed,values)
    return Solution(scene,nodes,triangles,boundary,K,phi,time.perf_counter()-start)


def comparison(scene):
    """Misma fuente, dominio, escala y sondas para comparar sin/con regla."""
    base=solve(Scene(mode='none',mesh_mm=scene.mesh_mm,domain_scale=scene.domain_scale,element_type=scene.element_type))
    hits=solve.cache_info().hits
    result=solve(scene)
    cached=solve.cache_info().hits>hits
    probes={}
    for name,point in PROBES.items():
        before,after=base.probe(point),result.probe(point)
        probes[name]=dict(x_cm=point[0]*100,y_cm=point[1]*100,before=before,after=after,
                          relative_percent=None if before is None or after is None else (after/before-1)*100)
    return dict(scene=asdict(scene),base_metrics=base.metrics,metrics=result.metrics,cached=cached,
                probes=probes,base=base.grid(),perturbed=result.grid(),
                interpretation='FEM planar normalizado V/V_fuente. No medición ni salida validada de los modelos del paper.')
