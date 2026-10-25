"""Dense FEM targets with the frozen five-LED calibration. No training here."""
from pathlib import Path
import os,sys,json,time,hashlib,collections
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'solver'))
SHARED=ROOT
import cv2,numpy as np
from scipy.interpolate import PchipInterpolator
from dataclasses import asdict
from fem_regla import Scene,SPHERE,RADIUS,solve
from geometria_docente import scene_from_mask
SIZE=448;DOMAIN=16.
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interiors(xy,scene):
    source=(np.linalg.norm(xy-SPHERE,axis=1)<=RADIUS)|((abs(xy[:,0])<=.015)&(xy[:,1]>=0)&(xy[:,1]<=.104))
    d=xy-np.array([scene.x_cm,scene.y_cm])*.01;a=np.deg2rad(scene.angle_deg)
    u=d[:,0]*np.cos(a)+d[:,1]*np.sin(a);v=-d[:,0]*np.sin(a)+d[:,1]*np.cos(a)
    ruler=(abs(u)<=.085)&(abs(v)<=.0085)&(scene.mode=='ground')
    return source,ruler
def field(solution,xy,scene):
    value=solution.interpolator(xy[:,0],xy[:,1]);valid=~np.ma.getmaskarray(value);phi=np.ma.filled(value,0).astype('float32')
    source,ruler=interiors(xy,scene);phi[source]=1;phi[ruler]=0;valid[source|ruler]=True
    return phi,valid,source,ruler
def calibration(base):
    cal=json.loads((ROOT/'config/calibracion_led.json').read_text());d=np.linspace(5,17,512)
    p=np.asarray(base.interpolator(.0225+d*.01,np.full_like(d,.1225)))
    voltage=cal['A_V']*d**(-cal['gamma'])
    assert np.all(np.diff(p)<0),'Baseline FEM profile must be strictly monotone'
    x=p[::-1];y=voltage[::-1]
    # Ground is a boundary assumption, not an extra LED observation. Keep all
    # original A/gamma coefficients. Only extend outside their supported interval.
    curve=PchipInterpolator(np.r_[0,x],np.r_[0,y],extrapolate=False)
    lut_phi=np.linspace(0,1,4097);high=lut_phi>x[-1];lut=np.empty_like(lut_phi)
    lut[~high]=curve(lut_phi[~high]);slope=float(curve.derivative()(x[-1]))
    lut[high]=y[-1]+slope*(lut_phi[high]-x[-1])
    assert np.all(np.isfinite(lut)) and lut[0]==0 and np.min(np.diff(lut))>=-1e-10
    fit=np.interp(p,lut_phi,lut);error=float(np.max(abs(fit-voltage)))
    assert error<.001,error
    np.savez_compressed(ROOT/'config/transformacion_led.npz',phi=lut_phi.astype('float32'),voltage=lut.astype('float32'),phi_supported=np.array([x[0],x[-1]],'float32'),distance_cm=d,baseline_phi=p,fitted_voltage=voltage)
    report=dict(method='Fixed monotone FEM-amplitude to LED-equivalent-voltage map; PCHIP through dense samples of the ORIGINAL fitted law along the no-ruler reference profile.',
                A_V=cal['A_V'],gamma=cal['gamma'],distance_interval_cm=[5,17],phi_supported=[float(x[0]),float(x[-1])],
                source_equivalent_voltage_extrapolated=float(lut[-1]),ground_voltage=0.,max_inrange_reproduction_error_V=error,
                assumptions=['Original coil and operating conditions; no LED refit.',
                             'Same amplitude remapping is transferred to the grounded-ruler FEM. This transfer is a hypothesis, not a measured calibration with the ruler.',
                             'Ground value 0 is a physical boundary convention, not an additional LED observation; low-amplitude extension is extrapolation.',
                             'Above the fitted amplitude interval, extend with its endpoint tangent; source value is extrapolated and is NOT measured terminal voltage.',
                             'Nonlinear LED remapping is a surrogate visualization. Do not impose Laplace on the remapped voltage or claim its gradient is a measured electric field.'],
                fixed_FEM_domain_scale=DOMAIN,bounds_m=list(base.scene.bounds()))
    save(ROOT/'config/transferencia_calibracion.json',report)
    return report
def main():
    assert not (ROOT/'revision/congelacion_prueba.json').exists(),'Frozen experiment: create a new version before regenerating teachers'
    start=time.monotonic();rows=json.loads((ROOT/'config/manifest.json').read_text());records=[];excluded=[]
    output=ROOT/'datos/docente';output.mkdir(exist_ok=True,parents=True)
    base_scene=Scene(mode='none',domain_scale=DOMAIN);base=solve(base_scene);report=calibration(base)
    pixel=np.stack(np.meshgrid(np.arange(SIZE),np.arange(SIZE)),axis=-1).reshape(-1,2)
    for i,row in enumerate(rows):
        try:
            for k in ['image','mask']:assert sha(SHARED/row[k])==row[k+'_sha256'],row['id']
            mask=cv2.imread(str(SHARED/row['mask']),0)
            scene,transform=scene_from_mask(mask,domain_scale=DOMAIN);origin,basis,scale=transform
            xy=(pixel-origin)@basis*scale+SPHERE;solution=base if scene.mode=='none' else solve(scene)
            phi,valid,source,ruler=field(solution,xy,scene);phi0,valid0,_,_=field(base,xy,base_scene)
            lb=row['letterbox'];content=np.zeros((SIZE,SIZE),bool);content[lb['top']:lb['top']+lb['height'],lb['left']:lb['left']+lb['width']]=True
            valid=(valid&valid0).reshape(SIZE,SIZE)&content;air=valid&~(source|ruler).reshape(SIZE,SIZE)
            assert valid.sum()>500 and air.sum()>500
            path=output/(row['id']+'.npz')
            np.savez_compressed(path,phi=phi.reshape(SIZE,SIZE),phi0=phi0.reshape(SIZE,SIZE),valid=valid.astype('uint8'),air=air.astype('uint8'),rule=ruler.reshape(SIZE,SIZE).astype('uint8'))
            records.append(dict(**row,teacher='datos/docente/'+path.name,teacher_sha256=sha(path),scene=asdict(scene),transform=[np.asarray(v).tolist() for v in transform],
                                valid_pixels=int(valid.sum()),air_pixels=int(air.sum()),content_coverage=float(valid.sum()/content.sum())))
        except (ValueError,RuntimeError,AssertionError) as e:excluded.append(dict(id=row['id'],split=row['split'],camera=row['camera'],reason=str(e)))
        if i%20==0:save(ROOT/'revision/docente.json',dict(status='generating',completed=i+1,total=len(rows),valid=len(records),excluded=len(excluded),seconds=round(time.monotonic()-start,1)))
    save(ROOT/'config/manifest_docente.json',records);save(ROOT/'revision/excluidos_docente.json',excluded)
    save(ROOT/'revision/docente.json',dict(status='complete',images=len(records),excluded=len(excluded),counts=dict(collections.Counter(r['split'] for r in records)),
                                        by_camera_split=dict(collections.Counter(r['camera']+':'+r['split'] for r in records)),seconds=round(time.monotonic()-start,1)))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
