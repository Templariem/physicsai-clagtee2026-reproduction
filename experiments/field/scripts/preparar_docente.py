"""Shared refined-mesh FEM teachers, identical for all six training runs."""
from pathlib import Path
import os,sys,time,json,hashlib,collections,multiprocessing
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'solver'))
import cv2,numpy as np
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor
from fem_regla import Scene,SPHERE,solve
from geometria_docente import scene_from_mask
from docente_base import field,calibration
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');t.replace(p)
BASE=None
def init():
    global BASE
    BASE=solve(Scene(mode='none',domain_scale=16,mesh_mm=1.5))
def one(row):
    try:
        for k in ['image','mask']:assert sha(ROOT/row[k])==row[k+'_sha256']
        mask=cv2.imread(str(ROOT/row['mask']),0);scene,transform=scene_from_mask(mask,domain_scale=16)
        scene=replace(scene,mesh_mm=1.5);origin,basis,scale=transform
        pixel=np.stack(np.meshgrid(np.arange(448),np.arange(448)),axis=-1).reshape(-1,2);xy=(pixel-origin)@basis*scale+SPHERE
        solution=BASE if scene.mode=='none' else solve(scene)
        phi,valid,source,ruler=field(solution,xy,scene);phi0,valid0,_,_=field(BASE,xy,BASE.scene)
        lb=row['letterbox'];content=np.zeros((448,448),bool);content[lb['top']:lb['top']+lb['height'],lb['left']:lb['left']+lb['width']]=True
        valid=(valid&valid0).reshape(448,448)&content;air=valid&~(source|ruler).reshape(448,448)
        assert valid.sum()>500 and air.sum()>500
        assert phi.min()>-1e-4 and phi.max()<1.0001
        p=ROOT/'datos/docente'/(row['id']+'.npz')
        np.savez_compressed(p,phi=phi.reshape(448,448),phi0=phi0.reshape(448,448),valid=valid.astype('uint8'),air=air.astype('uint8'),rule=ruler.reshape(448,448).astype('uint8'))
        return dict(**row,teacher=str(p.relative_to(ROOT)),teacher_sha256=sha(p),scene=asdict(scene),
            transform=[np.asarray(v).tolist() for v in transform],valid_pixels=int(valid.sum()),air_pixels=int(air.sum()),
            content_coverage=float(valid.sum()/content.sum()),mesh_metrics=solution.metrics),None
    except (ValueError,RuntimeError,AssertionError) as e:
        return None,dict(id=row['id'],split=row['split'],camera=row['camera'],reason=str(e))
def main():
    assert not (ROOT/'revision/congelacion_prueba.json').exists(),'Frozen experiment'
    assert not (ROOT/'config/manifest_docente.json').exists(),'Teachers already generated'
    start=time.monotonic();rows=json.loads((ROOT/'config/manifest.json').read_text());records=[];excluded=[]
    base=solve(Scene(mode='none',domain_scale=16,mesh_mm=1.5));calibration(base)
    save(ROOT/'revision/malla_docente.json',dict(mesh_mm=1.5,baseline=base.metrics,
        policy='Refined 12.5 mm near-field plateau out to 0.35 m, gradual transition to far boundary over 2 m; solver copied from previous mesh-sensitivity audit.',
        original_LED_A_gamma_and_observations_unchanged=True,amplitude_mapping_recomputed_for_refined_reference=True))
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        for i,(record,error) in enumerate(pool.map(one,rows,chunksize=1)):
            if record:records.append(record)
            else:excluded.append(error)
            if i%10==0:save(ROOT/'revision/docente.json',dict(status='generating',completed=i+1,total=len(rows),valid=len(records),excluded=len(excluded),seconds=round(time.monotonic()-start,1)))
    save(ROOT/'config/manifest_docente.json',records);save(ROOT/'revision/excluidos_docente.json',excluded)
    save(ROOT/'revision/docente.json',dict(status='complete',images=len(records),excluded=len(excluded),
        counts=dict(collections.Counter(r['split'] for r in records)),by_camera_split=dict(collections.Counter(r['camera']+':'+r['split'] for r in records)),
        seconds=round(time.monotonic()-start,1)))
    print((ROOT/'revision/docente.json').read_text(),flush=True)
if __name__=='__main__':main()
