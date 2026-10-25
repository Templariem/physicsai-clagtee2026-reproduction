from pathlib import Path
import os,json,hashlib,shutil,sys,time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'solver'))
from fem_regla import Scene,solve,SPHERE
SOURCE=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def make_raw(row,indices,base):
    origin,basis,scale=row['transform'];pix=np.column_stack([indices%448,indices//448])
    xy=(pix-np.asarray(origin))@np.asarray(basis)*scale+SPHERE
    raw=np.zeros((len(xy),10),np.float32);raw[:,:2]=xy
    scene=row['scene'];raw[:,2:4]=[scene['x_cm']*.01,scene['y_cm']*.01]
    angle=np.deg2rad(scene['angle_deg']);raw[:,4:6]=[np.cos(angle),np.sin(angle)];raw[:,9]=scene['mode']=='ground'
    phi=base.interpolator(xy[:,0],xy[:,1]);gx,gy=base.interpolator.gradient(xy[:,0],xy[:,1])
    raw[:,6]=np.ma.filled(phi,0);raw[:,7]=np.ma.filled(gx,0);raw[:,8]=np.ma.filled(gy,0)
    source=(np.linalg.norm(xy-SPHERE,axis=1)<=.0225)|((abs(xy[:,0])<=.015)&(xy[:,1]>=0)&(xy[:,1]<=.104))
    raw[source,6]=1;raw[source,7:9]=0
    return raw
def main():
    assert not (ROOT/'revision/preparacion.json').exists(),'Do not overwrite prepared experiment'
    for name in ['manifest_docente.json','transformacion_led.npz','calibracion_led.json','transferencia_calibracion.json']:
        shutil.copy2(SOURCE/'config'/name,ROOT/'config'/name)
    shutil.copy2(SOURCE/'config/protocolo.json',ROOT/'config/protocolo_directos_original.json')
    rows=json.loads((ROOT/'config/manifest_docente.json').read_text());lut=np.load(ROOT/'config/transformacion_led.npz')
    for row in rows:assert sha(SOURCE/row['teacher'])==row['teacher_sha256'],row['id']
    scene=Scene(mode='none',domain_scale=16.,mesh_mm=1.5);base=solve(scene)
    np.savez_compressed(ROOT/'modelos/base_sin_regla_v2.npz',nodes=base.nodes,triangles=base.triangles,phi=base.phi)
    output=ROOT/'datos/cache';output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(4242);records=[];start=time.monotonic()
    for i,row in enumerate(rows):
        if row['split']=='test':continue
        with np.load(SOURCE/row['teacher']) as z:
            a={k:z[k] for k in ['phi','phi0','valid','air','rule']}
        valid=a['valid'].astype(bool).ravel();ids=np.flatnonzero(valid)
        if row['split']=='train':
            ids=rng.choice(ids,4096,replace=len(ids)<4096)
            right=np.minimum(ids+1,448*448-1);down=np.minimum(ids+448,448*448-1)
            indices=np.stack([ids,right,down],1);raw=make_raw(row,indices.ravel(),base).reshape(-1,3,10)
            edge=np.stack([(ids%448<447)&valid[right],(ids//448<447)&valid[down]],1).astype('float32')
        else:indices=ids;raw=make_raw(row,ids,base);edge=np.empty(0,'float32')
        target=a['phi'].ravel()[indices].astype('float32')
        before=np.interp(a['phi0'],lut['phi'],lut['voltage']);truth=np.interp(a['phi'],lut['phi'],lut['voltage'])
        weights=(1+4*((abs(before-truth)>.1)&a['air'].astype(bool))+9*a['rule']).ravel()[indices].astype('float32')
        path=output/(row['id']+'.npz')
        np.savez_compressed(path,raw=raw,target=target,weights=weights,edge=edge,indices=ids.astype('int32'))
        records.append(dict(id=row['id'],split=row['split'],path=str(path.relative_to(ROOT)),sha256=sha(path)))
        if i%50==0:print('prepare',i+1,len(rows),round(time.monotonic()-start,1),flush=True)
    save(ROOT/'config/cache.json',records)
    save(ROOT/'revision/preparacion.json',dict(status='complete',source=str(SOURCE),teacher_manifest_sha256=sha(SOURCE/'config/manifest_docente.json'),
        calibration_sha256=sha(SOURCE/'config/transformacion_led.npz'),solver_sha256=sha(ROOT/'solver/fem_regla.py'),
        source_solver_sha256=sha(SOURCE/'solver/fem_regla.py'),teacher_hashes_verified=len(rows),train=sum(r['split']=='train' for r in rows),
        val=sum(r['split']=='val' for r in rows),test=sum(r['split']=='test' for r in rows),test_used_for_training_or_selection=False,
        baseline_sha256=sha(ROOT/'modelos/base_sin_regla_v2.npz'),seconds=time.monotonic()-start))
if __name__=='__main__':main()
