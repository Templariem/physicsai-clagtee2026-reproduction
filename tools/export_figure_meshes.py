"""Reconstruct the recorded teacher meshes, validate against saved targets, export only.

No training dataset, checkpoint or original result is overwritten.
"""
from pathlib import Path
import sys, os, json, hashlib
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]/'experiments/field'
sys.path[:0]=[str(ROOT/'solver'),str(ROOT/'scripts')]
import numpy as np
import gmsh
from fem_regla import Scene, SPHERE, solve
from docente_base import field

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
rows={r['id']:r for r in json.loads((ROOT/'config/manifest_docente.json').read_text())}
selected=json.loads((ROOT/'config/figura_seleccion.json').read_text())
pixel=np.stack(np.meshgrid(np.arange(448),np.arange(448)),axis=-1).reshape(-1,2)
report={'gmsh_version':gmsh.__version__,'mesh_origin':'Reconstructed with the recorded teacher scene and unchanged solver; checked against stored target fields and mesh counts.','cases':[]}
for item in selected:
    r=rows[item['id']];scene=Scene(**r['scene']);s=solve(scene)
    origin,basis,scale=r['transform'];origin=np.asarray(origin);basis=np.asarray(basis)
    xy=(pixel-origin)@basis*scale+SPHERE
    values,_,_,_=field(s,xy,scene)
    saved=np.load(ROOT/r['teacher']);valid=saved['valid'].astype(bool)
    error=float(np.max(np.abs(values.reshape(448,448)[valid]-saved['phi'][valid])))
    same_counts=all(s.metrics[k]==r['mesh_metrics'][k] for k in ['nodes','triangles'])
    assert same_counts and error<1e-6,(r['id'],s.metrics,r['mesh_metrics'],error)
    uv=((s.nodes-SPHERE)/scale)@basis.T+origin
    np.savez_compressed(out/(r['id']+'.npz'),nodes=s.nodes,triangles=s.triangles,
                        pixels=uv,source=s.boundary['source'],ruler=s.boundary['ruler'],outer=s.boundary['outer'])
    report['cases'].append(dict(id=r['id'],scene=r['scene'],transform=r['transform'],nodes=len(s.nodes),triangles=len(s.triangles),counts_match=True,target_phi_max_abs_error=error))
    print(r['id'],len(s.nodes),len(s.triangles),'error',error,flush=True)
report['solver_sha256']=hashlib.sha256((ROOT/'solver/fem_regla.py').read_bytes()).hexdigest()
(out/'procedencia.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPLETE',flush=True)
