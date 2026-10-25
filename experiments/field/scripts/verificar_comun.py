from pathlib import Path
import json,numpy as np,torch
from datos import ROOT,Data,load_rows
from sampling import weights
from modelo_factory import Model
from red_potencial import loss

def main():
 torch.set_num_threads(4);rows=[r for r in load_rows() if r['split']=='train'];w=weights(rows);records=[]
 for camera in sorted(set(r['camera'] for r in rows)):
  r=next(r for r in rows if r['camera']==camera);ds=Data([r],train=True,epoch=3,geometry=True);x,a,_=ds[0];x2,b,_=ds[0]
  assert torch.equal(x,x2) and all(torch.equal(a[k],b[k]) for k in a)
  assert all(torch.isfinite(v).all() for v in a.values()) and a['valid'].sum()>100
  # P1 phi0 and raw channel 6 carry the same baseline field. Paired warps must agree in valid air.
  mask=a['air'][0].bool()&a['valid'][0].bool();error=(a['phi0'][0][mask]-a['raw'][6][mask]).abs().max().item();assert error<1e-4,(camera,error)
  records.append(dict(camera=camera,id=r['id'],augmentation_repeated_exactly=True,paired_geometry_max_error=error))
 cal=np.load(ROOT/'config/transformacion_led.npz')['voltage'];r=next(r for r in rows if r['scene']['mode']=='ground');rgb,a,_=Data([r],geometry=True)[0]
 m=Model('dinov3','residual',cal,False).cuda().eval();m.head.context=a['raw'][None].cuda()
 with torch.no_grad():v,p=m(rgb[None].cuda(),True)
 assert torch.isfinite(v).all() and p.min()>=0 and p.max()<=1
 ck=m.checkpoint();n=Model('dinov3','residual',cal,False);n.restore(ck);n.cuda().eval();n.head.context=m.head.context
 with torch.no_grad():vv,pp=n(rgb[None].cuda(),True)
 assert torch.equal(v,vv),'Checkpoint restore changed prediction'
 (ROOT/'revision/verificacion_comun.json').write_text(json.dumps(dict(status='passed',paired_geometry=records,residual_checkpoint_roundtrip=True,condition_probability={c:float(w[[r['scene']['mode']==c for r in rows]].sum()) for c in ['none','ground']}),indent=2))
 print('PASS',flush=True)
if __name__=='__main__':main()
