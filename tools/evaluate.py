"""Recompute all paper accuracy metrics, optionally rerun the eight RGB checkpoints.
Writes only runs/evaluation; never trains or replaces published checkpoints/results.
"""
from pathlib import Path
import sys,argparse,json,csv,gc
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/field';sys.path.insert(0,str(F/'scripts'))
from datos import load_rows
from metricas import per_image,summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--infer',action='store_true');a=p.parse_args();out=ROOT/'runs/evaluation';out.mkdir(parents=True,exist_ok=True)
 rows=[r for r in load_rows() if r['split']=='test'];cal=np.load(F/'config/transformacion_led.npz');mean=np.load(F/'modelos/mapa_promedio_train.npz')['voltage'];summary=[];comparisons={}
 expected={r['model']:r for r in csv.DictReader((F/'resultados/tabla_comparacion.csv').open())}
 for name in ['physical']+[k+'_'+arm for k in ['dinov3','ijepa'] for arm in ['frozen','partial','full','residual']]:
  records=[]
  members=None
  if a.infer and name=='physical':
   import torch
   from red_fisica import PhysicalField
   torch.set_num_threads(4);ck=torch.load(F/'physical/modelos/physics_regla_ground_v2.pt',map_location='cpu',weights_only=True);members=[]
   for state in ck['state_dicts']:
    member=PhysicalField().cuda().eval();member.load_state_dict(state);members.append(member)
  if a.infer and name!='physical':
   import torch
   from modelo_factory import Model
   from entrenar import evaluate
   torch.set_num_threads(4);m=Model(*name.rsplit('_',1),cal['voltage'],False);m.restore(torch.load(F/'modelos'/(name+'.pt'),map_location='cpu',weights_only=True));m.cuda().eval()
   metrics,records=evaluate(m,rows,mean,cal['phi_supported'],direction=True);del m;gc.collect();torch.cuda.empty_cache()
  else:
   for r in rows:
    with np.load(F/r['teacher']) as t:
     if name=='physical':
      raw=np.load(F/'datos/geometria'/(r['id']+'.npz'))['raw'];phi=raw[:,:,10]
      if members:
       with torch.inference_mode():
        q=torch.tensor(raw[:,:,:10].reshape(-1,10).copy(),device='cuda');phi=torch.cat([torch.stack([m(z) for m in members]).mean(0).ravel() for z in q.split(65536)]).cpu().numpy().reshape(448,448)
      v=np.interp(phi,cal['phi'],cal['voltage'])
     else:
      z=np.load(F/'resultados'/('predicciones_'+name)/(r['id']+'.npz'));phi=z['phi'];v=z['voltage']
     truth=np.interp(t['phi'],cal['phi'],cal['voltage']);before=np.interp(t['phi0'],cal['phi'],cal['voltage']);supported=(t['phi']>=cal['phi_supported'][0])&(t['phi']<=cal['phi_supported'][1])
     records.append(per_image(v,truth,before,phi,t['valid'],t['air'],t['rule'],mean,supported,r,True))
  real=summarize([r for r in records if r['camera']!='synthetic']);item=dict(model=name,**{k:real[k] for k in ['air_mae_V','air_rmse_V','effect_mae_V','grounded_rule_mae_V']})
  for camera in ['legacy_led','iphone','webcam','synthetic']:item[camera+'_air_mae_V']=summarize([r for r in records if r['camera']==camera])['air_mae_V']
  delta={k:abs(v-float(expected[name][k])) for k,v in item.items() if k!='model'};comparisons[name]=delta
  # Cached predictions and original CUDA LUT evaluation differ only by FP32 interpolation rounding.
  assert max(delta.values())<5e-5,(name,delta)
  (out/(name+'.json')).write_text(json.dumps(dict(metrics=item,records=records,max_difference=max(delta.values())),indent=2));summary.append(item);print(name,'PASS',max(delta.values()),flush=True)
 report=dict(status='passed',mode='fresh checkpoint inference' if a.infer else 'stored-prediction metric replay',rows=summary,differences=comparisons,unit='LED-equivalent V',independent_ground_truth=False)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
