from pathlib import Path
import sys,json,gc,hashlib,importlib.util
import numpy as np,torch
import matplotlib.tri as mtri
from sklearn.decomposition import PCA
from modelo import Model as DenseModel
from datos import Data,load_rows
from red_fisica import PhysicalField
ROOT=Path(__file__).resolve().parents[1]
def main():
 torch.set_num_threads(4);rows=load_rows();train=[r for r in rows if r['split']=='train'];lut=np.load(ROOT/'config/transformacion_led.npz')['voltage']
 for kind in ['dinov3','ijepa']:
  model=DenseModel(kind,'frozen',lut,False).cuda().eval();vectors=[]
  with torch.inference_mode():
   for rgb,a,ids in torch.utils.data.DataLoader(Data(train),batch_size=4 if kind=='dinov3' else 1,num_workers=2):
    x=rgb.cuda();out=model.encoder(pixel_values=(x-model.mean)/model.std,interpolate_pos_encoding=True).last_hidden_state
    if kind=='dinov3':out=out[:,1+model.encoder.config.num_register_tokens:]
    vectors.append(out.mean(1).cpu().numpy())
  x=np.concatenate(vectors);pca=PCA(128,svd_solver='randomized',random_state=4242).fit(x);scale=pca.transform(x).std(0).clip(.00001)
  np.savez_compressed(ROOT/'modelos'/f'pca_{kind}_train.npz',mean=pca.mean_,components=pca.components_,scale=scale)
  print('PCA',kind,len(x),flush=True);del model,x,vectors;gc.collect();torch.cuda.empty_cache()
 (ROOT/'revision/pca_residuales.json').write_text(json.dumps(dict(train_ids=[r['id'] for r in train],fit_validation_or_test=False,dimension=128,source='official frozen encoder, mean patch tokens, training RGB only'),indent=2))
 P=ROOT/'physical';spec=importlib.util.spec_from_file_location('physical_preparation',P/'scripts/preparar.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 z=np.load(P/'modelos/base_sin_regla_v2.npz');base=type('Base',(),{})();base.interpolator=mtri.LinearTriInterpolator(mtri.Triangulation(z['nodes'][:,0],z['nodes'][:,1],z['triangles']),z['phi'])
 ck=torch.load(P/'modelos/physics_regla_ground_v2.pt',map_location='cpu',weights_only=True);members=[]
 for state in ck['state_dicts']:
  m=PhysicalField().cuda().eval();m.load_state_dict(state);members.append(m)
 with torch.inference_mode():
  for i,r in enumerate(rows):
   raw=mod.make_raw(r,np.arange(448*448),base);parts=[]
   for part in torch.tensor(raw,device='cuda').split(32768):parts.append(torch.stack([m(part) for m in members]).mean(0).cpu().numpy())
   phi=np.concatenate(parts);arr=np.concatenate([raw,phi],1).reshape(448,448,11)
   np.savez_compressed(ROOT/'datos/geometria'/(r['id']+'.npz'),raw=arr.astype('float32'))
   if i%25==0:print('geometry',i,len(rows),flush=True)
 (ROOT/'revision/geometria_residuales.json').write_text(json.dumps(dict(images=len(rows),physical_checkpoint_sha256=hashlib.sha256((P/'modelos/physics_regla_ground_v2.pt').read_bytes()).hexdigest(),targets_used=False,source='reviewed mask-derived transforms; oracle geometry input'),indent=2))
if __name__=='__main__':main()
