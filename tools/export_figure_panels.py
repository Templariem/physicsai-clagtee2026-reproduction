"""Post-training draft figures: identical cases, crops, voltage scale and train-fit PCA.
Never modifies the manuscript. All case selection is frozen before training.
"""
from pathlib import Path
import gc,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments/field/scripts"))
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from modelo_factory import Model
from datos import ROOT,Data

def main():
 torch.set_num_threads(4);outdir=Path(sys.argv[1]);outdir.mkdir(parents=True,exist_ok=True)
 selected=json.loads((ROOT/'config/figura_seleccion.json').read_text());fit=json.loads((ROOT/'config/figura_pca_train.json').read_text());assert all(r['split']=='train' for r in fit)
 cal=np.load(ROOT/'config/transformacion_led.npz');lut=cal['voltage'];latent={};pred={};meta={};palette=json.loads((ROOT/'config/dino_display_palette.json').read_text())
 for kind in ['dinov3','ijepa']:
  patches={};train=[];g=None
  for arm in ['frozen','partial','full','residual']:
   ck=torch.load(ROOT/'modelos'/f'{kind}_{arm}.pt',map_location='cpu',weights_only=True);m=Model(kind,arm,lut,False);m.restore(ck);m.cuda().eval();g=m.grid;current=[];fields=[]
   rows=selected if arm=='residual' else fit+selected
   with torch.inference_mode():
    for i,r in enumerate(rows):
     rgb,a,_=Data([r],geometry=arm=='residual')[0];rgb=rgb[None].cuda()
     with torch.autocast('cuda',dtype=torch.bfloat16):
      tok=m.encoder(pixel_values=(rgb-m.mean)/m.std,interpolate_pos_encoding=True).last_hidden_state
      if kind=='dinov3':tok=tok[:,1+m.encoder.config.num_register_tokens:]
      if arm=='residual':m.head.context=a['raw'][None].cuda()
      v=m.head(tok.reshape(1,g,g,m.dimension).permute(0,3,1,2),rgb)
     aa=tok[0].float().cpu().numpy()
     if arm!='residual' and i<len(fit):
      lb=r['letterbox'];yy,xx=np.mgrid[:g,:g];good=((xx+.5)*448/g>=lb['left'])&((xx+.5)*448/g<lb['left']+lb['width'])&((yy+.5)*448/g>=lb['top'])&((yy+.5)*448/g<lb['top']+lb['height']);train.append(aa[good.ravel()])
     else:
      current.append(aa);cached=ROOT/'resultados'/f'predicciones_{kind}_{arm}'/(r['id']+'.npz');fields.append(np.load(cached)['voltage'] if cached.exists() else v[0,0].float().cpu().numpy())
   patches[arm]=current;pred[kind+'_'+arm]=fields;del m,ck,tok,v,rgb;gc.collect();torch.cuda.empty_cache();print('figure inference',kind,arm,flush=True)
  # Common projection per backbone across frozen/partial/full TRAIN representations.
  train=np.concatenate(train);direct=PCA(3,svd_solver='randomized',random_state=4242).fit(train);pca=PCA(128,svd_solver='randomized',random_state=4242).fit(train);compressed=pca.transform(train);rgbpca=PCA(3,svd_solver='randomized',random_state=4242).fit(compressed)
  for key in ['raw','pca128']:
   vals=direct.transform(train) if key=='raw' else rgbpca.transform(compressed);lo,hi=np.percentile(vals,[1,99],axis=0)
   for arm in patches:
    tiles=[]
    for aa in patches[arm]:
     z=direct.transform(aa) if key=='raw' else rgbpca.transform(pca.transform(aa));z=np.clip((z-lo)/(hi-lo+1e-8),0,1);z=cv2.resize(z.reshape(g,g,3),(448,448),interpolation=cv2.INTER_CUBIC);z=np.clip(z,0,1)
     if kind=='dinov3':z=np.clip(z@np.asarray(palette['matrix'])+np.asarray(palette['offset']),0,1)
     tiles.append(np.rint(z*255).astype('uint8'))
    latent[kind+'_'+arm+'_'+key]=tiles
  meta[kind]=dict(training_images=[r['id'] for r in fit],basis='joint training representations of frozen, partial and full, reused unchanged for residual',colors='training 1st/99th percentiles fixed across images and arms; not cross-backbone physical quantities')
  del train,compressed,patches;gc.collect()
 # True square display crop on original 448 frame, not squashing wide camera frames.
 photos=[];refs=[];phys=[];crops=[];cm=plt.get_cmap('turbo');norm=plt.Normalize(.1,8)
 def blend(im,v,mask):
  out=im.copy();color=np.uint8(cm(norm(v))[...,:3]*255);out[mask]=cv2.addWeighted(color,.6,im,.4,0)[mask];return out
 for r in selected:
  im=cv2.cvtColor(cv2.imread(str(ROOT/r['image'])),cv2.COLOR_BGR2RGB);mask=cv2.imread(str(ROOT/r['mask']),0);yy,xx=np.where((mask==1)|(mask==2));uy,ux=np.where(mask>0);lb=r['letterbox'];cx,cy=(xx.min()+xx.max()+1)/2,(yy.min()+yy.max()+1)/2
  side=int(np.ceil(max((yy.max()-yy.min()+1)*1.8,(ux.max()-ux.min()+1)*1.12,(uy.max()-uy.min()+1)*1.12)));side=min(side,lb['width'],lb['height'])
  assert side>=max(ux.max()-ux.min()+1,uy.max()-uy.min()+1)
  x0=int(np.clip(round(cx-side/2),max(lb['left'],ux.max()+1-side),min(ux.min(),lb['left']+lb['width']-side)));y0=int(np.clip(round(cy-side/2),max(lb['top'],uy.max()+1-side),min(uy.min(),lb['top']+lb['height']-side)));crops.append((x0,y0,side))
  a=np.load(ROOT/r['teacher']);air=a['valid'].astype(bool)&a['air'].astype(bool);raw=np.load(ROOT/'datos/geometria'/(r['id']+'.npz'))['raw'];photos.append(im);refs.append(blend(im,np.interp(a['phi'],cal['phi'],lut),air));phys.append(blend(im,np.interp(raw[:,:,10],cal['phi'],lut),air))
  for kind in ['dinov3','ijepa']:
   for arm in ['frozen','partial','full','residual']:pred[kind+'_'+arm][len(photos)-1]=blend(im,pred[kind+'_'+arm][len(photos)-1],air)
 from PIL import Image
 for block,arm in enumerate(['residual','full']):
  for j,r in enumerate(selected):
   col=[photos[j],refs[j],latent['dinov3_'+arm+'_raw'][j],latent['ijepa_'+arm+'_raw'][j],latent['dinov3_'+arm+'_pca128'][j],latent['ijepa_'+arm+'_pca128'][j],phys[j],pred['dinov3_'+arm][j],pred['ijepa_'+arm][j]]
   x0,y0,side=crops[j]
   for i,im in enumerate(col):
    tile=cv2.resize(im[y0:y0+side,x0:x0+side],(224,224),interpolation=cv2.INTER_AREA)
    Image.fromarray(tile).save(outdir/f'{block*108+j*9+i:03d}.png')
 (outdir/'provenance.json').write_text(json.dumps(dict(cases=[dict(id=r['id'],crop=list(c)) for r,c in zip(selected,crops)],PCA=meta),indent=2))
 print('PANELS COMPLETE',flush=True)
if __name__=='__main__':main()
