from pathlib import Path
import hashlib,json
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
def load_rows():return json.loads((ROOT/'config/manifest_docente.json').read_text())
def augment(rgb,a,ident,epoch):
    seed=int(hashlib.sha256(f'4242:{epoch}:{ident}'.encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed)
    p=rng.random();scale=1. if p<.35 else rng.uniform(.5,.85) if p<.75 else rng.uniform(.22,.45)
    mat=cv2.getRotationMatrix2D((223.5,223.5),rng.uniform(-8,8),scale);mat[:,2]+=rng.uniform(-.08,.08,2)*448
    valid=a['valid'];support=cv2.warpAffine(valid,mat,(448,448),flags=cv2.INTER_LINEAR,borderValue=0)
    out={}
    for key in ['phi','phi0']+(['raw'] if 'raw' in a else []):
        den=valid[...,None] if a[key].ndim==3 else valid
        numerator=(np.stack([cv2.warpAffine(a[key][:,:,c]*valid,mat,(448,448),flags=cv2.INTER_LINEAR,borderValue=0) for c in range(a[key].shape[2])],axis=2) if a[key].ndim==3 else cv2.warpAffine(a[key]*den,mat,(448,448),flags=cv2.INTER_LINEAR,borderValue=0))
        out[key]=numerator/np.maximum(support[...,None] if a[key].ndim==3 else support,1e-6)
    out['valid']=(support>.999).astype('float32')
    for key in ['air','rule']:out[key]=cv2.warpAffine(a[key],mat,(448,448),flags=cv2.INTER_NEAREST,borderValue=0)*out['valid']
    rgb=cv2.warpAffine(rgb,mat,(448,448),flags=cv2.INTER_LINEAR,borderValue=(114,114,114))
    if rng.random()<.5:rgb=rgb[:,::-1].copy();out={k:v[:,::-1].copy() for k,v in out.items()}
    colors=rgb.astype('float32');colors=(colors-127.5)*rng.uniform(.85,1.15)+127.5
    colors*=rng.uniform(.85,1.15)*rng.uniform(.9,1.1,size=(1,1,3));colors=np.clip(colors,0,255)
    if rng.random()<.2:colors=cv2.GaussianBlur(colors,(0,0),rng.uniform(.3,1.2))
    if rng.random()<.25:colors+=rng.normal(0,rng.uniform(0,4),size=colors.shape)
    rgb=np.clip(colors,0,255).astype('uint8')
    if rng.random()<.2:
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(65,96))]);assert ok
        rgb=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    return rgb,out
class Data(torch.utils.data.Dataset):
    def __init__(self,rows,train=False,epoch=0,geometry=False):self.rows,self.training,self.epoch,self.geometry=rows,train,epoch,geometry
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];image=cv2.imread(str(ROOT/row['image']));assert image is not None
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with np.load(ROOT/row['teacher']) as z:a={k:z[k].astype('float32') for k in ['phi','phi0','valid','air','rule']}
        if self.geometry:a['raw']=np.load(ROOT/'datos/geometria'/(row['id']+'.npz'))['raw'].astype('float32')
        if self.training:rgb,a=augment(rgb,a,row['id'],self.epoch)
        return torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255,{k:(torch.from_numpy(v.copy()).permute(2,0,1) if k=='raw' else torch.from_numpy(v.copy())[None]) for k,v in a.items()},row['id']
