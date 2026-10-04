"""Regenerate the unitless Figure 5 spatial-PCA displays from the actual paper encoders.

PCA is fitted on patch tokens of Videos 1--7 only. These visualization PCAs
are separate from the predictor PCA fitted to global+local feature vectors.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA
from .core import load_samples
from .runtime import ROOT,choose_device,load_backbone,model_spec,environment,dump

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--output',type=Path,default=ROOT/'runs/figures')
    p.add_argument('--offline',action='store_true')
    a=p.parse_args();device=choose_device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    samples=load_samples(ROOT);videos=np.array([s.video_id for s in samples]);selected=[]
    for video in range(1,10):
        ids=np.flatnonzero(videos==video)
        selected.extend([ids[len(ids)//3],ids[2*len(ids)//3]] if video in [5,8,9] else [ids[len(ids)//2]])
    arrays={'frame':np.array([samples[i].frame_file for i in selected])};meta={}
    for kind in ['dinov3','ijepa']:
        processor,model=load_backbone(kind,device,a.offline);allpatch=[]
        with torch.inference_mode():
            for i,s in enumerate(samples):
                output=model(**{k:v.to(device) for k,v in processor(images=[s.image_rgb],return_tensors='pt').items()})
                hidden=output.last_hidden_state
                if kind=='dinov3':hidden=hidden[:,5:,:]
                allpatch.append(hidden[0].cpu().numpy())
                if i%60==0:print(f'{kind} spatial tokens {i+1}/{len(samples)}',flush=True)
        train=np.concatenate([v for i,v in enumerate(allpatch) if videos[i]<8])
        direct=PCA(n_components=3,svd_solver='randomized',random_state=42).fit(train)
        spatial=PCA(n_components=128,svd_solver='randomized',random_state=42).fit(train)
        compressed=spatial.transform(train)
        rgb=PCA(n_components=3,svd_solver='randomized',random_state=42).fit(compressed)
        grid=14 if kind=='dinov3' else 16
        for label in ['raw','pca128']:
            tiles=[]
            for i in selected:
                z=direct.transform(allpatch[i]) if label=='raw' else rgb.transform(spatial.transform(allpatch[i]))
                z=z.reshape(grid,grid,3);z=(z-z.min())/(z.max()-z.min()+1e-8)
                tiles.append(cv2.resize((z*255).astype(np.uint8),(224,224),interpolation=cv2.INTER_CUBIC))
            arrays[kind+'_'+label]=np.stack(tiles)
        np.savez_compressed(a.output/f'{kind}_spatial_pca.npz',direct_mean=direct.mean_,direct_components=direct.components_,
             spatial_mean=spatial.mean_,spatial_components=spatial.components_,rgb_mean=rgb.mean_,rgb_components=rgb.components_)
        meta[kind]={'checkpoint':model_spec(kind),'fit_videos':list(range(1,8)),
                    'fit_frames':int((videos<8).sum()),'fit_patch_tokens':len(train),
                    'grid':grid,'dimensions':int(train.shape[1]),'spatial_components':128,
                    'display_components':3,'per_panel_minmax_RGB':True,'physical_units':None}
        del model,output,hidden,allpatch,train,compressed;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
        elif device.type=='cuda':torch.cuda.empty_cache()
    np.savez_compressed(a.output/'latent_tiles.npz',**arrays)
    dump(a.output/'latent_provenance.json',{'environment':environment(device),'models':meta,
         'predictor_PCA':False,'frames':arrays['frame'].tolist(),
         'change':'Replaced archived DINOv2 visualization with actual official DINOv3-S/16; regenerated both encoders consistently.'})

if __name__=='__main__':main()
