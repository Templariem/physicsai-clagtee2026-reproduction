"""Regenerate frozen features from official checkpoints, without pretrained weights in Git."""
import argparse
from pathlib import Path
import gc
import numpy as np
import torch
from .core import load_samples, make_split, jittered_image
from .runtime import ROOT, choose_device, environment, dump, load_backbone, model_spec, pooled

FILENAMES={'dinov3':'features_dinov3_s16_aug14.npz','ijepa':'features_ijepa_h14_aug14.npz'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backbone',choices=['both','dinov3','ijepa'],default='both')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--clean-only',action='store_true',help='Evaluate saved models without regenerating training augmentations.')
    a=p.parse_args();d=choose_device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    samples=load_samples(ROOT);dev,_=make_split(samples,2)
    dump(a.output/'environment.json',environment(d))
    for kind in (['dinov3','ijepa'] if a.backbone=='both' else [a.backbone]):
        path=a.output/FILENAMES[kind]
        if path.exists(): raise FileExistsError(f'Retain completed features: {path}')
        spec=model_spec(kind);processor,model=load_backbone(kind,d,a.offline)
        clean=np.zeros((len(samples),spec['dimension']),np.float32)
        aug=np.zeros((len(samples),14,spec['dimension']),np.float32)
        devset=set(dev.tolist())
        # Invalid-query frames retain centre-pooled display features but are
        # excluded by the fixed regression manifest before fitting or scoring.
        entries=[(i,j) for i in range(len(samples)) for j in range(1 if a.clean_only else 15)
                 if j==0 or i in devset]
        batchsize=spec['batch_size']
        with torch.inference_mode():
            for start in range(0,len(entries),batchsize):
                batch=entries[start:start+batchsize]
                images=[jittered_image(samples[i],j) for i,j in batch]
                inputs=processor(images=images,return_tensors='pt')
                output=model(**{k:v.to(d) for k,v in inputs.items()})
                vectors=pooled(output,kind,samples,[i for i,_ in batch])
                for v,(i,j) in zip(vectors,batch):
                    if j==0: clean[i]=v
                    else: aug[i,j-1]=v
                if start%max(batchsize,100//batchsize*batchsize)==0:
                    print(f'{kind}: {start+len(batch)}/{len(entries)}',flush=True)
        np.savez_compressed(path,clean=clean,augmented=aug,augmentations=0 if a.clean_only else 14,
                            development_indices=dev,frame=np.array([s.frame_file for s in samples]))
        dump(a.output/f'{kind}_model.json',{'checkpoint':spec,'processor':processor.to_dict(),
             'attention_implementation':getattr(model.config,'_attn_implementation',None),
             'clean_only':a.clean_only,'batch_size':batchsize})
        del model,output;gc.collect()
        if d.type=='cuda':torch.cuda.empty_cache()
        elif d.type=='mps':torch.mps.empty_cache()

if __name__=='__main__':main()
