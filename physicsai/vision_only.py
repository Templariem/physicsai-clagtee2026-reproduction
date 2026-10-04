"""Separate vision-only diagnostic from Table V: 14 jitters and five-fold averaging."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from . import core
from .features import FILENAMES
from .runtime import ROOT,choose_device,environment,dump
from .evaluate import scores

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--features',type=Path,default=ROOT/'reference/features')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();device=choose_device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'completed.json').exists():raise FileExistsError(a.output)
    models=a.output/'checkpoints';models.mkdir(exist_ok=True)
    samples=core.load_samples(ROOT);names=np.array([s.frame_file for s in samples]);groups=np.array([s.video_id for s in samples]);y=np.array([s.target for s in samples])
    dev=np.flatnonzero(np.isfinite(y)&(groups<8));test=np.flatnonzero(np.isfinite(y)&(groups>=8))
    assert (len(dev),len(test))==(128,36)
    proto=json.loads((ROOT/'protocol.json').read_text());rows=[];summary=[];oof=[]
    predictions={'frame':names[test],'target_V':y[test]}
    for kind,name in [('dinov3','DINOv3'),('ijepa','I-JEPA')]:
        c=np.load(a.features/FILENAMES[kind]);clean,aug=c['clean'],c['augmented']
        assert int(c['augmentations'])==14
        members=[]
        for fold,videos in enumerate(proto['validation_video_groups'],1):
            tr=dev[~np.isin(groups[dev],videos)];va=dev[np.isin(groups[dev],videos)]
            expanded=np.concatenate([clean[tr,None,:],aug[tr,:14]],axis=1).reshape(-1,clean.shape[1])
            pca=PCA(n_components=128,svd_solver='randomized',random_state=42).fit(expanded)
            scaler=StandardScaler().fit(pca.transform(expanded))
            model=core.train_adapter(scaler.transform(pca.transform(expanded)),np.repeat(y[tr],15),128,180,device,42+fold-1)
            vp=core.predict(model,scaler.transform(pca.transform(clean[va])),device)
            members.append(core.predict(model,scaler.transform(pca.transform(clean[test])),device))
            rows.append({'Model':name,'fold':fold,**scores(y[va],vp)})
            oof.extend({'Model':name,'fold':fold,'frame':names[i],'target_V':y[i],'prediction_V':vp[k]} for k,i in enumerate(va))
            torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'pca':pca,'scaler':scaler,'seed':42+fold-1,'train_indices':tr.tolist(),'val_indices':va.tolist()},models/f'{kind}_vision_fold_{fold}.pt')
            print(f'{kind}: vision-only fold {fold}/5 complete',flush=True)
            pd.DataFrame(rows).to_csv(a.output/'vision_only_cv_folds.csv',index=False)
        mean=np.mean(members,axis=0);predictions[name+'_vision_only_V']=mean
        cv=pd.DataFrame([r for r in rows if r['Model']==name])
        row={'Model':name,**{'Test_'+k:v for k,v in scores(y[test],mean).items()}}
        for k in ['MSE_V2','MAE_V','R2']:
            row['CV_'+k+'_mean']=float(cv[k].mean());row['CV_'+k+'_sd']=float(cv[k].std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(a.output/'vision_only_summary.csv',index=False)
    pd.DataFrame(predictions).to_csv(a.output/'vision_only_holdout_predictions.csv',index=False)
    pd.DataFrame(oof).to_csv(a.output/'vision_only_oof_predictions.csv',index=False)
    dump(a.output/'environment.json',environment(device));dump(a.output/'completed.json',{'status':'complete','models_trained':10,'device':str(device)})

if __name__=='__main__':main()
