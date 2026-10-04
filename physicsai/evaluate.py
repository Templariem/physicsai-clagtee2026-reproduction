"""Evaluate the saved paper heads on supplied features and quantify platform differences."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from . import core
from .residual import VisualResidualMLP
from .runtime import ROOT,choose_device,environment,dump
from .features import FILENAMES

def scores(y,p):
    return {'N':len(y),'MSE_V2':float(mean_squared_error(y,p)),
            'MAE_V':float(mean_absolute_error(y,p)),
            'RMSE_V':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}

def predict_checkpoint(ck,physical,features,indices,device):
    x=ck['physics']['scaler'].transform(physical[indices]);members=[]
    for state in ck['physics']['state_dicts']:
        head=core.PhysicsAdapter(5).to(device).eval();head.load_state_dict(state)
        members.append(core.predict(head,x,device))
    result={'Physical':np.mean(members,axis=0)}
    for kind,name in [('dinov3','DINOv3'),('ijepa','I-JEPA')]:
        c=ck[name];visual=c['visual_scaler'].transform(c['pca'].transform(features[kind][indices]))
        head=VisualResidualMLP(128).to(device).eval();head.load_state_dict(c['residual_state_dict'])
        result[name]=result['Physical']+c['gate']*core.predict(head,visual,device)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--features',type=Path,default=ROOT/'reference/features')
    p.add_argument('--checkpoints',type=Path,default=ROOT/'reference/checkpoints')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();device=choose_device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'completed.json').exists():raise FileExistsError(a.output)
    samples=core.load_samples(ROOT);names=np.array([s.frame_file for s in samples]);groups=np.array([s.video_id for s in samples])
    y=np.array([s.target for s in samples]);phys=np.stack([s.phys for s in samples])
    valid=np.isfinite(y);dev=np.flatnonzero(valid&(groups<8));test=np.flatnonzero(valid&(groups>=8))
    assert (len(samples),len(dev),len(test))==(271,128,36)
    original=np.load(ROOT/'reference/results/sample_features.npz')
    np.testing.assert_array_equal(names,original['frame'])
    np.testing.assert_allclose(y,original['target'],rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(phys,original['physical'],rtol=1e-6,atol=1e-7,equal_nan=True)
    features={kind:np.load(a.features/file)['clean'] for kind,file in FILENAMES.items()}
    ck=torch.load(a.checkpoints/'corrected_final.pt',map_location='cpu',weights_only=False)
    pred=predict_checkpoint(ck,phys,features,test,device)
    distance=original['distance_cm'][test];within=original['within_interval'][test]
    table=pd.DataFrame({'frame':names[test],'target_V':y[test],'distance_cm':distance,'within_interval':within,**pred})
    table.to_csv(a.output/'holdout_predictions.csv',index=False)
    summary=[]
    for domain,mask in [('all',np.ones(len(test),bool)),('within_5_17_cm',within),('extrapolated',~within)]:
        for model,v in pred.items():summary.append({'Domain':domain,'Model':model,**scores(y[test][mask],v[mask])})
    pd.DataFrame(summary).to_csv(a.output/'holdout_metrics.csv',index=False)
    cv=[];oof=[]
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for fold,videos in enumerate(protocol['validation_video_groups'],1):
        idx=dev[np.isin(groups[dev],videos)]
        c=torch.load(a.checkpoints/f'cv_fold_{fold}.pt',map_location='cpu',weights_only=False)
        foldpred=predict_checkpoint(c,phys,features,idx,device)
        for name,v in foldpred.items():cv.append({'fold':fold,'Model':name,**scores(y[idx],v)})
        for j,i in enumerate(idx):oof.append({'frame':names[i],'fold':fold,'target_V':y[i],**{n:float(v[j]) for n,v in foldpred.items()}})
    pd.DataFrame(cv).to_csv(a.output/'cv_fold_metrics.csv',index=False)
    pd.DataFrame(oof).to_csv(a.output/'cv_predictions.csv',index=False)
    vision=[]
    for kind,name in [('dinov3','DINOv3'),('ijepa','I-JEPA')]:
        members=[]
        for fold in range(1,6):
            c=torch.load(a.checkpoints/f'{kind}_vision_fold_{fold}.pt',map_location='cpu',weights_only=False)
            head=core.PhysicsAdapter(128).to(device).eval();head.load_state_dict(c['model_state_dict'])
            x=c['scaler'].transform(c['pca'].transform(features[kind][test]))
            members.append(core.predict(head,x,device))
        v=np.mean(members,axis=0);table[name+'_vision_only']=v
        vision.append({'Model':name,**scores(y[test],v)})
    table.to_csv(a.output/'holdout_predictions.csv',index=False)
    pd.DataFrame(vision).to_csv(a.output/'vision_only_metrics.csv',index=False)
    reference=pd.read_csv(ROOT/'reference/results/holdout_predictions.csv').set_index('frame').loc[names[test]]
    differences=[]
    for name,v in pred.items():
        ref=reference[name+'_corrected'].to_numpy();delta=v-ref
        differences.append({'Model':name,'max_abs_prediction_difference_V':float(abs(delta).max()),
                            'mean_abs_prediction_difference_V':float(abs(delta).mean()),
                            'delta_R2':scores(y[test],v)['R2']-scores(y[test],ref)['R2'],
                            'delta_MAE_V':scores(y[test],v)['MAE_V']-scores(y[test],ref)['MAE_V']})
    pd.DataFrame(differences).to_csv(a.output/'comparison_to_paper.csv',index=False)
    rawdiff={}
    for kind,file in FILENAMES.items():
        ref=np.load(ROOT/'reference/features'/file)['clean'];delta=features[kind][valid]-ref[valid]
        rawdiff[kind]={'max_abs':float(abs(delta).max()),'mean_abs':float(abs(delta).mean()),
                       'valid_frames':int(valid.sum())}
    dump(a.output/'feature_comparison.json',rawdiff)
    dump(a.output/'environment.json',environment(device))
    dump(a.output/'completed.json',{'status':'complete','source':'saved heads; no retraining',
         'input_features':str(a.features),'device':str(device),'holdout_frames':len(test),
         'fresh_backbone_features':a.features.resolve()!=(ROOT/'reference/features').resolve()})
    print(pd.DataFrame(differences).to_string(index=False),flush=True)

if __name__=='__main__':main()
