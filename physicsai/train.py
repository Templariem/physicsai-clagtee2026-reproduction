"""Train the final paper's fixed physical ensemble and sequential visual residuals.

Uses archived or freshly generated frozen-encoder features. All learned
preprocessing and heads are fitted on the fixed valid-query development set.
The archived holdout has already been examined: this is a corrected reanalysis.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

from .runtime import ROOT, choose_device, environment
RUN=ROOT
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from . import core as original
from .residual import train_residual, residual_predict

OUT=None
MODELS=None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def log(message):
    line=f'{datetime.datetime.now().isoformat(timespec="seconds")} {message}'
    print(line,flush=True)
    with (OUT/'training.log').open('a') as f:
        f.write(line+'\n')

def metrics(y,p):
    assert np.isfinite(p).all()
    return {'N':len(y),'MSE_V2':float(mean_squared_error(y,p)),
            'MAE_V':float(mean_absolute_error(y,p)),
            'RMSE_V':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p))}

def cpu_state(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}

def fit_physics(train_idx,eval_idx,phys,y,device,first_seed,label):
    assert not set(train_idx)&set(eval_idx)
    assert len(train_idx)%32!=1, 'Avoid an unplanned singleton BatchNorm batch.'
    scaler=StandardScaler().fit(phys[train_idx])
    xtrain=scaler.transform(phys[train_idx]);xeval=scaler.transform(phys[eval_idx])
    train_members=[];eval_members=[];states=[]
    for seed in range(first_seed,first_seed+3):
        start=time.perf_counter()
        model=original.train_adapter(xtrain,y[train_idx],5,180,device,seed)
        train_members.append(original.predict(model,xtrain,device))
        eval_members.append(original.predict(model,xeval,device))
        states.append(cpu_state(model))
        log(f'{label}: physical seed {seed} completed in {time.perf_counter()-start:.1f}s')
        del model
    artifact={'state_dicts':states,'scaler':scaler,'seeds':list(range(first_seed,first_seed+3)),
              'train_indices':train_idx.tolist(),'eval_indices':eval_idx.tolist()}
    return np.mean(train_members,axis=0),np.mean(eval_members,axis=0),artifact

def fit_visual(kind,train_idx,eval_idx,y,physics_train,physics_eval,cache,device,seed,label):
    start=time.perf_counter()
    clean,augmented=cache
    variants=np.concatenate([clean[train_idx,None,:],augmented[train_idx,:4]],axis=1)
    train_visual=variants.reshape(-1,variants.shape[-1])
    residual_y=np.repeat(y[train_idx]-physics_train,5)
    assert train_visual.shape[0]>=128
    pca=PCA(n_components=128,svd_solver='randomized',random_state=42).fit(train_visual)
    scaler=StandardScaler().fit(pca.transform(train_visual))
    xtrain=scaler.transform(pca.transform(train_visual))
    xeval=scaler.transform(pca.transform(clean[eval_idx]))
    model=train_residual(xtrain,residual_y,0.01,device,seed)
    delta=residual_predict(model,xeval,device)
    prediction=physics_eval+0.25*delta
    artifact={'pca':pca,'visual_scaler':scaler,'residual_state_dict':cpu_state(model),
              'seed':seed,'gate':0.25,'train_indices':train_idx.tolist(),'eval_indices':eval_idx.tolist()}
    log(f'{label}: {kind} residual completed in {time.perf_counter()-start:.1f}s')
    return prediction,delta,artifact

def fit_set(train_idx,eval_idx,phys,y,caches,device,physics_seed,residual_seed,label):
    ptr,pev,physical_artifact=fit_physics(train_idx,eval_idx,phys,y,device,physics_seed,label)
    predictions={'Physical':pev}
    artifact={'physics':physical_artifact,'protocol_sha256':sha(RUN/'protocol.json'),
              'curation_sha256':sha(ROOT/'data/regression_annotations.json')}
    for kind in ['DINOv3','I-JEPA']:
        pred,delta,a=fit_visual(kind,train_idx,eval_idx,y,ptr,pev,caches[kind],device,residual_seed,label)
        predictions[kind]=pred;artifact[kind]=a
    torch.save(artifact,MODELS/f'{label}.pt')
    return predictions


def main():
    from .calibration import A_V, GAMMA, voltage, decay_derivative, within_calibration_interval
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--features',type=Path,default=ROOT/'reference/features')
    args=parser.parse_args()
    global OUT, MODELS
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    MODELS=OUT/'checkpoints';MODELS.mkdir(exist_ok=True)
    assert not (OUT/'completed.json').exists(), 'Retain the completed experiment.'
    device=choose_device(args.device)
    started=time.perf_counter()
    protocol=json.loads((RUN/'protocol.json').read_text())
    curated=json.loads((ROOT/'data/regression_annotations.json').read_text())
    assert sha(ROOT/'data/annotations.json')==protocol['original_annotation_sha256']
    samples=original.load_samples(ROOT)
    assert len(samples)==271
    frames=np.asarray([s.frame_file for s in samples]);groups=np.asarray([s.video_id for s in samples])
    phys=np.stack([s.phys for s in samples]);y=np.asarray([s.target for s in samples])
    old_dev,old_test=original.make_split(samples,2)
    keep=np.asarray([name in curated for name in frames])
    dev=old_dev[keep[old_dev]];test=old_test[keep[old_test]]
    assert (len(dev),len(test))==(128,36)
    assert np.isnan(y[~keep]).all() and np.isfinite(y[keep]).all()
    assert np.isfinite(phys[keep]).all()
    distance=np.full(len(samples),np.nan)
    for i in np.flatnonzero(keep):
        conductors=[]
        for p in curated[frames[i]]:
            points=np.asarray(p['points'],dtype=np.int32);mom=cv2.moments(points)
            if mom['m00']<=0:continue
            center=[int(mom['m10']/mom['m00']),int(mom['m01']/mom['m00'])]
            if p['class'] in ('esfera','bobina_cobre'):conductors.append(points)
            if p['class']=='esfera':scale=4.5/(2*int(np.mean(np.linalg.norm(points-center,axis=1))))
            if p['class']=='diodo_mano':query=center
        distance[i]=max(0,min(-cv2.pointPolygonTest(p,tuple(map(float,query)),True) for p in conductors))*scale
    within=within_calibration_interval(distance)
    assert (within[dev].sum(),within[test].sum())==(71,15)
    assert distance[keep].min()>0.05
    assert np.allclose(y[keep],voltage(distance[keep]),atol=1e-12,rtol=0)
    assert np.allclose(phys[keep,1],decay_derivative(distance[keep]),atol=3e-7,rtol=0)
    caches={};inputs={}
    for kind,filename in [('DINOv3','features_dinov3_s16_aug14.npz'),('I-JEPA','features_ijepa_h14_aug14.npz')]:
        path=args.features/filename
        with np.load(path) as c:
            assert int(c['augmentations'])==14, 'Training requires full augmented features, not --clean-only.'
            assert np.array_equal(c['development_indices'],old_dev)
            clean,aug=c['clean'].copy(),c['augmented'].copy()
            assert clean.shape[0]==271 and aug.shape[:2]==(271,14)
            caches[kind]=(clean,aug)
        inputs[filename]=sha(path)
    check={'original_frames':271,'retained':164,'excluded_missing_query':107,'development':128,'holdout':36,
           'within_interval_development':71,'extrapolated_development':57,'within_interval_holdout':15,'extrapolated_holdout':21,
           'distance_range_cm':[float(np.nanmin(distance)),float(np.nanmax(distance))],
           'holdout_target_range_V':[float(y[test].min()),float(y[test].max())],
           'calibration_A_V':A_V,'calibration_gamma':GAMMA,'label_clipping':False,
           'centre_distance_offset_removed_cm':2.25,'labels_empirical_slopes_and_image_coordinates_unchanged':True,'frozen_visual_features_unchanged':True,
           'labels_and_derivatives_independently_checked':True,'missing_query_targets_are_NaN':True}
    dump(OUT/'dataset_checks.json',check)
    pd.DataFrame({'original_index':np.arange(len(samples)),'frame':frames,'video':groups,'retained':keep,
                  'partition':np.where(groups>=8,'holdout','development'),'distance_cm':distance,
                  'calibration_domain':np.where(~keep,'missing_query',np.where(within,'within_5_17_cm','extrapolated')),
                  'target_V':y}).to_csv(OUT/'dataset_manifest.csv',index=False)
    np.savez_compressed(OUT/'sample_features.npz',physical=phys,target=y,frame=frames,video=groups,distance_cm=distance,within_interval=within)
    if args.preflight:
        print(json.dumps(check,indent=2));return
    dump(OUT/'environment.json',environment(device))
    cvrows=[];oof=[];folds=[]
    for fold,videos in enumerate(protocol['validation_video_groups'],1):
        va=dev[np.isin(groups[dev],videos)];tr=dev[~np.isin(groups[dev],videos)]
        assert not set(groups[tr])&set(groups[va])
        folds.append({'fold':fold,'validation_videos':videos,'train_indices':tr.tolist(),'validation_indices':va.tolist(),
                      'train_count':len(tr),'validation_count':len(va)})
        log(f'Fixed-protocol CV fold {fold}: train={len(tr)}, validation={len(va)}')
        predictions=fit_set(tr,va,phys,y,caches,device,42+10*fold,42+100*fold+128,f'cv_fold_{fold}')
        for name,pred in predictions.items():cvrows.append({'fold':fold,'Model':name,**metrics(y[va],pred)})
        for offset,idx in enumerate(va):
            oof.append({'frame':frames[idx],'video':int(groups[idx]),'fold':fold,'distance_cm':distance[idx],
                        'within_interval':bool(within[idx]),'target_V':y[idx],**{n:float(p[offset]) for n,p in predictions.items()}})
        pd.DataFrame(cvrows).to_csv(OUT/'cv_fold_metrics.csv',index=False)
        pd.DataFrame(oof).to_csv(OUT/'cv_predictions.csv',index=False)
        dump(OUT/'fold_partitions.json',folds)
    assert len(oof)==128 and len({r['frame'] for r in oof})==128
    cv=pd.DataFrame(cvrows);summary=[]
    for name in ['Physical','DINOv3','I-JEPA']:
        part=cv[cv.Model==name];row={'Model':name,'folds':5}
        for metric in ['MSE_V2','MAE_V','RMSE_V','R2']:
            row[metric+'_mean']=float(part[metric].mean());row[metric+'_sd']=float(part[metric].std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'cv_summary.csv',index=False)
    log('Final five-LED refit on 128 development frames; same 36-frame holdout.')
    predictions=fit_set(dev,test,phys,y,caches,device,42,170,'corrected_final')
    rows=[{'Stage':'Surface_distance_retrained_36','Model':name,**metrics(y[test],pred)} for name,pred in predictions.items()]
    pd.DataFrame(rows).to_csv(OUT/'holdout_comparison.csv',index=False)
    pd.DataFrame({'frame':frames[test],'video':groups[test],'distance_cm':distance[test],'within_interval':within[test],
                  'target_V':y[test],**{name+'_corrected':pred for name,pred in predictions.items()}}).to_csv(OUT/'holdout_predictions.csv',index=False)
    domains=[]
    for label,mask in [('within_5_17_cm',within[test]),('extrapolated',~within[test])]:
        for name,pred in predictions.items():domains.append({'Domain':label,'Model':name,**metrics(y[test][mask],pred[mask])})
    pd.DataFrame(domains).to_csv(OUT/'holdout_by_calibration_domain.csv',index=False)
    by_video=[]
    for video in [8,9]:
        mask=groups[test]==video
        for name,pred in predictions.items():by_video.append({'video':video,'Model':name,**metrics(y[test][mask],pred[mask])})
    pd.DataFrame(by_video).to_csv(OUT/'holdout_by_video.csv',index=False)
    for filename,checksum in inputs.items():assert sha(args.features/filename)==checksum
    dump(OUT/'completed.json',{'status':'complete','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-started,'device':str(device),'retrained_physics_ensembles':6,
        'retrained_visual_heads':12,'hyperparameters_unchanged':True,'new_independent_holdout':False,
        'original_annotations_and_caches_unchanged':True,'calibration':'five_LED_power_law_only'})
    log('Completed fixed five-LED protocol without hyperparameter selection.')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(domains).to_string(index=False),flush=True)


if __name__=='__main__':main()
