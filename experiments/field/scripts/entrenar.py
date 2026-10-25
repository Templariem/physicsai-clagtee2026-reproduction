from pathlib import Path
import argparse,collections,gc,hashlib,json,random,time
import numpy as np,torch
from modelo_factory import Model
from red_potencial import loss
from datos import Data,load_rows,ROOT
from metricas import per_image,summarize
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def mean_baseline(rows,lut):
    sums=np.zeros((448,448),np.float64);counts=np.zeros_like(sums)
    for row in rows:
        with np.load(ROOT/row['teacher']) as a:
            valid=a['valid'];sums+=np.interp(a['phi'],np.linspace(0,1,len(lut)),lut)*valid;counts+=valid
    mean=(sums/np.maximum(counts,1)).astype('float32')
    np.savez_compressed(ROOT/'modelos/mapa_promedio_train.npz',voltage=mean,count=counts.astype('int32'))
    save(ROOT/'revision/procedencia_promedio.json',dict(train_ids=[r['id'] for r in rows],test_or_validation_used=False))
    return mean
@torch.inference_mode()
def evaluate(model,rows,meanmap,interval,direction=False,output=None):
    model.eval();lookup={r['id']:r for r in rows};records=[];dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    loader=torch.utils.data.DataLoader(Data(rows,geometry=model.arm=='residual'),batch_size=4 if model.kind=='dinov3' else 1,num_workers=2,pin_memory=True)
    for rgb,a,ids in loader:
        if model.arm=='residual':model.head.context=a['raw'].cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=dtype):pred,phi=model(rgb.cuda(non_blocking=True),True)
        pred=pred.float().cpu().numpy()[:,0];phi=phi.float().cpu().numpy()[:,0]
        truth=model.head.calibrate(a['phi'].cuda()).cpu().numpy()[:,0];before=model.head.calibrate(a['phi0'].cuda()).cpu().numpy()[:,0]
        for i,ident in enumerate(ids):
            arrays={k:v[i,0].numpy() for k,v in a.items()};support=(arrays['phi']>=interval[0])&(arrays['phi']<=interval[1])
            record=per_image(pred[i],truth[i],before[i],phi[i],arrays['valid'],arrays['air'],arrays['rule'],meanmap,support,lookup[ident],direction)
            assert record['finite'];records.append(record)
            if output is not None:np.savez_compressed(output/(ident+'.npz'),voltage=pred[i],phi=phi[i])
    real=[r for r in records if r['camera']!='synthetic']
    return dict(real=summarize(real),all=summarize(records)),records
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['dinov3','ijepa'],required=True);p.add_argument('--arm',choices=['frozen','partial','full','residual'],required=True);args=p.parse_args()
    assert not (ROOT/'revision/congelacion_prueba.json').exists(),'Frozen after training; no retraining'
    run=args.model+'_'+args.arm;checkpoint=ROOT/'modelos'/(run+'.pt');assert not checkpoint.exists(),'Existing model: inspect before overwriting'
    torch.set_num_threads(4);torch.manual_seed(4242);np.random.seed(4242);random.seed(4242);torch.backends.cudnn.benchmark=False
    protocol=json.loads((ROOT/'config/protocolo.json').read_text());epochs=protocol['training']['max_epochs'];patience=protocol['training']['patience']
    rows=load_rows();train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val' and r['camera']!='synthetic']
    cal=np.load(ROOT/'config/transformacion_led.npz');lut=cal['voltage'];interval=cal['phi_supported']
    meanpath=ROOT/'modelos/mapa_promedio_train.npz';mean=mean_baseline(train,lut) if not meanpath.exists() else np.load(meanpath)['voltage']
    model=Model(args.model,args.arm,lut).cuda();groups=[dict(params=model.head.parameters(),lr=.001)]
    if args.arm in ['partial','full']:groups.append(dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=.00001))
    opt=torch.optim.AdamW(groups,weight_decay=.001,foreach=False);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=0)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    batch=4 if args.model=='dinov3' else 1;effective=16;start=time.monotonic();best=float('inf');bestepoch=0;bad=0;history=[]
    counts=collections.Counter(r['scene']['mode']=='ground' for r in train)
    from sampling import weights as shared_weights
    weights=shared_weights(train)
    save(ROOT/'revision'/('inicio_'+run+'.json'),dict(kind=args.model,arm=args.arm,micro_batch=batch,effective_batch=effective,
        trainable_encoder_parameters=model.trainable_encoder_params,partial_blocks=model.partial_blocks,
        encoder_block_path=model.block_path,decoder_initialization_seed=4242,
        teacher_manifest_sha256=sha(ROOT/'config/manifest_docente.json'),protocol_sha256=sha(ROOT/'config/protocolo.json')))
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        begin=time.monotonic();model.train();losses=[];seen=0;opt.zero_grad(set_to_none=True)
        generator=torch.Generator().manual_seed(4242+epoch)
        indices=torch.multinomial(weights,len(train),replacement=True,generator=generator).tolist()
        dataset=torch.utils.data.Subset(Data(train,train=True,epoch=epoch,geometry=args.arm=='residual'),indices)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch,num_workers=4,pin_memory=True)
        for rgb,a,ids in loader:
            window=min(effective,len(train)-(seen//effective)*effective);a={k:v.cuda(non_blocking=True) for k,v in a.items()}
            if args.arm=='residual':model.head.context=a['raw']
            with torch.autocast('cuda',dtype=dtype):
                voltage,phi=model(rgb.cuda(non_blocking=True),True)
                value=loss(model.head,voltage,phi,a['phi'],a['phi0'],a['valid'],a['air'],a['rule'])
            assert torch.isfinite(value),(run,epoch,'nonfinite loss');scaler.scale(value*(rgb.shape[0]/window)).backward();seen+=rgb.shape[0];losses.append(float(value.detach()))
            if seen%effective==0 or seen==len(train):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        del loader,dataset;metrics,_=evaluate(model,val,mean,interval);rv=metrics['real'];criterion=.5*rv['air_mae_V']+.5*rv['effect_mae_V'];sched.step()
        item=dict(epoch=epoch,loss=float(np.mean(losses)),validation=rv,criterion=criterion,seconds=round(time.monotonic()-begin,1));history.append(item)
        print(run,json.dumps(item),flush=True)
        if criterion<best-.0001:
            best=criterion;bestepoch=epoch;bad=0;ck=model.checkpoint();ck.update(epoch=epoch,validation=metrics,
                teacher_manifest_sha256=sha(ROOT/'config/manifest_docente.json'),calibration_sha256=sha(ROOT/'config/transformacion_led.npz'),protocol_sha256=sha(ROOT/'config/protocolo.json'))
            tmp=checkpoint.with_suffix('.pt.tmp');torch.save(ck,tmp);tmp.replace(checkpoint);del ck
        else:bad+=1
        save(ROOT/'resultados'/('historia_'+run+'.json'),history)
        save(ROOT/'revision/entrenamiento.json',dict(status='training',run=run,epoch=epoch,max_epochs=epochs,best_epoch=bestepoch,
            best_validation_criterion_V=best,epoch_seconds_recent=float(np.median([x['seconds'] for x in history[-3:]])),
            eta_current_run_max_minutes=float(np.median([x['seconds'] for x in history[-3:]]))*(epochs-epoch)/60,
            elapsed_minutes=(time.monotonic()-start)/60,micro_batch=batch,effective_batch=effective,
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
        if bad>=patience:break
    save(ROOT/'resultados'/('resumen_'+run+'.json'),dict(run=run,best_epoch=bestepoch,epochs=len(history),best_validation_criterion_V=best,
        elapsed_minutes=(time.monotonic()-start)/60,checkpoint_sha256=sha(checkpoint),checkpoint=str(checkpoint),
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,validation_only=True))
    save(ROOT/'revision/entrenamiento.json',dict(status='run_complete_validation_only',run=run,best_epoch=bestepoch,epochs=len(history)))
if __name__=='__main__':main()
