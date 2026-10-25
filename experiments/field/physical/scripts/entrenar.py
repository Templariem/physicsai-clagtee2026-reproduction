from pathlib import Path
import os,time,json,hashlib,collections,copy
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,torch
from red_fisica import PhysicalField
from metricas import per_image,summarize
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def calibrate(p,lut):
    p=p.float().clamp(0,1)*(len(lut)-1);lo=p.long().clamp(0,len(lut)-2);a=p-lo
    return lut[lo]*(1-a)+lut[lo+1]*a
@torch.inference_mode()
def predict(models,raw):
    output=[]
    for x in raw.split(65536):output.append(torch.stack([m(x) for m in models]).mean(0).ravel())
    return torch.cat(output)
@torch.inference_mode()
def evaluate(models,cases,rows,lut,interval,meanmap,direction=False,output=None):
    records=[]
    for row in rows:
        raw,ids=cases[row['id']];phi=np.zeros((448,448),np.float32)
        phi.ravel()[ids]=predict(models,raw).cpu().numpy()
        voltage=np.interp(phi,np.linspace(0,1,len(lut)),lut.cpu().numpy()).astype('float32')
        with np.load(SOURCE/row['teacher']) as z:a={k:z[k] for k in ['phi','phi0','valid','air','rule']}
        truth=np.interp(a['phi'],np.linspace(0,1,len(lut)),lut.cpu().numpy());before=np.interp(a['phi0'],np.linspace(0,1,len(lut)),lut.cpu().numpy())
        supported=(a['phi']>=interval[0])&(a['phi']<=interval[1])
        records.append(per_image(voltage,truth,before,phi,a['valid'],a['air'],a['rule'],meanmap,supported,row,direction))
        if output is not None:np.savez_compressed(output/(row['id']+'.npz'),phi=phi,voltage=voltage)
    return dict(real=summarize([r for r in records if r['camera']!='synthetic']),all=summarize(records)),records
def main():
    assert not (ROOT/'revision/congelacion_prueba.json').exists(),'Frozen: never retrain'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    rows=json.loads((ROOT/'config/manifest_docente.json').read_text());cache=json.loads((ROOT/'config/cache.json').read_text());lookup={r['id']:r for r in rows}
    cal=np.load(ROOT/'config/transformacion_led.npz');lut=torch.tensor(cal['voltage'],device='cuda');interval=cal['phi_supported']
    meanmap=np.load(SOURCE/'modelos/mapa_promedio_train.npz')['voltage']
    train=[];arrays={k:[] for k in ['raw','target','weights','edge']};valcases={}
    start=time.monotonic()
    for item in cache:
        assert sha(ROOT/item['path'])==item['sha256']
        row=lookup[item['id']]
        with np.load(ROOT/item['path']) as a:
            if item['split']=='train':
                train.append(row)
                for k in arrays:arrays[k].append(a[k])
            elif row['camera']!='synthetic':valcases[row['id']]=(torch.tensor(a['raw'],device='cuda'),a['indices'])
    data={k:torch.tensor(np.stack(v),device='cuda') for k,v in arrays.items()};del arrays
    val=[r for r in rows if r['id'] in valcases];n=len(train)
    counts=collections.Counter(r['scene']['mode']=='ground' for r in train)
    import sys
    sys.path.insert(0,str(SOURCE/'scripts'))
    from sampling import weights as shared_weights
    weights=shared_weights(train)
    members=[];reports=[]
    for seed in [4242,4243,4244]:
        torch.manual_seed(seed);model=PhysicalField().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001,foreach=False)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,60);best=float('inf');bad=0;best_epoch=0;history=[]
        for epoch in range(1,61):
            begin=time.monotonic();model.train();gen=torch.Generator().manual_seed(4242+epoch)
            scene_ids=torch.multinomial(weights,n,replacement=True,generator=gen).tolist();losses=[]
            for begin_scene in range(0,n,16):
                chosen=scene_ids[begin_scene:begin_scene+16];points=torch.randint(0,4096,(len(chosen),512),generator=gen).cuda()
                ii=torch.tensor(chosen,device='cuda')[:,None];raw=data['raw'][ii,points];target=data['target'][ii,points];w=data['weights'][ii,points];edge=data['edge'][ii,points]
                pred=model(raw.reshape(-1,10)).reshape(-1,3);target=target.reshape(-1,3);w=w.reshape(-1,3);edge=edge.reshape(-1,2)
                voltage=calibrate(pred,lut);truth=calibrate(target,lut)
                fit=(torch.nn.functional.smooth_l1_loss(voltage/3,truth/3,beta=.1,reduction='none')*w).sum()/w.sum()
                normalized=((pred-target).square()*w).sum()/w.sum()
                delta=(pred[:,1:]-pred[:,:1])-(target[:,1:]-target[:,:1])
                gradient=(delta.abs()*edge).sum(0)/edge.sum(0).clamp_min(1)
                loss=fit+.2*normalized+gradient.sum();assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            model.eval();metrics,_=evaluate([model],valcases,val,lut,interval,meanmap)
            rv=metrics['real'];criterion=.5*rv['air_mae_V']+.5*rv['effect_mae_V'];sched.step()
            if criterion<best-.0001:
                best=criterion;best_epoch=epoch;bad=0;best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
                torch.save(dict(state_dict=best_state,seed=seed,epoch=epoch,validation=metrics),ROOT/'modelos'/f'miembro_{seed}.pt')
            else:bad+=1
            item=dict(seed=seed,epoch=epoch,loss=float(np.mean(losses)),validation=rv,criterion_V=criterion,best_epoch=best_epoch,seconds=time.monotonic()-begin)
            history.append(item);print(json.dumps(item),flush=True)
            save(ROOT/'resultados'/f'historia_{seed}.json',history)
            save(ROOT/'revision/entrenamiento.json',dict(status='training',seed=seed,member=len(members)+1,total_members=3,epoch=epoch,max_epochs=60,
                best_epoch=best_epoch,best_validation_V=best,epoch_seconds_recent=float(np.median([i['seconds'] for i in history[-3:]])),elapsed_minutes=(time.monotonic()-start)/60,
                gpu_allocated_gib=torch.cuda.memory_allocated()/1024**3))
            if bad>=10:break
        model.load_state_dict(best_state);model.eval().requires_grad_(False);members.append(model)
        reports.append(dict(seed=seed,best_epoch=best_epoch,epochs=len(history),criterion_V=best))
    checkpoint=ROOT/'modelos/physics_regla_ground_v2.pt'
    torch.save(dict(state_dicts=[{k:v.cpu() for k,v in m.state_dict().items()} for m in members],condition='ground_only',unit='V/V_source',domain_scale=16.,mesh_mm=1.5,
        architecture='PhysicalField_v2_domain16',teacher_manifest_sha256=sha(ROOT/'config/manifest_docente.json'),calibration_sha256=sha(ROOT/'config/transformacion_led.npz')),checkpoint)
    save(ROOT/'revision/congelacion_prueba.json',dict(checkpoint_sha256=sha(checkpoint),teacher_manifest_sha256=sha(ROOT/'config/manifest_docente.json'),
        calibration_sha256=sha(ROOT/'config/transformacion_led.npz'),selected_with_validation_only=True,training_reports=reports,unix_time=time.time()))
    validation,_=evaluate(members,valcases,val,lut,interval,meanmap);del valcases,data;torch.cuda.empty_cache()
    save(ROOT/'revision/entrenamiento.json',dict(status='complete_validation_only',training=reports,validation=validation,seconds=time.monotonic()-start))
if __name__=='__main__':main()
