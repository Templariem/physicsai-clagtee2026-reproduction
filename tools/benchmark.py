"""Read-only inference benchmark of the nine frozen study checkpoints.

Outputs only to the specified audit directory; no training or accuracy rescoring.
Inputs are preloaded validation images/geometry, batch one, full 448-square maps.
Residual timings include fresh physical-ensemble evaluation, never its cached output.
"""
from pathlib import Path
import os, sys, json, time, hashlib, gc, platform
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import torch
import transformers

ROOT=Path(__file__).resolve().parents[1]/'experiments/field'
sys.path.insert(0,str(ROOT/'physical/scripts'))
sys.path.insert(0,str(ROOT/'scripts'))
from datos import Data
from modelo_factory import Model
from red_fisica import PhysicalField

OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parents[1]/'runs/inference'; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''): h.update(part)
    return h.hexdigest()

selected=[r for r in json.loads((ROOT/'config/figura_seleccion.json').read_text()) if r['split']=='val']
assert len(selected)==4
cal=np.load(ROOT/'config/transformacion_led.npz')['voltage']
lut=torch.tensor(cal,device='cuda')
physical_path=ROOT/'physical/modelos/physics_regla_ground_v2.pt'
checkpoint_hashes={str(physical_path):sha(physical_path)}
ck=torch.load(physical_path,map_location='cpu',weights_only=True)
members=[]
for state in ck['state_dicts']:
    m=PhysicalField().cuda().eval();m.load_state_dict(state);members.append(m)
del ck
physical_parameters=sum(p.numel() for m in members for p in m.parameters())
inputs=[]
for row in selected:
    rgb,a,ident=Data([row],geometry=True)[0]
    raw=a['raw'][:10].permute(1,2,0).reshape(-1,10).contiguous().cuda()
    inputs.append((rgb[None].cuda(),raw,ident))

def physical_phi(raw):
    parts=[]
    for part in raw.split(65536):
        parts.append(torch.stack([m(part) for m in members]).mean(0).ravel())
    return torch.cat(parts).reshape(1,1,448,448)

def calibrate(phi):
    p=phi.float().clamp(0,1)*(len(lut)-1)
    lo=p.long().clamp(0,len(lut)-2);a=p-lo
    return lut[lo]*(1-a)+lut[lo+1]*a

@torch.inference_mode()
def measure(key,fn,counts):
    for i in range(12):fn(inputs[i%4])
    torch.cuda.synchronize()
    ms=[]
    for i in range(100):
        torch.cuda.synchronize();start=time.perf_counter()
        result=fn(inputs[i%4])
        torch.cuda.synchronize();ms.append((time.perf_counter()-start)*1000)
    assert result.shape==(1,1,448,448) and torch.isfinite(result).all()
    record=dict(counts,median_ms=float(np.median(ms)),mean_ms=float(np.mean(ms)),
                p10_ms=float(np.percentile(ms,10)),p90_ms=float(np.percentile(ms,90)),samples_ms=ms)
    results[key]=record
    (OUT/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    print(key,record['total_parameters'],round(record['median_ms'],3),'ms',flush=True)

results={}
measure('physical',lambda inp:calibrate(physical_phi(inp[1])),
        dict(total_parameters=physical_parameters,trained_parameters=physical_parameters,
             encoder_parameters=0,physical_parameters=physical_parameters,head_parameters=0))

for kind in ['dinov3','ijepa']:
    for arm in ['residual','frozen','partial','full']:
        key=kind+'_'+arm;path=ROOT/'modelos'/(key+'.pt');checkpoint_hashes[str(path)]=sha(path)
        ck=torch.load(path,map_location='cpu',weights_only=True)
        model=Model(kind,arm,cal,False);model.restore(ck);model.cuda().eval();del ck
        encoder=sum(p.numel() for p in model.encoder.parameters())
        head=sum(p.numel() for p in model.head.parameters())
        extra=physical_parameters if arm=='residual' else 0
        counts=dict(total_parameters=encoder+head+extra,
                    trained_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    encoder_parameters=encoder,head_parameters=head,physical_parameters=extra)
        def forward(inp):
            rgb,raw,_=inp
            if arm=='residual':
                base=physical_phi(raw)
                geometry=raw.reshape(1,448,448,10).permute(0,3,1,2)
                model.head.context=torch.cat([geometry,base],1)
            with torch.autocast('cuda',dtype=torch.bfloat16):return model(rgb)
        measure(key,forward,counts)
        del model;gc.collect();torch.cuda.empty_cache()

assert all(sha(Path(p))==h for p,h in checkpoint_hashes.items())
record=dict(protocol=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
    transformers=transformers.__version__,cuda=torch.version.cuda,python=platform.python_version(),
    resolution=[448,448],batch=1,warmup=12,repeats=100,statistic='median of synchronized wall-clock time',
    validation_cases=[r['id'] for r in selected],precision='FP32 physical ensemble; BF16 autocast RGB encoder/head, matching study evaluation',
    included='GPU-resident model inputs to calibrated dense voltage; residuals include fresh physical ensemble, PCA and residual head',
    excluded='disk I/O, image/geometry preparation, host-device copies, detector, FEM solve, camera and rendering',
    parameters='nn.Parameter totals, including frozen encoder and three physical ensemble members when required; excludes non-parameter PCA/LUT buffers',
    checkpoint_hashes=checkpoint_hashes,checkpoints_unchanged=True),results=results)
(OUT/'benchmark.json').write_text(json.dumps(record,indent=2)+'\n')
print('COMPLETE',flush=True)
