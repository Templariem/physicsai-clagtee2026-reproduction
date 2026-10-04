"""Five alternating timing passes; batch one; five warm-ups then 36 holdout frames."""
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
import torch
from . import core as module
from .residual import VisualResidualMLP
from .runtime import ROOT,choose_device,environment,dump,load_backbone
RESULTS_DIR=ROOT/'reference/checkpoints'

def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def benchmark_model(kind: str, module, samples, test_indices, device: torch.device) -> dict:
    if kind == "dinov3":
        display_name = "DINOv3-S/16"
        model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        checkpoint_path = RESULTS_DIR / "checkpoint_dinov3_residual_final.pt"
        processor, backbone = load_backbone("dinov3", device, offline=True)
    else:
        display_name = "I-JEPA-H/14"
        model_name = "facebook/ijepa_vith14_1k"
        checkpoint_path = RESULTS_DIR / "checkpoint_ijepa_residual_fixed.pt"
        processor, backbone = load_backbone("ijepa", device, offline=True)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        combined = torch.load(RESULTS_DIR/'corrected_final.pt', map_location='cpu', weights_only=False)
        visual = combined['DINOv3' if kind=='dinov3' else 'I-JEPA']
        checkpoint = {'physics_state_dicts':combined['physics']['state_dicts'],
                      'physics_scaler':combined['physics']['scaler'],
                      'residual_input_dim':128, **visual}
    physics_models = []
    for state_dict in checkpoint["physics_state_dicts"]:
        physics_model = module.PhysicsAdapter(5).to(device).eval()
        physics_model.load_state_dict(state_dict)
        physics_models.append(physics_model)
    residual_model = VisualResidualMLP(checkpoint["residual_input_dim"]).to(device).eval()
    residual_model.load_state_dict(checkpoint["residual_state_dict"])
    pca = checkpoint["pca"]
    visual_scaler = checkpoint["visual_scaler"]
    physical_scaler = checkpoint["physics_scaler"]
    gate = checkpoint["gate"]

    def one_sample(sample_idx: int):
        sample = samples[sample_idx]
        t0 = time.perf_counter()
        inputs = processor(images=[sample.image_rgb.copy()], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        synchronize(device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            output = backbone(pixel_values=pixel_values)
            if kind == "dinov3":
                global_feature = output.pooler_output.detach().cpu().numpy()
                patch_tokens = output.last_hidden_state[:, 5:, :]
                local_feature = module.weighted_local(
                    patch_tokens, samples, [sample_idx], grid=14
                )
            else:
                hidden = output.last_hidden_state
                global_feature = hidden.mean(dim=1).detach().cpu().numpy()
                local_feature = module.weighted_local(
                    hidden, samples, [sample_idx], grid=16
                )
        synchronize(device)
        t2 = time.perf_counter()
        raw = np.concatenate([global_feature, local_feature], axis=1)
        visual = visual_scaler.transform(pca.transform(raw))
        physical = physical_scaler.transform(sample.phys.reshape(1, -1))
        with torch.inference_mode():
            physical_tensor = torch.tensor(physical, dtype=torch.float32, device=device)
            physics_prediction = torch.stack([
                physics_model(physical_tensor) for physics_model in physics_models
            ]).mean(dim=0)
            residual_prediction = residual_model(
                torch.tensor(visual, dtype=torch.float32, device=device)
            )
            prediction = physics_prediction + gate * residual_prediction
        synchronize(device)
        _ = float(prediction.cpu().item())
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    for sample_idx in test_indices[:5]:
        one_sample(int(sample_idx))
    timings = np.asarray([one_sample(int(idx)) for idx in test_indices])
    total = timings.sum(axis=1)
    backbone_params = sum(parameter.numel() for parameter in backbone.parameters())
    adapter_params = (
        sum(sum(parameter.numel() for parameter in model.parameters()) for model in physics_models)
        + sum(parameter.numel() for parameter in residual_model.parameters())
    )
    row = {
        "Model": display_name,
        "Official_Model_ID": model_name,
        "Backbone_Params": backbone_params,
        "Adapter_Params": adapter_params,
        "Total_Params": backbone_params + adapter_params,
        "Parameter_Memory_FP32_MB": (backbone_params + adapter_params) * 4 / 1024**2,
        "Preprocess_Mean_ms": float(timings[:, 0].mean()),
        "Backbone_Mean_ms": float(timings[:, 1].mean()),
        "PCA_Adapter_Mean_ms": float(timings[:, 2].mean()),
        "End_to_End_Mean_ms": float(total.mean()),
        "End_to_End_Std_ms": float(total.std()),
        "FPS": float(1000.0 / total.mean()),
        "N_Test_Frames": len(test_indices),
        "Device": str(device),
    }
    del backbone, physics_models, residual_model
    if device.type == "mps":
        torch.mps.empty_cache()
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--checkpoints',type=Path,default=ROOT/'reference/checkpoints')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();device=choose_device(a.device)
    global RESULTS_DIR
    RESULTS_DIR=a.checkpoints
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'completed.json').exists():raise FileExistsError(a.output)
    samples=module.load_samples(ROOT)
    indices=np.array([i for i,s in enumerate(samples) if s.video_id>=8 and np.isfinite(s.target)])
    assert len(indices)==36
    dump(a.output/'environment.json',environment(device))
    rows=[]
    for trial in range(1,6):
        order=['dinov3','ijepa'] if trial%2 else ['ijepa','dinov3']
        for order_index,kind in enumerate(order,1):
            row=benchmark_model(kind,module,samples,indices,device)
            row.update(trial=trial,order_in_trial=order_index);rows.append(row)
            pd.DataFrame(rows).to_csv(a.output/'benchmark_trials.csv',index=False)
            print(f"{kind} pass {trial}: {row['End_to_End_Mean_ms']:.3f} ms; {row['FPS']:.2f} FPS",flush=True)
            if device.type=='cuda':torch.cuda.empty_cache()
    frame=pd.DataFrame(rows);summary=[]
    for name,part in frame.groupby('Model',sort=False):
        row=part.iloc[0].to_dict()
        for k in ['Preprocess_Mean_ms','Backbone_Mean_ms','PCA_Adapter_Mean_ms','End_to_End_Mean_ms']:
            row[k]=float(part[k].mean())
        for k in ['trial','order_in_trial','End_to_End_Std_ms']:row.pop(k,None)
        row.update(Passes=5,Frames_timed_total=180,Pass_mean_latency_SD_ms=float(part.End_to_End_Mean_ms.std(ddof=1)))
        row['FPS']=1000/row['End_to_End_Mean_ms'];summary.append(row)
    pd.DataFrame(summary).to_csv(a.output/'benchmark_summary.csv',index=False)
    dump(a.output/'completed.json',{'status':'complete','passes_per_model':5,'frames_per_pass':36,
         'batch_size':1,'warmup_frames_per_pass':5,'load_time_included':False,
         'included':'RGB preprocessing, transfer, frozen backbone, pooling, CPU transfer, PCA/scalers, three physical adapters and residual head',
         'excluded':'disk decode, acquisition, annotations, geometry, FEM solve, rendering, model loading',
         'FPS_definition':'1000 / mean of all timed per-frame latencies'})

if __name__=='__main__':main()
