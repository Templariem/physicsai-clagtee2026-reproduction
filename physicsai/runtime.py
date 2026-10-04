"""Device and environment handling shared by all reproduction commands."""
from pathlib import Path
import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')

def choose_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install the NVIDIA-compatible PyTorch build; see README.')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable in this Python process.')
    torch.set_num_threads(4)
    if name == 'cuda':
        # FP32, as in the paper. TF32/AMP would introduce a different protocol.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    return torch.device(name)

def environment(device):
    packages = ['torch','torchvision','transformers','numpy','scipy','scikit-learn',
                'pandas','opencv-python','Pillow','matplotlib','huggingface-hub']
    hardware = platform.machine()
    if sys.platform == 'darwin':
        try:
            hardware = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            hardware += ' (hardware query unavailable)'
    result = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'platform':platform.platform(),'python':sys.version,'device':str(device),
              'hardware':hardware,'torch_threads':torch.get_num_threads(),
              'packages':{p:importlib.metadata.version(p) for p in packages},
              'cuda_runtime':torch.version.cuda,'FP32':True,'TF32':False,'AMP':False}
    if device.type == 'cuda':
        p=torch.cuda.get_device_properties(device)
        result.update(gpu=p.name,compute_capability=list(torch.cuda.get_device_capability(device)),
                      gpu_memory_bytes=p.total_memory,compiled_architectures=torch.cuda.get_arch_list())
    return result

def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def model_spec(kind):
    return json.loads((ROOT/'protocol.json').read_text())['models'][kind]

def load_backbone(kind, device, offline=False):
    from transformers import AutoImageProcessor, AutoModel, IJepaModel
    spec=model_spec(kind)
    kw={'revision':spec['revision'],'local_files_only':offline}
    processor=AutoImageProcessor.from_pretrained(spec['id'],**kw)
    cls=AutoModel if kind=='dinov3' else IJepaModel
    backbone=cls.from_pretrained(spec['id'],**kw).to(device).eval()
    backbone.requires_grad_(False)
    return processor,backbone

def pooled(output,kind,samples,indices):
    import numpy as np
    from .core import weighted_local
    if kind=='dinov3':
        glob=output.pooler_output.detach().cpu().numpy()
        tokens,grid=output.last_hidden_state[:,5:,:],14
    else:
        tokens,grid=output.last_hidden_state,16
        glob=tokens.mean(dim=1).detach().cpu().numpy()
    return np.concatenate([glob,weighted_local(tokens,samples,indices,grid)],axis=1)
