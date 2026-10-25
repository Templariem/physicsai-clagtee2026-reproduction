"""Download official encoders at pinned revisions and verify the training weight hashes.
DINOv3 requires acceptance of Meta's license/access terms and a personal HF token.
I-JEPA is noncommercial; see licenses/IJEPA-LICENSE.txt. No credentials are stored here.
"""
from pathlib import Path
import json,hashlib,os
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];out=Path(os.environ.get('PHYSICS_ENCODERS',ROOT/'encoders'))
repos={'dinov3':('facebook/dinov3-vits16-pretrain-lvd1689m','114c1379950215c8b35dfcd4e90a5c251dde0d32'),'ijepa':('facebook/ijepa_vith14_1k','f157467ea509bc356ff9f61fd3c0d840eec5e04e')}
expected=json.loads((ROOT/'experiments/field/config/encoders.json').read_text())
for kind,(repo,revision) in repos.items():
 snapshot_download(repo,revision=revision,local_dir=out/kind,allow_patterns=['config.json','preprocessor_config.json','model.safetensors'],token=os.environ.get('HF_TOKEN'))
 h=hashlib.sha256()
 with (out/kind/'model.safetensors').open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==expected[kind]['weights_sha256'],kind
 print(kind,'verified')
