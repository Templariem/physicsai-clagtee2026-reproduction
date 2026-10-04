"""Download the pinned backbone weights from their official Hugging Face repositories."""
import argparse
from huggingface_hub import snapshot_download
from .runtime import model_spec

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',choices=['both','dinov3','ijepa'],default='both')
    a=p.parse_args()
    for kind in (['dinov3','ijepa'] if a.backbone=='both' else [a.backbone]):
        spec=model_spec(kind)
        path=snapshot_download(spec['id'],revision=spec['revision'],
             allow_patterns=['config.json','preprocessor_config.json','*.safetensors','*.safetensors.index.json'])
        print(f'{kind}: {path}',flush=True)

if __name__=='__main__':main()
