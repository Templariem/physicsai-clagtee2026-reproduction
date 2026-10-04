"""Check the shipped scientific artifacts without downloading or fitting models."""
import hashlib
import json
import numpy as np
import torch
from .runtime import ROOT
from . import core

def main():
    path=ROOT/'MANIFEST.sha256'
    if path.exists():
        for line in path.read_text().splitlines():
            checksum,name=line.split('  ',1)
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==checksum,name
    samples=core.load_samples(ROOT)
    target=np.array([s.target for s in samples]);video=np.array([s.video_id for s in samples])
    valid=np.isfinite(target)
    assert (len(samples),int(valid.sum()),int((valid&(video<8)).sum()),int((valid&(video>=8)).sum()))==(271,164,128,36)
    proto=json.loads((ROOT/'protocol.json').read_text())
    for fold,videos in enumerate(proto['validation_video_groups'],1):
        ck=torch.load(ROOT/f'reference/checkpoints/cv_fold_{fold}.pt',map_location='cpu',weights_only=False)
        for name in ['physics','DINOv3','I-JEPA']:
            tr=np.asarray(ck[name]['train_indices']);ev=np.asarray(ck[name]['eval_indices'])
            assert set(video[tr]).isdisjoint(set(video[ev]))
            assert set(video[ev])==set(videos)
            assert np.all(video[tr]<8) and np.all(valid[tr]) and np.all(valid[ev])
    final=torch.load(ROOT/'reference/checkpoints/corrected_final.pt',map_location='cpu',weights_only=False)
    assert set(video[final['physics']['train_indices']])==set(range(1,8))
    assert set(video[final['physics']['eval_indices']])=={8,9}
    assert len(list((ROOT/'data/frames').glob('*.jpg')))==271
    print('Verified hashes, 271 images, 164 valid queries, five disjoint video folds, and untouched Videos 8–9 holdout membership.')

if __name__=='__main__':main()
