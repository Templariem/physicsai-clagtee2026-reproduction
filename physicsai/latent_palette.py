"""Display-only DINO RGB palette; never applied to embeddings or model inputs.

The fixed affine RGB transform aligns the three visualization channels with
the historical green/orange/purple palette. Its coefficients were obtained
from the eight development thumbnails only; holdout colors and regression
labels were not used. The same transform is used for both DINO display rows
and every frame. The underlying official DINOv3 projections stay unchanged.
"""
import json
from pathlib import Path
import numpy as np

PALETTE_PATH = Path(__file__).resolve().parents[1]/'reference/figures/dino_display_palette.json'

def apply_dino_palette(tile):
    """Map an RGB PCA thumbnail to the fixed display palette, clipping RGB only."""
    spec=json.loads(PALETTE_PATH.read_text())
    rgb=np.asarray(tile,dtype=np.float64)/255.0
    mapped=rgb@np.asarray(spec['matrix'])+np.asarray(spec['offset'])
    return np.rint(np.clip(mapped,0.0,1.0)*255.0).astype(np.uint8)
