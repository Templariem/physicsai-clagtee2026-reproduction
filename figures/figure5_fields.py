"""Figure 5 only: retained PCA panels, numeric potential and image-plane gradients.

No training, normalization per image, smoothing, or differentiation of colored tiles.
Gradients are computed on the 448-pixel inference grid before display cropping.
"""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from scipy.ndimage import binary_erosion

R = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--fem-rgb',action='store_true',help='Separate preview with an extra FEM/RGB row; leave approved outputs untouched.')
extra_fem_rgb=parser.parse_args().fem_rgb if __name__ == '__main__' else False
V = R / 'reference/figure_inputs'
A = R / 'reference/figure_inputs'
cases = json.loads((V / 'cases.json').read_text())['cases']
data = {r['id']: dict(np.load(A / 'field_maps' / (r['id'] + '.npz'))) for r in cases}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
potential_norm = plt.Normalize(.1, 8)
field = {}
training_values = []
for r in cases:
    a = data[r['id']]
    valid = a['valid'].astype(bool)
    # Exclude padding and any central-difference stencil that touches padding.
    stencil = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    air = a['air'].astype(bool) & binary_erosion(valid, structure=stencil)
    field[r['id']] = {'mask': binary_erosion(valid, structure=stencil)}
    for key in ['physical', 'dinov3_residual', 'ijepa_residual', 'dinov3_full', 'ijepa_full']:
        v = a[key].astype('float64')
        assert v.shape == (448, 448) and np.isfinite(v).all()
        gy, gx = np.gradient(v, 1., 1., edge_order=2)
        e = np.hypot(gx, gy)
        field[r['id']][key] = e
        if r['split'] == 'train':
            x, y, side = r['crop']
            z = e[y:y+side, x:x+side]; m = air[y:y+side, x:x+side]
            training_values.append(z[m])
# A single robust display bound, fixed from TRAIN examples only, for every arm/model.
raw_limit = float(np.percentile(np.concatenate(training_values), 99.5))
field_limit = float(f'{raw_limit:.2g}')
field_norm = plt.Normalize(0, field_limit)
labels = ['RGB', 'FEM P1\nmesh', 'DINOv3\nPCA-128', 'I-JEPA\nPCA-128',
          'Physical\n$|E_p|$', 'DINOv3\n$|E_p|$', 'I-JEPA\n$|E_p|$',
          'Physical\n(eq. V)', 'DINOv3\n(eq. V)', 'I-JEPA\n(eq. V)']
if extra_fem_rgb: labels.insert(2,'FEM P1\n+ RGB')

def scalar(ax, values, mask, crop, norm, cmap):
    x, y, side = crop
    arr = np.ma.array(values[y:y+side, x:x+side], mask=~mask[y:y+side, x:x+side])
    colors = plt.get_cmap(cmap).copy(); colors.set_bad('#eeeeee')
    ax.imshow(arr, norm=norm, cmap=colors, interpolation='nearest', rasterized=True)

def mesh_panel(ax, case):
    z=np.load(R/'reference/figure_inputs/meshes'/(case['id']+'.npz'))
    pts=z['pixels']; tri=z['triangles']; x,y,side=case['crop']
    # Keep actual edges of intersecting elements, without coarsening or retriangulation.
    bounds=pts[tri]
    visible=((bounds[:,:,0].max(1)>=x-.5)&(bounds[:,:,0].min(1)<=x+side-.5)&
             (bounds[:,:,1].max(1)>=y-.5)&(bounds[:,:,1].min(1)<=y+side-.5))
    subset=tri[visible]
    edges=np.unique(np.sort(np.concatenate([subset[:,[0,1]],subset[:,[1,2]],subset[:,[2,0]]]),axis=1),axis=0)
    ax.set_facecolor('white')
    ax.add_collection(LineCollection(pts[edges],colors='#8d959c',linewidths=.12))
    all_edges,counts=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    boundary=all_edges[counts==1]
    for key,color,outline in [('source','#ff9500','#b75a00'),('ruler','#8ebee5','#276393')]:
        selected=boundary[np.isin(boundary,z[key]).all(1)]
        if not len(selected): continue
        neighbors={}
        for i,j in selected:
            neighbors.setdefault(int(i),[]).append(int(j));neighbors.setdefault(int(j),[]).append(int(i))
        assert all(len(v)==2 for v in neighbors.values())
        remaining=set(neighbors)
        while remaining:
            first=min(remaining);loop=[first];previous=None;current=first
            while True:
                nxt=next(n for n in neighbors[current] if n!=previous)
                if nxt==first:break
                loop.append(nxt);previous,current=current,nxt
                assert len(loop)<=len(neighbors)
            remaining.difference_update(loop)
            ax.add_patch(Polygon(pts[loop],closed=True,facecolor=color,edgecolor=outline,linewidth=.4))
    ax.set_xlim(x-.5,x+side-.5);ax.set_ylim(y+side-.5,y-.5);ax.set_aspect('equal')
