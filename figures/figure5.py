"""Compact Figure 5: unchanged examples and numeric maps, shared labels/scales."""
import hashlib
import json
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import figure5_fields as f

R=f.R
(R/"runs/figures").mkdir(parents=True,exist_ok=True)
labels=['RGB image', 'FEM\nMesh', 'FEM\nPotential + RGB',
        'DINOv3\nLatent (PCA-128)', 'I-JEPA\nLatent (PCA-128)',
        'Physical Adapter\nField magnitude', 'DINOv3\nField magnitude',
        'I-JEPA\nField magnitude', 'Physical Adapter\nPotential',
        'DINOv3\nPotential', 'I-JEPA\nPotential']
tile=.85
ystep=.875
group_gap=.14
block_gap=.23
left=2.32
bottom=.18
nrows=len(labels)
block_width=12*tile+2*group_gap
fw=left+2*block_width+block_gap+1.02
fh=bottom+nrows*ystep+1.0
fig=plt.figure(figsize=(fw,fh))
panel_records=[]

for arm_index,arm in enumerate(['residual','full']):
    bx=left+arm_index*(block_width+block_gap)
    for j,case in enumerate(f.cases):
        x=bx+j*tile+(j//4)*group_gap
        a=f.data[case['id']]; e=f.field[case['id']]
        keys=['physical','dinov3_'+arm,'ijepa_'+arm]
        for row in range(nrows):
            ax=fig.add_axes([x/fw,(bottom+(nrows-1-row)*ystep)/fh,tile/fw,tile/fh])
            ax.axis('off')
            if row==1:
                f.mesh_panel(ax,case)
            elif row in (0,2,3,4):
                old_row={0:0,2:1,3:4,4:5}[row]
                ax.imshow(Image.open(R/'reference/figure_inputs/panels'/f'{arm_index*108+j*9+old_row:03d}.png'))
            elif row<8:
                f.scalar(ax,e[keys[row-5]],e['mask'],case['crop'],f.field_norm,'inferno')
            else:
                f.scalar(ax,a[keys[row-8]],a['valid'].astype(bool),case['crop'],f.potential_norm,'turbo')
        fig.text((x+tile/2)/fw,(bottom+nrows*ystep+.035)/fh,
                 ['IP+LED','ST','IP+RULE','WB'][j%4],ha='center',fontsize=12)
        panel_records.append({'arm':arm,'id':case['id'],'x':x,'width':tile,'split':case['split']})
    for group,title in enumerate(['Train','Validation','Test']):
        gx=bx+group*(4*tile+group_gap)
        fig.text((gx+2*tile)/fw,(bottom+nrows*ystep+.32)/fh,title,
                 ha='center',fontsize=15,fontweight='bold')
        fig.add_artist(plt.Line2D([gx/fw,(gx+4*tile)/fw],
            [(bottom+nrows*ystep+.235)/fh]*2,transform=fig.transFigure,color='.4',linewidth=.6))
    title='(a) Frozen-encoder residual adapters' if arm=='residual' else '(b) Full fine-tuning (Full FT)'
    fig.text((bx+block_width/2)/fw,(bottom+nrows*ystep+.68)/fh,
             title,ha='center',fontsize=16,fontweight='bold')

for row,label in enumerate(labels):
    fig.text((left-.09)/fw,(bottom+(nrows-1-row)*ystep+tile/2)/fh,
             label,ha='right',va='center',fontsize=14,fontweight='bold',linespacing=1.12)

# One shared scale for each physical quantity; no colorbar between the blocks.
bar_x=left+2*block_width+block_gap+.12
for low,norm,cmap,label,extend in [
    (3,f.field_norm,'inferno',r'$|E_p|$ (eq. V/pixel)','max'),
    (0,f.potential_norm,'turbo','LED-equivalent V','both')]:
    cax=fig.add_axes([bar_x/fw,(bottom+low*ystep)/fh,.12/fw,(3*ystep-.025)/fh])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,extend=extend)
    cb.ax.tick_params(labelsize=10,pad=2)
    cb.set_label(label,fontsize=12)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for artist in fig.findobj(plt.Text):
    if artist.get_visible() and artist.get_text():
        bbox=artist.get_window_extent(renderer)
        assert bbox.x0 >= 0 and bbox.x1 <= fig.bbox.width and bbox.y0 >= 0 and bbox.y1 <= fig.bbox.height, (artist.get_text(), bbox)

path=R/'runs/figures/figure5.pdf'
fig.savefig(path,dpi=300)
fig.savefig(R/'runs/figures/figure5.png',dpi=170)
plt.close(fig)

audit=R/'runs/figure5'
audit.mkdir(exist_ok=True)
(audit/'procedencia.json').write_text(json.dumps({
    'rows':labels,'cases':f.cases,'panel_positions':panel_records,
    'tile_inches':tile,'within_group_gap_inches':0,'between_group_gap_inches':group_gap,
    'between_block_gap_inches':block_gap,'source_color':'#ff9500',
    'field_scale':[0,f.field_limit],'potential_scale':[.1,8],
    'field_computation':'Same 448x448 gradients, masks and train-derived scale as figura5_campo.py',
    'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((f.A/'field_maps').glob('*.npz'))},
    'mesh_provenance':'reference/figure_inputs/meshes/procedencia.json',
    'rgb_overlay':'Original FEM target over RGB, retained without changes',
    'dimensions_inches':[fw,fh]
},indent=2)+'\n')
print(path)
