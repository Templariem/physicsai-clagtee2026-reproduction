from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'runs/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42})

def triangles(e):
    return np.vstack([e[:, [0,1,2]], e[:, [0,2,3]]])

def reference_fields(nodes):
    x,y=nodes.T;r=np.hypot(x,y-0.5)
    return [1000*np.log(5/r)/np.log(25),x*x-y*y,np.exp(-(1+1j)*x),
            np.where(.8-x>0,np.sin(5*(.8-x))**3,0)]

def plot_verification(n=None,e=None,exacts=None):
    """Replot the same analytical profiles without rerunning the FEM solver."""
    if n is None:
        d=np.load(BASE/'fem/gmsh_meshes/exp_1_mesh_transfinite.npz')
        n,e=d['nodes'],d['elements']
    if exacts is None:
        exacts=reference_fields(n)
    x,y=n.T
    fig=plt.figure(figsize=(12.0,2.2))
    panels=[]
    for col,(vals,title,label,cmap) in enumerate(zip(
            [exacts[0],exacts[1],exacts[2].real,exacts[3]],
            ['(a) Electrostatic','(b) Magnetostatic','(c) Harmonic','(d) Transient, t = 0.8'],
            ['$V$ (V)','$A_z/A_0$\n(dimensionless)',
             r'$\mathrm{Re}(\hat A_z/A_0)$'+'\n(dimensionless)','$u$\n(dimensionless)'],
            ['magma','viridis','inferno','cividis'])):
        # Fixed panel/color-bar slots keep units separate and aligned.
        ax=fig.add_axes([(.48+3*col)/12,.40/2.2,1.5/12,1.5/2.2])
        cf=ax.tricontourf(x,y,triangles(e),vals,levels=50,cmap=cmap)
        # Rasterize only the filled contours to prevent PDF polygon hairlines.
        cf.set_rasterized(True)
        ax.set_aspect('equal');ax.set_title(title,pad=5,fontweight='bold')
        ax.set_xlabel('x (m)',labelpad=3);ax.set_ylabel('y (m)',labelpad=3)
        ax.set_xticks(np.linspace(0,1,5))
        ax.tick_params(labelsize=9,pad=2)
        cax=fig.add_axes([(2.04+3*col)/12,.40/2.2,.075/12,1.5/2.2])
        cb=fig.colorbar(cf,cax=cax,orientation='vertical')
        cb.set_label(label,fontsize=9,labelpad=4)
        cb.ax.tick_params(labelsize=9,pad=2)
        cb.locator=MaxNLocator(nbins=4);cb.update_ticks()
        panels.append({'title':title,'variable':label.replace('\n',' '),'colormap':cmap,
                       'min':float(np.min(vals)),'max':float(np.max(vals)),
                       'contour_levels':cf.levels.tolist()})
    fig.savefig(OUT/'fig1_verification.pdf',dpi=450)
    fig.savefig(OUT/'fig1.png',dpi=180)
    plt.close(fig)
    return {'figure_size_inches':[12.0,2.2],'colorbars':'vertical, right of each panel',
            'filled_contour_raster_dpi':450,'text_and_axes':'vector','panel_title_fontweight':'bold',
            'same_reference_fields_and_contour_levels':True,'panels':panels}


if __name__=="__main__":plot_verification()
