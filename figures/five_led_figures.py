"""Current Figure 4: one five-LED fitted law for contour remapping and labels."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from voltage_display import VOLTAGE_NORM, VOLTAGE_CMAP, VOLTAGE_TICKS

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from physicsai.calibration import voltage, DISTANCE_CM, VOLTAGE_V

OUTPUT=BASE/'runs/figures'
OUTPUT.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.titlesize':10,'axes.labelsize':9,
                     'pdf.fonttype':42,'ps.fonttype':42})


def calibration_plot(y,predictions,within):
    data=np.load(BASE/'data/fem_prior.npz')
    n,e=data['nodes'],data['elements']
    triangles=np.vstack([e[:,[0,1,2]],e[:,[0,2,3]]])
    fem=mtri.LinearTriInterpolator(mtri.Triangulation(n[:,0],n[:,1],triangles),data['U_abs'])
    r_s,y_s=.0225,.1225
    xp=np.linspace(r_s+.0001,4.9,1000);up=np.asarray(fem(xp,np.full_like(xp,y_s)))
    dp=(xp-r_s)*100;order=np.argsort(up)
    u_to_d=interp1d(up[order],dp[order],bounds_error=False,fill_value=(dp[-1],dp[0]))
    xg,yg=np.meshgrid(np.linspace(0,.30,400),np.linspace(0,.30,400))
    distance=u_to_d(np.asarray(fem(xg,yg)))
    z=voltage(distance)
    z=np.ma.masked_where((np.hypot(xg,yg-y_s)<=r_s)|((xg<=.015)&(yg<=.10)),z)
    fig,(a,b)=plt.subplots(1,2,figsize=(11.3,4.5),layout='constrained')
    cf=a.pcolormesh(xg*100,yg*100,z,norm=VOLTAGE_NORM,cmap=VOLTAGE_CMAP,shading='nearest',rasterized=True)
    for d,v,c in zip(DISTANCE_CM,VOLTAGE_V,['white','#4297ff','#31c459','#ffd43b','#f04e4e']):
        a.contour(xg*100,yg*100,z,levels=[v],colors=[c],linewidths=1.2)
        a.scatter(d+2.25,y_s*100,c=c,edgecolors='black',s=30,zorder=5)
    theta=np.linspace(-np.pi/2,np.pi/2,100)
    a.fill(2.25*np.cos(theta),12.25+2.25*np.sin(theta),color='silver',edgecolor='black')
    a.fill([0,1.5,1.5,0],[0,0,10,10],color='#bb8647',edgecolor='black')
    a.set(xlim=(0,30),ylim=(0,30),xlabel='x (cm)',ylabel='y (cm)',title='(a) FEM contours with five-LED fitted remapping')
    a.set_aspect('equal')
    fig.colorbar(cf,ax=a,label='Calibration-derived voltage (V)',shrink=.85,ticks=VOLTAGE_TICKS,extend='max')
    all_values=np.concatenate([y,*predictions.values()])
    low=.5*np.floor((all_values.min()-.1)/.5);high=.5*np.ceil((all_values.max()+.1)/.5)
    b.plot([low,high],[low,high],'--',color='.35',lw=1)
    legend=[]
    for name,c,m in [('Physical','#159447','s'),('DINOv3','#2468d8','o'),('I-JEPA','#8b4bc1','^')]:
        for mask,face in [(within,c),(~within,'none')]:
            b.scatter(y[mask],predictions[name][mask],s=23,facecolors=face,edgecolors=c,marker=m,alpha=.8,linewidths=.9)
        legend.append(Line2D([],[],marker=m,color=c,linestyle='none',label=name,markersize=5))
    legend.extend([Line2D([],[],marker='o',color='.25',linestyle='none',label='Within 5-17 cm (15)',markersize=5),
                   Line2D([],[],marker='o',color='.25',markerfacecolor='none',linestyle='none',label='Extrapolated (21)',markersize=5)])
    b.set(xlim=(low,high),ylim=(low,high),xlabel='Five-LED fitted target (V)',ylabel='Predicted surrogate voltage (V)',title=f'(b) Videos 8-9 holdout ({len(y)} frames)')
    b.set_aspect('equal');b.legend(handles=legend,fontsize=7.5,loc='lower right');b.grid(alpha=.2)
    fig.savefig(OUTPUT/'fig4_calibration_holdout.pdf')
    fig.savefig(OUTPUT/'fig4.png',dpi=180);plt.close(fig)
    return {'mapping':'five-LED fitted power law applied to surface distances associated with FEM contour magnitudes',
            'within_interval_holdout':int(within.sum()),'extrapolated_holdout':int((~within).sum()),
            'display_range_V':[.1,8.],'targets_clipped':False}
