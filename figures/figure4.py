from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[1]
P=R
F=R/"runs/figures";F.mkdir(parents=True,exist_ok=True)
# Keep the original two-panel calibration/parity layout and P1 calibration panel.
sys.path.insert(0,str(P/'figures'))
source=(P/'figures/calibration_base.py').read_text()
source=source.replace("triangles=np.vstack([e[:,[0,1,2]],e[:,[0,2,3]]])","triangles=e")
source=source.replace("s=23,facecolors=face","s=8,facecolors=face").replace("alpha=.8,linewidths=.9","alpha=.45,linewidths=.45")
source=source.replace("label=name,markersize=5","label=(name if name=='Physical' else name+' full'),markersize=5")
source=source.replace("label='Within 5-17 cm (15)'","label='Supported amplitude'").replace("label='Extrapolated (21)'","label='Extrapolated amplitude'")
source=source.replace("xlabel='Five-LED fitted target (V)'","xlabel='P1 + LED target (equivalent V)'")
source=source.replace("title=f'(b) Videos 8-9 holdout ({len(y)} frames)'","title='(b) Real test: 92 frames, 736 air queries'")
source=source.replace("'axes.titlesize':10", "'axes.titlesize':13,'axes.titleweight':'bold'")
# Overlay the exact P1 triangulation supplying this panel, not a display remesh.
source=source.replace("from matplotlib.lines import Line2D", "from matplotlib.lines import Line2D\nfrom matplotlib.collections import LineCollection")
needle="    for d,v,c in zip(DISTANCE_CM,VOLTAGE_V,"
overlay="""    bounds=n[triangles]
    visible=((bounds[:,:,0].max(1)>=0)&(bounds[:,:,0].min(1)<=.30)&
             (bounds[:,:,1].max(1)>=0)&(bounds[:,:,1].min(1)<=.30))
    selected=triangles[visible]
    edges=np.unique(np.sort(np.concatenate([selected[:,[0,1]],selected[:,[1,2]],selected[:,[2,0]]]),axis=1),axis=0)
    a.add_collection(LineCollection(n[edges]*100,colors='black',linewidths=.08,alpha=.18,zorder=1.1))
"""
assert needle in source
source=source.replace(needle,overlay+needle)
# Conductor silhouettes, LED contours and markers remain above the transparent mesh.
source=source.replace("color='silver',edgecolor='black')", "color='silver',edgecolor='black',zorder=3)")
source=source.replace("color='#bb8647',edgecolor='black')", "color='#bb8647',edgecolor='black',zorder=3)")
ns={'__file__':str(P/'figures/calibration_base.py'),'__name__':'paper_calibration'}
exec(compile(source,'paper_calibration','exec'),ns)
ns['BASE']=P/'fem/results/P1';ns['OUTPUT']=F
records=json.loads((R/'reference/figure_inputs/figure4_queries.json').read_text())['records']
assert len(records)==736 and len({r['id'] for r in records})==92
ns['calibration_plot'](np.array([r['target'] for r in records]),{k:np.array([r[k] for r in records]) for k in ['Physical','DINOv3','I-JEPA']},np.array([r['supported'] for r in records]))

