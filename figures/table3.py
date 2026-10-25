"""Table III: immutable accuracy results plus measured model size and inference time."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[1]
V=R/'experiments/field'
(R/"runs").mkdir(exist_ok=True)
rows={r['model']:r for r in csv.DictReader((V/'resultados/tabla_comparacion.csv').open())}
cost=json.loads((R/'reference/inference_homelab.json').read_text())['results']
order=['physical','dinov3_residual','ijepa_residual','dinov3_frozen','dinov3_partial','dinov3_full','ijepa_frozen','ijepa_partial','ijepa_full']
names=['Physical Adapter','DINOv3 residual','I-JEPA residual','DINOv3 frozen','DINOv3 partial','DINOv3 full','I-JEPA frozen','I-JEPA partial','I-JEPA full']
lines=[r'\begin{table*}[!t]',r'\centering\caption{Potential errors against FEM targets (LED-equivalent V; lower is better), model size and inference time. Recorded-video errors equally weight four reserved videos (92 frames); 26 synthetic ST images are scored separately. FEM supplies targets, not independent accuracy.}',r'\label{tab:all_models}',r'\setlength{\tabcolsep}{3pt}\footnotesize',r'\begin{tabular}{llrrrrrrrrrr}\toprule',r'Method & Input & Parameters & Inference & \multicolumn{4}{c}{Test on recorded videos} & \multicolumn{4}{c}{MAE by image source}\\',r'\cmidrule(lr){5-8}\cmidrule(lr){9-12}',r' & & (M) & (ms) & MAE & RMSE & \shortstack{Perturbed\\region MAE} & \shortstack{Ruler\\MAE} & IP+LED & IP+RULE & WB & ST\\\midrule']
keys=['air_mae_V','air_rmse_V','effect_mae_V','grounded_rule_mae_V','legacy_led_air_mae_V','iphone_air_mae_V','webcam_air_mae_V','synthetic_air_mae_V']
for i,(key,name) in enumerate(zip(order,names)):
    if i==3:lines.append(r'\midrule')
    inp='Geometry' if i==0 else (r'RGB + geometry' if i<3 else 'RGB')
    values=[f'{float(rows[key][k]):.4f}' for k in keys]
    size=f"{cost[key]['total_parameters']/1e6:.3f}";latency=f"{cost[key]['median_ms']:.1f}"
    lines.append(name+' & '+inp+' & '+size+' & '+latency+' & '+' & '.join(values)+r'\\')
lines.extend([r'\bottomrule\end{tabular}',r'\par\smallskip\begin{minipage}{.98\textwidth}\footnotesize Geometry: reviewed masks; RGB: color image. MAE/RMSE: mean absolute/root mean squared error outside conductors and padding. Perturbed-region MAE: air pixels changed $>0.1$ equivalent V from the no-ruler target; ruler MAE: ruler pixels. Geometry-assisted methods impose conductor values; physical FEM reuse explains near-zero IP+LED error. Parameters include frozen encoders and physical ensemble (M: millions). Timing: RTX 5060 Ti, $448\times448$, batch 1, 100-run median; residuals include physical inference. Excludes preprocessing, transfers, detection, FEM solving and rendering.\end{minipage}',r'\end{table*}'])
(R/'runs/table3.tex').write_text('\n'.join(lines)+'\n')
