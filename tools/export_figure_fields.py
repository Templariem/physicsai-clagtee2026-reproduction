"""Read-only checkpoint inference for selected figure cases; outputs to a separate directory."""
from pathlib import Path
import sys, json, hashlib, gc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]/'experiments/field'
sys.path.insert(0, str(ROOT / 'scripts'))
from datos import Data
from modelo_factory import Model

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
selected = json.loads((ROOT / 'config/figura_seleccion.json').read_text())
cal = np.load(ROOT / 'config/transformacion_led.npz')
records = {}
for row in selected:
    rgb, a, _ = Data([row], geometry=True)[0]
    teacher = np.load(ROOT / row['teacher'])
    raw = a['raw'].numpy().transpose(1, 2, 0)
    records[row['id']] = dict(
        rgb=rgb.numpy().transpose(1, 2, 0),
        air=teacher['air'], valid=teacher['valid'], rule=teacher['rule'],
        reference=np.interp(teacher['phi'], cal['phi'], cal['voltage']),
        physical=np.interp(raw[:, :, 10], cal['phi'], cal['voltage']))

provenance = {'models': {}, 'cases': [r['id'] for r in selected],
              'precision': 'bfloat16 autocast, matching original figure script',
              'original_models_and_results_modified': False}
for kind in ['dinov3', 'ijepa']:
    for arm in ['residual', 'full']:
        key = kind + '_' + arm
        path = ROOT / 'modelos' / (key + '.pt')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        ck = torch.load(path, map_location='cpu', weights_only=True)
        m = Model(kind, arm, cal['voltage'], False); m.restore(ck); m.cuda().eval()
        origin = {}
        with torch.inference_mode():
            for row in selected:
                cache = ROOT / 'resultados' / ('predicciones_' + key) / (row['id'] + '.npz')
                if cache.exists():
                    value = np.load(cache)['voltage']
                    origin[row['id']] = 'existing evaluated prediction'
                else:
                    rgb, a, _ = Data([row], geometry=arm == 'residual')[0]
                    if arm == 'residual': m.head.context = a['raw'][None].cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        value = m(rgb[None].cuda())[0, 0].float().cpu().numpy()
                    origin[row['id']] = 'checkpoint inference, no augmentation'
                assert value.shape == (448, 448) and np.isfinite(value).all()
                records[row['id']][key] = value
        provenance['models'][key] = {'checkpoint_sha256': digest.hexdigest(), 'origin': origin}
        del m, ck; gc.collect(); torch.cuda.empty_cache()
        print('Exported', key, flush=True)
for ident, values in records.items(): np.savez_compressed(out / (ident + '.npz'), **values)
(out / 'procedencia.json').write_text(json.dumps(provenance, indent=2) + '\n')
print('COMPLETE', out, flush=True)
