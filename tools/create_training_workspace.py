"""Make a distinct experiment copy for OPTIONAL retraining. Never train on invocation."""
from pathlib import Path
import argparse,shutil
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();dest=a.destination.resolve()
assert not dest.exists(),'Choose a new directory; existing paths are never overwritten'
assert not dest.is_relative_to(ROOT),'Use a sibling/separate folder'
shutil.copytree(ROOT/'experiments/field',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','resultados','revision','modelos','geometria'))
for sub in ['modelos','resultados','revision','logs','datos/geometria','physical/modelos','physical/resultados','physical/revision']:(dest/sub).mkdir(parents=True,exist_ok=True)
# Frozen no-ruler FEM prior is a numerical input; no trained model is copied.
shutil.copy2(ROOT/'experiments/field/physical/modelos/base_sin_regla_v2.npz',dest/'physical/modelos/base_sin_regla_v2.npz')
shutil.copy2(ROOT/'experiments/field/modelos/mapa_promedio_train.npz',dest/'modelos/mapa_promedio_train.npz')
print('Created',dest,'without trained weights or frozen-run guards. No training started.')
