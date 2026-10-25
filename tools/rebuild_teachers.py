"""Rebuild accepted P1 targets only in a new directory. Does not train models."""
from pathlib import Path
import argparse,shutil,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/field'
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();dest=a.destination.resolve();assert not dest.exists() and not dest.is_relative_to(ROOT)
for name in ['config','scripts','solver']:shutil.copytree(F/name,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['imagenes','mascaras']:shutil.copytree(F/'datos'/name,dest/'datos'/name)
for name in ['revision','datos/docente']:(dest/name).mkdir(parents=True,exist_ok=True)
rows=json.loads((dest/'config/manifest_docente.json').read_text())
computed=['teacher','teacher_sha256','scene','transform','valid_pixels','air_pixels','content_coverage','mesh_metrics']
for r in rows:
 for k in computed:r.pop(k,None)
(dest/'config/manifest.json').write_text(json.dumps(rows,indent=2));(dest/'config/manifest_docente.json').unlink()
subprocess.run([sys.executable,str(dest/'scripts/preparar_docente.py')],check=True)
