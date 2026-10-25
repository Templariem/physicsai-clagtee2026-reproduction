"""Check dataset identities, grouped split and trained artifacts without modifying them."""
from pathlib import Path
import json,hashlib,collections
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/field'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=json.loads((F/'config/manifest_docente.json').read_text());assert len(rows)==1143
assert dict(collections.Counter(r['split'] for r in rows))==dict(train=926,val=99,test=118)
assert len({r['id'] for r in rows})==len(rows)
groups=collections.defaultdict(set)
for r in rows:
 groups[r['group']].add(r['split'])
 for k in ['image','mask','teacher']:assert sha(F/r[k])==r[k+'_sha256'],(r['id'],k)
assert all(len(v)==1 for v in groups.values())
lock=json.loads((F/'revision/congelacion_prueba.json').read_text())['hashes']
weights={p:h for p,h in lock.items() if p.startswith('modelos/') or p.startswith('physical/modelos/')}
for p,h in weights.items():assert sha(F/p)==h,p
manifest=json.loads((ROOT/'asset_manifest.json').read_text());checked=0
for p,rec in manifest['files'].items():
 if (ROOT/p).exists():assert sha(ROOT/p)==rec['sha256'],p;checked+=1
out=ROOT/'runs';out.mkdir(exist_ok=True);report=dict(status='passed',images=len(rows),splits=dict(collections.Counter(r['split'] for r in rows)),groups=len(groups),weights=len(weights),asset_files_checked=checked)
(out/'audit.json').write_text(json.dumps(report,indent=2));print(report)
