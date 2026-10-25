"""Download checked split release archives; extract only manifest-listed regular files."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,shutil,tarfile,tempfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('groups',nargs='*',default=['dataset','models','geometry','predictions']);p.add_argument('--cache',type=Path,default=ROOT/'.downloads');p.add_argument('--local-assets',type=Path);a=p.parse_args()
 m=json.loads((ROOT/'asset_manifest.json').read_text());a.cache.mkdir(parents=True,exist_ok=True)
 for group in a.groups:
  info=m['groups'][group];parts=[]
  for part in info['parts']:
   dest=(a.local_assets or a.cache)/part['name']
   if not dest.exists() or dest.stat().st_size!=part['size'] or sha(dest)!=part['sha256']:
    if a.local_assets:raise ValueError('Local asset checksum failed: '+str(dest))
    url='https://github.com/Templariem/physicsai-clagtee2026-reproduction/releases/download/'+m['release_tag']+'/'+part['name']
    tmp=dest.with_suffix(dest.suffix+'.download')
    with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f,8*1024*1024)
    assert tmp.stat().st_size==part['size'] and sha(tmp)==part['sha256'],dest
    tmp.replace(dest)
   parts.append(dest)
  joined=a.cache/(group+'.tar.gz')
  with joined.open('wb') as f:
   for part in parts:
    with part.open('rb') as src:shutil.copyfileobj(src,f,8*1024*1024)
  assert sha(joined)==info['sha256']
  with tarfile.open(joined,'r:gz') as tar:
   for member in tar:
    name=member.name;dest=ROOT/name
    assert member.isfile() and dest.resolve().is_relative_to(ROOT.resolve()),name
    assert name in m['files'] or name in ['licenses/DINOv3-LICENSE.md','licenses/IJEPA-LICENSE.txt'],name
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
    if name in m['files']:assert sha(dest)==m['files'][name]['sha256'],name
  joined.unlink();print(group,'verified and extracted',flush=True)
if __name__=='__main__':main()
