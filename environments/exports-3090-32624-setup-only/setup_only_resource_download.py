import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import sys

base=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/root/fast-pgsr-setup-only/wheelhouse')
base.mkdir(exist_ok=True)
manifest=json.loads(Path(sys.argv[1] if len(sys.argv)>1 else '/root/fast-pgsr-setup-only/framework-wheels.json').read_text())
def download(item):
    dest=base/item['filename']
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        h=hashlib.sha256(dest.read_bytes()).hexdigest()
        if h==item['sha256']:return item
        raise RuntimeError('Existing wheel checksum mismatch: '+str(dest))
    print('DOWNLOAD',dest.name,flush=True)
    if 'bytes' in item:
        size=item['bytes']
    else:
        req=urllib.request.Request(item['url']+'?setup_probe=1',headers={'Range':'bytes=0-0'})
        with urllib.request.urlopen(req,timeout=60) as response:
            size=int(response.headers['Content-Range'].split('/')[-1])
    chunk=8*1024*1024
    parts=base/(dest.name+'.parts')
    parts.mkdir(exist_ok=True)
    def fetch(i):
        begin=i*chunk; end=min(size,begin+chunk)-1
        p=parts/str(i)
        if p.exists() and p.stat().st_size==end-begin+1:return p
        subprocess.run(['curl','-fLsS','--retry','4','--connect-timeout','20','--max-time','180','--range',f'{begin}-{end}',item['url']+'?setup_segment='+str(begin),'-o',str(p)+'.download'],check=True)
        temp=Path(str(p)+'.download')
        assert temp.stat().st_size==end-begin+1,(str(p),temp.stat().st_size,end-begin+1)
        temp.replace(p)
        print(dest.name,'chunk',i,flush=True)
        return p
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        paths=list(pool.map(fetch,range((size+chunk-1)//chunk)))
    h=hashlib.sha256()
    temp=Path(str(dest)+'.download')
    with temp.open('wb') as out:
        for path in paths:
            data=path.read_bytes();h.update(data);out.write(data)
    assert h.hexdigest()==item['sha256'],dest.name
    temp.replace(dest)
    print('OFFICIAL SHA256 MATCHED',dest.name,flush=True)
    return dict(item,bytes=size)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(download,manifest))
(base/'installed-downloads.json').write_text(json.dumps(records,indent=2))
