from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urljoin,urlsplit,unquote
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

work=Path('/root/fast-pgsr-setup-only')
rank={tag:i for i,tag in enumerate(sys_tags())}
specs=[line.split('==') for line in Path('/root/Fast-PGSR/requirements-Fast-Sam2.txt').read_text().splitlines() if line.startswith(('nvidia-','triton==','open3d=='))]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href')
            if href:self.links.append(href)
def get(spec):
    name,version=spec
    index='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/'+name+'/'
    for attempt in range(4):
        try:
            with urllib.request.urlopen(index,timeout=45) as response:html=response.read().decode()
            parser=Links();parser.feed(html)
            choices=[]
            for link in parser.links:
                filename=unquote(urlsplit(link).path).rsplit('/',1)[-1]
                try:parsed_name,ver,build,tags=parse_wheel_filename(filename)
                except Exception:continue
                if str(ver)!=version:continue
                compatible=[rank[t] for t in tags if t in rank]
                if compatible:choices.append((min(compatible),filename,urljoin(index,link)))
            _,filename,url=min(choices)
            digest=urlsplit(url).fragment.removeprefix('sha256=')
            assert len(digest)==64,(filename,digest)
            url=url.split('#')[0]
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as response:size=int(response.headers['Content-Length'])
            print(filename,size,flush=True)
            return dict(filename=filename,url=url,sha256=digest,bytes=size)
        except Exception:
            if attempt==3:raise
            time.sleep(2)
with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(get,specs))
(work/'large-wheels.json').write_text(json.dumps(records,indent=2))
print('PINNED LARGE WHEEL MANIFEST READY',len(records),sum(x['bytes'] for x in records),flush=True)
