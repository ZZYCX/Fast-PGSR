import hashlib
import json
from pathlib import Path

root=Path('/root/Fast-PGSR')
items=json.loads((root/'environments/exports-3090-32596/model-resources-sha256.json').read_text())
records=[]
for item in items:
    p=root/'third_party/Grounded-SAM-2'/item['path']
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert p.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256'],str(p)
    records.append(dict(item,status='target-size-and-sha256-matched'))
    print('FILE INTEGRITY MATCHED',item['path'],item['bytes'],flush=True)
Path('/root/fast-pgsr-setup-only/model-resources-installed.json').write_text(json.dumps(records,indent=2))
Path('/root/fast-pgsr-setup-only/models-integrity.exit').write_text('0\n')
print('Seven resources matched; no model loading performed.')
