import hashlib
import json
from pathlib import Path
import shutil

root=Path('/root/Fast-PGSR')
work=Path('/root/fast-pgsr-setup-only')
exports=root/'environments/exports-3090-32624-setup-only'
scope=json.loads((work/'final-scope-input.json').read_text())
manifest=json.loads((work/'source-manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
staged=work/'current-local.gitignore'
assert sha(staged)==scope['gitignore_sha256']
old=root/'.gitignore'
original=next(x for x in manifest if x['path']=='.gitignore')
assert sha(old)==original['sha256']
backup=work/'before-final-gitignore'
backup.mkdir(exist_ok=False)
shutil.copy2(old,backup/'.gitignore')
for entry in scope['excluded_transfer_files']:
    p=root/entry['path']
    assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink()
    assert p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],str(p)
for entry in scope['excluded_transfer_files']:(root/entry['path']).unlink()
shutil.copy2(staged,old)
excluded={x['path'] for x in scope['excluded_transfer_files']}
final=[]
installer_changes=[]
for entry in manifest:
    if entry['path'] in excluded:continue
    p=root/entry['path']
    actual=dict(path=entry['path'],bytes=p.stat().st_size,sha256=sha(p))
    if actual['sha256']!=entry['sha256'] and entry['path']!='.gitignore':installer_changes.append(entry['path'])
    final.append(actual)
result={'removed_owned_transfer_files':scope['excluded_transfer_files'],'updated_gitignore_sha256':scope['gitignore_sha256'],'final_source_file_count':len(final),'preserved_cpp_cuda_header_inputs':sum(x['path'].endswith(('.cpp','.cu','.h','.cuh')) for x in final),'installer_generated_source_changes':installer_changes}
(exports/'source-manifest-final.json').write_text(json.dumps(final,indent=2))
(exports/'final-migration-scope.json').write_text(json.dumps(result,indent=2))
archive=work/'project.tar.gz'
assert sha(archive)==scope['original_archive_sha256']
archive.unlink()
print(json.dumps({k:v for k,v in result.items() if k!='removed_owned_transfer_files'},indent=2))
print('Removed owned obsolete transfer archive; original audit manifest retained.')
