import hashlib,json,shutil,subprocess,socket
from pathlib import Path
root=Path('/root/Fast-PGSR');b=Path('/root/fast-pgsr-new-setup-20261017');meta=json.loads((b/'target.json').read_text());exports=root/'environments'/('exports-'+meta['name']+'-setup-only');exports.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def command(args):return subprocess.check_output(args,text=True,stderr=subprocess.PIPE)
r=dict(meta,scope='本次仅完成项目及环境安装，未执行测试、推理或训练。',hostname=socket.gethostname(),project=str(root),conda_base=(b/'conda-base.txt').read_text().strip() if (b/'conda-base.txt').exists() else None,envs_dir='/root/.conda/envs',cuda_home='/usr/local/cuda-11.8',gpu_architecture=(b/'gpu-architecture.txt').read_text().strip(),phases={},environments={},models=[],source_files=[])
for phase in ['system','wheels','large-wheels','environments','colmap','models-integrity','cache-copy','apply-cache']:
 p=b/(phase+'.exit');r['phases'][phase]=p.read_text().strip() if p.exists() else 'not finished'
for name in ['fast-pgsr-old','Fast-Sam2']:
 prefix=Path('/root/.conda/envs')/name
 if not (prefix/'bin/python').exists():r['environments'][name]={'status':'not created'};continue
 py=str(prefix/'bin/python');listing=command([py,'-m','pip','list','--format=json']);packages={x['name'].lower().replace('_','-'):x['version'] for x in json.loads(listing)}
 (exports/(name+'.pip-list.json')).write_text(listing);(exports/(name+'.freeze.txt')).write_text(command([py,'-m','pip','freeze','--all']))
 if r['conda_base']:(exports/(name+'.conda.json')).write_text(command([r['conda_base']+'/bin/conda','list','--prefix',str(prefix),'--json']))
 pins={}
 for line in (root/('requirements-'+name+'.txt')).read_text().splitlines():
  if '==' in line and not line.startswith(('#','--')):
   pkg,version=line.strip().split('==',1);pins[pkg.lower().replace('_','-')]=version
 mismatches=[{'name':pkg,'required':v,'installed':packages.get(pkg)} for pkg,v in pins.items() if packages.get(pkg)!=v]
 e={'pinned_package_count':len(pins),'requirements_pin_mismatches':mismatches,'prefix':str(prefix),'python':command([py,'--version']).strip(),'key_packages':{k:packages.get(k) for k in ['torch','torchvision','torchaudio','numpy','pycolmap','pytorch3d','diff-gaussian-rasterization-fastgs','diff-plane-rasterization','simple-knn','fused-ssim','groundingdino','sam-2']},'compiled_extensions':[]}
 candidates=[]
 for p in (prefix/'lib').glob('python*/site-packages/**/*.so'):
  if any(k in str(p) for k in ['pytorch3d','diff_gaussian','diff_plane','simple_knn','fused_ssim']):candidates.append(p)
 if name=='Fast-Sam2':
  for sub in ['sam2','grounding_dino/groundingdino']:candidates.extend((root/'third_party/Grounded-SAM-2'/sub).glob('_C*.so'))
 for p in candidates:
  a={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
  try:a['cuda_elf_listing']=command(['/usr/local/cuda-11.8/bin/cuobjdump','--list-elf',str(p)])
  except subprocess.CalledProcessError as ex:a['cuda_elf_listing_error']=str(ex)
  e['compiled_extensions'].append(a)
 r['environments'][name]=e
for i in json.loads((root/'environments/exports-3090-32596/model-resources-sha256.json').read_text()):
 p=root/'third_party/Grounded-SAM-2'/i['path'];e=dict(i,exists=p.exists())
 if p.exists():e.update(actual_bytes=p.stat().st_size,actual_sha256=sha(p));e['matched']=e['actual_bytes']==i['bytes'] and e['actual_sha256']==i['sha256']
 r['models'].append(e)
manifest={i['path']:i for i in json.loads((b/'source-manifest-final.json').read_text())}
for rel in ['scripts/mask_generation/instance_mask_guard.py','scripts/mask_generation/run_grounded_sam2_scene.py','scripts/mask_generation/run_rubik_scene_auto.py','scripts/validate_semantic_inputs.py','run_batch_gpu1.sh','render.py','scripts/export_target_gaussians.py']:
 p=root/rel;r['source_files'].append(dict(path=rel,sha256=sha(p),matched=sha(p)==manifest[rel]['sha256']))
changes=[];missing=[]
for rel,i in manifest.items():
 p=root/rel
 if not p.exists():missing.append(rel)
 elif sha(p)!=i['sha256']:changes.append({'path':rel,'before_sha256':i['sha256'],'actual_sha256':sha(p)})
r['source_post_install_audit']={'missing_original_files':missing,'installer_changed_files':changes,'original_file_count':len(manifest),'cpp_cuda_headers_count':sum(x.endswith(('.cpp','.cu','.h','.cuh')) for x in manifest)}
p=Path('/usr/local/bin/colmap');r['colmap']={'path':str(p),'exists':p.exists()}
if p.exists():r['colmap'].update(bytes=p.stat().st_size,sha256=sha(p),cuda_elf_listing=command(['/usr/local/cuda-11.8/bin/cuobjdump','--list-elf',str(p)]))
p=b/'src/colmap-3.9.1/build/CMakeCache.txt'
if p.exists():
 shutil.copy2(p,exports/'colmap-CMakeCache.txt');r['colmap']['cmake_configuration']=[x for x in p.read_text().splitlines() if x.startswith(('CUDA_ENABLED:','CMAKE_CUDA_ARCHITECTURES:','CMAKE_CUDA_COMPILER:','CMAKE_INSTALL_PREFIX:','TESTS_ENABLED:'))]
for fn in ['source-manifest-final.json','source-integrity.txt','framework-wheels.json','large-wheels.json','pytorch3d-source.tar.sha256.json','torch-build-architecture-adjustment.json','cpp-extension-restored.sha256','wheelhouse/installed-downloads.json','wheelhouse/large-installed-downloads.json','cache-transfer-installed.json','source-cache-transfer-manifest.json','models-integrity.json','conda-yaml-lock-provenance.json']:
 p=b/fn
 if p.exists():shutil.copy2(p,exports/p.name)
helper=Path('/root/.conda/envs/fast-pgsr-old/lib/python3.7/site-packages/torch/utils/cpp_extension.py')
original=b/'cpp_extension.py.original'
r['torch_build_helper_restored']={'adjustment_needed':original.exists(),'current_sha256':sha(helper),'original_sha256':sha(original) if original.exists() else None,'matches_original':sha(helper)==sha(original) if original.exists() else None}
p=root/'activate_setup_only.sh'
r['manual_environment_configuration']={'path':str(p),'sha256':sha(p)}
shutil.copy2(p,exports/p.name)
(exports/'system-packages.tsv').write_text(command(['dpkg-query','-W','-f=${binary:Package}\t${Version}\n']))
(exports/'installation-result.json').write_text(json.dumps(r,indent=2,ensure_ascii=False));(b/'installation-result.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
print(json.dumps(r,ensure_ascii=False,indent=2))
