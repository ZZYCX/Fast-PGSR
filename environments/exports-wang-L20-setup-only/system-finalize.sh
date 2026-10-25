set -e
B=/root/fast-pgsr-new-setup-20261017
for pat in "^bash $B/environments.sh$" "^bash $B/colmap-launcher.sh$"; do
 for p in $(pgrep -f "$pat" || true); do kill -TERM "$p"; done
done
for p in $(pgrep -f '^curl -fL --retry 3 https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh' || true); do kill -TERM "$p"; done
while pgrep -f "^bash $B/system.sh$" >/dev/null; do sleep 1; done
python3 - <<'PY'
import json,shlex,subprocess
from pathlib import Path
b=Path('/root/fast-pgsr-new-setup-20261017')
args=shlex.split(next(x for x in (b/'system.sh').read_text().splitlines() if x.startswith('apt-get install')))[3:]
rows=[]
for name in args:
 status=subprocess.check_output(['dpkg-query','-W','-f=${Status}',name],text=True)
 assert status=='install ok installed',(name,status)
 rows.append({'package':name,'status':status})
assert (b/'conda-bootstrap.exit').read_text().strip()=='0'
assert Path('/root/miniconda3/conda-meta/history').exists()
(b/'system-final-installation-metadata.json').write_text(json.dumps({'apt_packages':rows,'miniconda_installation':'cached installer bootstrap exit 0','note':'Stopped redundant installer download after cached bootstrap had installed base; no apt or dpkg stage interrupted'},indent=2))
(b/'conda-base.txt').write_text('/root/miniconda3\n')
p=b/'system.exit'
if p.exists():p.rename(b/'system-redundant-download.exit')
p.write_text('0\n')
print('APT package installation metadata complete; cached Miniconda bootstrap completed; system installation recorded as complete')
PY
if [ -f "$B/environments.exit" ]; then mv "$B/environments.exit" "$B/environments-before-system-finalize.exit"; fi
nohup bash -c "bash $B/environments.sh >> $B/environments.log 2>&1; echo \$? > $B/environments.exit" > /dev/null 2>&1 < /dev/null &
nohup bash "$B/colmap-launcher.sh" > /dev/null 2>&1 < /dev/null &
echo resumed-environments-and-colmap
