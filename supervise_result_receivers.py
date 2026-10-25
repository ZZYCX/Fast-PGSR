"""Keep local receivers running; never stop receivers or remote jobs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from receive_results import atomic_json, process_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.interval_seconds < 1:
        parser.error('--interval-seconds must be >= 1')
    root = Path(__file__).resolve().parent
    meta = root / '.receiver-supervisor'
    meta.mkdir(exist_ok=True)
    servers = json.loads((root / 'result_receivers.json').read_text(encoding='utf-8'))['servers']
    for server in servers:
        key = server.get('password_env', 'FAST_PGSR_SSH_PASSWORD')
        if not os.environ.get(key):
            parser.error('Missing credential environment variable: ' + key)
    try:
        with process_lock(meta / 'supervisor.lock'):
            while True:
                missing = []
                for server in servers:
                    receiver_meta = Path(server['local_output_dir']) / '.receiver'
                    try:
                        with process_lock(receiver_meta / 'receiver.lock'):
                            missing.append(server['name'])
                    except OSError:
                        pass
                code = None
                if missing:
                    print(datetime.datetime.now().isoformat(), 'restart missing receivers:', ', '.join(missing), flush=True)
                    code = subprocess.run([sys.executable, str(root / 'start_result_receivers.py')],
                                          stdin=subprocess.DEVNULL).returncode
                atomic_json(meta / 'status.json', dict(pid=os.getpid(),
                            checked_at=datetime.datetime.now().isoformat(timespec='seconds'),
                            interval_seconds=args.interval_seconds, missing_before_check=missing,
                            launcher_exit_code=code))
                if args.once:
                    return int(code or 0)
                time.sleep(args.interval_seconds)
    except OSError as error:
        print('Supervisor unavailable or already running:', error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
