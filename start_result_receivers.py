#!/usr/bin/env python3
"""Start the configured read-only receivers as independent background processes."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

from receive_results import process_lock


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'result_receivers.json')
    parser.add_argument('--password-env', default='FAST_PGSR_SSH_PASSWORD')
    parser.add_argument('--known-hosts', type=Path)
    parser.add_argument('--poll-seconds', type=int, default=1800)
    parser.add_argument('--status', action='store_true', help='Read status without starting processes')
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error('--poll-seconds must be >= 1')
    servers = json.loads(args.config.read_text(encoding='utf-8'))['servers']
    if args.status:
        for server in servers:
            meta = Path(server['local_output_dir']).expanduser().resolve() / '.receiver'
            try:
                status = json.loads((meta / 'status.json').read_text(encoding='utf-8'))
                with process_lock(meta / 'receiver.lock'):
                    running = False
            except OSError:
                running = (meta / 'status.json').exists()
                status = json.loads((meta / 'status.json').read_text(encoding='utf-8')) if running else {}
            print(json.dumps({'name': server['name'], 'running': running,
                              'verified_receipts': len(list((meta / 'receipts').glob('*.json'))),
                              'status': status}, ensure_ascii=True))
        return
    password = os.environ.get(args.password_env)
    if not password:
        if not sys.stdin.isatty():
            parser.error('Set ' + args.password_env + ' before background launch')
        password = getpass.getpass('Default SSH password: ')
    environment = os.environ.copy()
    environment[args.password_env] = password
    environment['PYTHONIOENCODING'] = 'utf-8'
    for server in servers:
        meta = Path(server['local_output_dir']).expanduser().resolve() / '.receiver'
        meta.mkdir(parents=True, exist_ok=True)
        try:
            with process_lock(meta / 'receiver.lock'):
                pass
        except OSError:
            print(server['name'] + ': already running')
            continue
        password_env = server.get('password_env', args.password_env)
        if not environment.get(password_env):
            if not sys.stdin.isatty():
                parser.error('Set ' + password_env + ' for ' + server['name'])
            environment[password_env] = getpass.getpass('SSH password for ' + server['name'] + ': ')
        command = [sys.executable, str(root / 'receive_results.py'), '--host', server['host'],
                   '--port', str(server['port']), '--local-output-dir', server['local_output_dir'],
                   '--password-env', password_env, '--poll-seconds', str(args.poll_seconds)]
        if args.known_hosts:
            command += ['--known-hosts', str(args.known_hosts.resolve())]
        options = {'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} \
            if os.name == 'nt' else {'start_new_session': True}
        with (meta / 'background.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       env=environment, close_fds=True, **options)
        print(server['name'] + ': started PID=' + str(process.pid)
              + ' -> ' + server['local_output_dir'])


if __name__ == '__main__':
    main()
