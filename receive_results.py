#!/usr/bin/env python3
"""Read-only SFTP receiver for completed Fast-PGSR scene outputs (Python 3.8+)."""
import argparse
import contextlib
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys
import time

import paramiko

MARKER = '.reconstruction_complete.json'
BLOCK = 1024 * 1024
LOG = logging.getLogger('result_receiver')


def digest(path, limit=None):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining = limit
        while remaining is None or remaining > 0:
            data = stream.read(BLOCK if remaining is None else min(BLOCK, remaining))
            if not data:
                break
            hasher.update(data)
            if remaining is not None:
                remaining -= len(data)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def safe_relative(name):
    # Preserve filenames exactly: refuse Windows traversal, ADS and aliases.
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Unsafe output path: ' + repr(name))
    path = PurePosixPath(name)
    parts = name.split('/')
    if path.is_absolute() or any(part in ('', '.', '..') for part in parts):
        raise ValueError('Unsafe output path: ' + repr(name))
    for part in parts:
        if (re.search(r'[<>:"|?*\x00-\x1f]', part) or part.endswith((' ', '.'))
                or re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', part.split('.')[0])):
            raise ValueError('Filename cannot be preserved on Windows: ' + repr(name))
    return path


def local_path(base, relative):
    parts = safe_relative(relative).parts
    candidate = base.joinpath(*parts)
    for path in [base] + [base.joinpath(*parts[:i]) for i in range(1, len(parts) + 1)]:
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('Refusing local link: ' + str(path))
    if base.resolve() not in candidate.resolve().parents:
        raise ValueError('Output path escapes destination')
    return candidate


def validate_manifest(raw, scene):
    manifest = json.loads(raw)
    if manifest.get('version') != 1 or manifest.get('scene') != scene:
        raise ValueError('Unsupported or mismatched scene manifest')
    if not re.fullmatch('[a-f0-9]{32}', manifest.get('generation', '')):
        raise ValueError('Invalid generation')
    files, directories = manifest.get('files'), manifest.get('directories')
    if not isinstance(files, list) or not files or not isinstance(directories, list):
        raise ValueError('Empty/invalid manifest')
    seen = set()
    for name in directories + [item['path'] for item in files]:
        safe_relative(name)
        key = name.casefold()
        if key in seen or name in (MARKER, MARKER + '.tmp'):
            raise ValueError('Duplicate/reserved output path: ' + name)
        seen.add(key)
    file_names = {item['path'].casefold() for item in files}
    for name in seen:
        if any(str(parent).casefold() in file_names for parent in PurePosixPath(name).parents
               if str(parent) != '.'):
            raise ValueError('File/directory path collision')
    for item in files:
        if (type(item.get('size')) is not int or item['size'] < 0
                or not re.fullmatch('[a-f0-9]{64}', item.get('sha256', ''))):
            raise ValueError('Invalid file hash/size')
    return manifest


class Receiver:
    def __init__(self, host, port, local_output_dir, password, username='root',
                 remote_output_dir='/root/Fast-PGSR/output', known_hosts=None):
        self.host, self.port, self.username = host, port, username
        self.root = Path(local_output_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = self.root / '.receiver'
        self.meta.mkdir(exist_ok=True)
        if self.meta.is_symlink():
            raise ValueError('Receiver metadata must not be a symlink')
        self.remote_root = remote_output_dir.rstrip('/')
        self.password, self.known_hosts = password, known_hosts
        self.client = self.sftp = None
        self.status = {'host': host, 'port': port, 'local_output_dir': str(self.root),
                       'remote_output_dir': self.remote_root, 'pid': os.getpid(),
                       'state': 'starting'}

    def update(self, **values):
        self.status.update(values)
        self.status['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        atomic_json(self.meta / 'status.json', self.status)

    def connect(self):
        self.close()
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        if self.known_hosts:
            self.client.load_host_keys(str(self.known_hosts))
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.client.connect(self.host, self.port, username=self.username,
                            password=self.password, timeout=20, banner_timeout=20,
                            auth_timeout=20, allow_agent=False, look_for_keys=False)
        self.client.get_transport().set_keepalive(15)
        self.sftp = self.client.open_sftp()
        self.sftp.get_channel().settimeout(60)

    def close(self):
        if self.client:
            self.client.close()
        self.client = self.sftp = None

    def marker(self, remote):
        path = remote + '/' + MARKER
        info = self.sftp.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024:
            raise ValueError('Invalid completion marker: ' + path)
        with self.sftp.open(path, 'rb') as stream:
            return stream.read()

    def prefix_hash(self, remote, count):
        pipeline = 'head -c %d -- %s | sha256sum' % (count, shlex.quote(remote))
        _, output, error = self.client.exec_command(
            'bash -o pipefail -c ' + shlex.quote(pipeline), timeout=120)
        result = output.read().decode().strip()
        message = error.read().decode().strip()
        if output.channel.recv_exit_status() or not re.fullmatch('[a-f0-9]{64}  -', result):
            raise IOError('Cannot validate resume prefix: ' + message)
        return result.split()[0]

    def fetch_file(self, remote, destination, partial, record):
        info = self.sftp.lstat(remote)
        if not stat.S_ISREG(info.st_mode) or info.st_size != record['size']:
            raise IOError('Remote file changed or is not regular: ' + remote)
        if (destination.is_file() and destination.stat().st_size == record['size']
                and digest(destination) == record['sha256']):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial.parent.mkdir(parents=True, exist_ok=True)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > record['size'] or (offset and digest(partial) != self.prefix_hash(remote, offset)):
            LOG.warning('RESTART invalid partial %s', remote)
            with partial.open('wb'):
                pass
            offset = 0
        if offset:
            LOG.info('RESUME %s at byte %d', remote, offset)
        self.update(state='downloading', file=record['path'], file_bytes=offset,
                    file_size=record['size'], error=None)
        last = time.monotonic()
        with self.sftp.open(remote, 'rb') as source, partial.open('ab' if offset else 'wb') as target:
            source.seek(offset)
            while offset < record['size']:
                data = source.read(min(BLOCK, record['size'] - offset))
                if not data:
                    raise IOError('Unexpected end of remote file: ' + remote)
                target.write(data)
                offset += len(data)
                if time.monotonic() - last >= 5:
                    target.flush()
                    self.update(file_bytes=offset)
                    last = time.monotonic()
            target.flush()
            os.fsync(target.fileno())
        if digest(partial) != record['sha256']:
            # Keep the data; next retry validates its prefix and restarts if corrupt.
            raise IOError('SHA-256 mismatch: ' + remote)
        os.replace(partial, destination)
        LOG.info('VERIFIED %s (%d bytes)', remote, record['size'])

    def sync_scene(self, scene):
        safe_relative(scene)
        if '/' in scene or scene.casefold() == '.receiver':
            raise ValueError('Invalid scene directory')
        remote = self.remote_root + '/' + scene
        raw = self.marker(remote)
        manifest = validate_manifest(raw, scene)
        manifest_hash = hashlib.sha256(raw).hexdigest()
        receipt = local_path(self.meta / 'receipts', scene + '.json')
        scene_root = local_path(self.root, scene)
        partial_root = local_path(self.meta / 'partials', scene)
        try:
            previous = json.loads(receipt.read_text(encoding='utf-8'))
        except (FileNotFoundError, ValueError):
            previous = {}
        # Unchanged receipts need only stat checks; changed local files are rehashed/repaired.
        if previous.get('manifest_sha256') == manifest_hash:
            unchanged = True
            for item in previous.get('local_files', []):
                path = local_path(scene_root, item['path'])
                if (not path.is_file() or path.stat().st_size != item['size']
                        or path.stat().st_mtime_ns != item['mtime_ns']):
                    unchanged = False
                    break
            if unchanged and len(previous.get('local_files', [])) == len(manifest['files']):
                if all(local_path(scene_root, directory).is_dir()
                       for directory in manifest['directories']):
                    return False
        self.update(state='syncing_scene', scene=scene, error=None)
        scene_root.mkdir(parents=True, exist_ok=True)
        # A previous local receipt must not describe an in-progress replacement.
        (scene_root / MARKER).unlink(missing_ok=True)
        for directory in manifest['directories']:
            local_path(scene_root, directory).mkdir(parents=True, exist_ok=True)
        for item in manifest['files']:
            relative = item['path']
            # Reject links in every remote parent before reading file contents.
            for parent in reversed(PurePosixPath(relative).parents):
                parent_path = remote if str(parent) == '.' else remote + '/' + str(parent)
                if not stat.S_ISDIR(self.sftp.lstat(parent_path).st_mode):
                    raise ValueError('Remote parent is not a real directory: ' + parent_path)
            self.fetch_file(remote + '/' + relative, local_path(scene_root, relative),
                            local_path(partial_root, relative + '.part'), item)
        if self.marker(remote) != raw:
            raise IOError('Scene generation changed while downloading: ' + scene)
        local_files = []
        for item in manifest['files']:
            path = local_path(scene_root, item['path'])
            if path.stat().st_size != item['size'] or digest(path) != item['sha256']:
                raise IOError('Final scene verification failed: ' + str(path))
            local_files.append({'path': item['path'], 'size': item['size'],
                                'mtime_ns': path.stat().st_mtime_ns})
        # Save the server's exact manifest alongside the downloaded data.
        atomic_json(scene_root / MARKER, manifest)
        atomic_json(receipt, {'manifest_sha256': manifest_hash, 'generation': manifest['generation'],
                             'verified_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                             'local_files': local_files})
        LOG.info('SCENE_COMPLETE %s: %d files', scene, len(manifest['files']))
        return True

    def scan_once(self):
        if self.sftp is None:
            self.connect()
        completed, ready, errors = 0, [], []
        try:
            entries = self.sftp.listdir_attr(self.remote_root)
        except FileNotFoundError:
            # A server that has not reconstructed its first plant has no output root yet.
            if self.status.get('remote_output_exists') is not False:
                LOG.info('WAITING_FOR_OUTPUT_DIRECTORY %s', self.remote_root)
            self.update(state='waiting', remote_output_exists=False, scene=None, file=None,
                        ready_scenes=0, completed_this_scan=0, errors=[], error=None)
            return []
        for entry in entries:
            if not stat.S_ISDIR(entry.st_mode):
                continue
            scene = entry.filename
            try:
                self.sftp.lstat(self.remote_root + '/' + scene + '/' + MARKER)
            except FileNotFoundError:
                continue
            ready.append(scene)
            try:
                completed += bool(self.sync_scene(scene))
            except Exception as error:
                LOG.exception('SCENE_RETRY %s', scene)
                errors.append({'scene': scene, 'error': str(error)})
                # Transport faults are retried by the outer loop after reconnecting.
                if not self.client.get_transport() or not self.client.get_transport().is_active():
                    raise
        self.update(state='waiting' if not errors else 'retrying', scene=None, file=None,
                    remote_output_exists=True,
                    ready_scenes=len(ready), completed_this_scan=completed,
                    errors=errors, error=None if not errors else str(errors))
        return errors


@contextlib.contextmanager
def process_lock(path):
    with Path(path).open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--username', default='root')
    parser.add_argument('--local-output-dir', required=True)
    parser.add_argument('--remote-output-dir', default='/root/Fast-PGSR/output')
    parser.add_argument('--password-env', default='FAST_PGSR_SSH_PASSWORD')
    parser.add_argument('--known-hosts')
    parser.add_argument('--poll-seconds', type=float, default=1800)
    parser.add_argument('--once', action='store_true', help='One scan; nonzero on errors')
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error('--poll-seconds must be >= 1')
    password = os.environ.get(args.password_env)
    if not password:
        if not sys.stdin.isatty():
            parser.error('Set ' + args.password_env + ' before background launch')
        password = getpass.getpass('SSH password: ')
    receiver = Receiver(args.host, args.port, args.local_output_dir, password,
                        args.username, args.remote_output_dir, args.known_hosts)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(receiver.meta / 'receiver.log', encoding='utf-8'),
                                  logging.StreamHandler()])
    try:
        with process_lock(receiver.meta / 'receiver.lock'):
            LOG.info('START %s:%d -> %s', args.host, args.port, receiver.root)
            atomic_json(receiver.meta / 'process.json', {'pid': os.getpid(), 'host': args.host,
                        'port': args.port, 'local_output_dir': str(receiver.root)})
            while True:
                try:
                    errors = receiver.scan_once()
                    if args.once:
                        return 1 if errors else 0
                except Exception as error:
                    LOG.exception('CONNECTION_RETRY')
                    receiver.update(state='retrying', error=str(error))
                    if args.once:
                        return 1
                finally:
                    # Fresh authenticated connection per scan also clears broken SFTP channels.
                    receiver.close()
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        receiver.update(state='stopped')
    finally:
        receiver.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
