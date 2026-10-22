"""Explicit pinned source/weight fetching. Importing performs no I/O/network."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from data.action4_paths import checked_root
import subprocess
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen


def _path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Output/source path escapes project root')
    return path


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _public_https(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('A credential-free public HTTPS URL is required')
    return url


def _git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


def fetch_upstreams(root, lock_path='configs/data/upstream.lock.json', *, audit_only=False):
    root = checked_root(root)
    lock = json.loads(_path(root, lock_path).read_text(encoding='utf-8'))
    results = []
    for entry in lock['repositories']:
        target = _path(root, entry['path'])
        url = _public_https(entry['url'])
        commit = entry['commit']
        if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('Source commit must be a full hexadecimal SHA')
        if not target.exists():
            if audit_only:
                raise FileNotFoundError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            # A new temporary clone is retained on failure for diagnosis; no user checkout is overwritten.
            temporary = Path(tempfile.mkdtemp(prefix=entry['name'] + '-download-', dir=target.parent))
            subprocess.run(['git', 'clone', '--no-checkout', url, str(temporary)], check=True)
            _git(temporary, 'checkout', '--detach', commit)
            if target.exists():
                raise FileExistsError('Concurrent checkout appeared; keep downloaded temporary directory')
            temporary.rename(target)
        origin = _git(target, 'remote', 'get-url', 'origin')
        if origin.rstrip('/').removesuffix('.git') != url.rstrip('/').removesuffix('.git'):
            raise ValueError(f'Origin differs for {entry["name"]}')
        tracked_dirty = _git(target, 'status', '--porcelain', '--untracked-files=no')
        if tracked_dirty:
            raise ValueError(f'Tracked changes in {entry["name"]}; preserved without reset or checkout')
        head = _git(target, 'rev-parse', 'HEAD')
        if head != commit:
            if audit_only:
                raise ValueError(f'HEAD differs for {entry["name"]}')
            _git(target, 'fetch', 'origin', commit)
            _git(target, 'checkout', '--detach', commit)
            head = _git(target, 'rev-parse', 'HEAD')
        untracked = _git(target, 'ls-files', '--others', '--exclude-standard').splitlines()
        results.append({'name': entry['name'], 'path': entry['path'], 'origin': origin,
                        'head': head, 'verified': head == commit, 'tracked_clean': True,
                        'untracked_file_count': len(untracked), 'untracked_preserved': True})
    return {'status': 'pinned_sources_verified', 'repositories': results}


def fetch_weights(root, lock_path='configs/model/pretrained.lock.json', *, audit_only=False):
    root = checked_root(root)
    lock = json.loads(_path(root, lock_path).read_text(encoding='utf-8'))
    target = _path(root, lock['path'])
    url = _public_https(lock['url'])
    if not target.exists():
        if audit_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, filename = tempfile.mkstemp(prefix=target.name + '.', suffix='.partial', dir=target.parent)
        temporary = Path(filename)
        with os.fdopen(fd, 'wb') as stream, urlopen(url, timeout=60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(chunk)
        if temporary.stat().st_size != lock['bytes'] or _sha(temporary) != lock['sha256']:
            raise ValueError(f'Weight identity mismatch; failed download preserved at {temporary}')
        # Hard-link creation fails atomically if another process populated the output.
        os.link(temporary, target)
        temporary.unlink()
    if target.stat().st_size != lock['bytes'] or _sha(target) != lock['sha256']:
        raise ValueError('Existing weight differs from pinned identity; file preserved')
    result = {**lock, 'verified': True, 'verified_utc': datetime.now(timezone.utc).isoformat()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('kind', choices=['upstream', 'weights'])
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--output', help='Optional NEW JSON report path inside the project')
    args = parser.parse_args(argv)
    root = Path(args.root).resolve(strict=True)
    if args.output and _path(root, args.output).exists():
        raise FileExistsError('Report already exists')
    result = (fetch_upstreams if args.kind == 'upstream' else fetch_weights)(root, audit_only=args.audit_only)
    if args.output:
        path = _path(root, args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
