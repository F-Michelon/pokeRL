"""Read a prepared checkout's revision without a Git executable."""
from pathlib import Path
import re


def read_revision(source):
    git_dir = Path(source) / '.git'
    if git_dir.is_file():
        pointer = git_dir.read_text().strip()
        if not pointer.startswith('gitdir: '):
            raise RuntimeError('Invalid upstream .git file')
        git_dir = (git_dir.parent / pointer[8:]).resolve()
    common_dir = git_dir
    if (git_dir / 'commondir').is_file():
        common_dir = (git_dir / (git_dir / 'commondir').read_text().strip()).resolve()
    revision = (git_dir / 'HEAD').read_text().strip()
    if revision.startswith('ref: '):
        ref = revision[5:]
        for directory in (git_dir, common_dir):
            path = directory / ref
            if path.is_file():
                revision = path.read_text().strip()
                break
        else:
            packed = common_dir / 'packed-refs'
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    fields = line.split()
                    if len(fields) == 2 and fields[1] == ref:
                        revision = fields[0]
                        break
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', revision):
        raise RuntimeError('Cannot resolve upstream revision; prepare the checkout before submitting')
    return revision
