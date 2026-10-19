"""Fetch exact upstream revision and bundle unchanged simulation modules."""
import hashlib
import json
import subprocess
from pathlib import Path
from build_full_game import build

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT / 'upstream.lock.json').read_text())
src = ROOT / 'upstream'
if not src.exists():
    subprocess.run(['git', 'clone', lock['repository'], str(src)], check=True)
    subprocess.run(['git', '-C', str(src), 'checkout', '-b', lock['branch'], lock['revision']], check=True)
revision = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip()
if revision != lock['revision']:
    raise SystemExit('Upstream revision mismatch; use a clean checkout of upstream.lock.json revision.')
files = ['core/util', 'data/pokedex', 'data/types', 'data/moves', 'systems/stats',
         'entities/player', 'entities/enemy', 'entities/projectile', 'entities/pickup', 'systems/combat']
parts = []
for name in files:
    path = 'js/' + name + '.js'
    original = subprocess.check_output(['git', '-C', str(src), 'show', f'{revision}:{path}'])
    if (src / path).read_bytes() != original:
        raise SystemExit(f'Modified upstream module: {path}')
    parts.append(original.decode())
notice = '// Required Notice: Copyright 2026 JPTER78 (https://github.com/JPTER78/pokesurvivor)\n// https://polyformproject.org/licenses/noncommercial/1.0.0/\n'
body = notice + 'export function createGame(Math, performance) {\nconst window = {G:{}}; const G = window.G;\n' + '\n'.join(parts) + '\nreturn G;\n}\n'
(ROOT / 'rl/upstream-source.js').write_text(body)
build()
print(json.dumps({'revision': revision, 'bundle_sha256': hashlib.sha256(body.encode()).hexdigest()}))
