"""Build an offline entrypoint around the pinned, unmodified full game."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def build():
    source = ROOT / 'upstream'
    target = ROOT / 'full_game'
    target.mkdir(exist_ok=True)
    html = (source / 'dev.html').read_text()
    html = re.sub(r'<!-- seo:begin -->.*?<!-- seo:end -->', '', html, flags=re.S)
    policy = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; media-src 'self' blob:; connect-src 'none'; worker-src 'none'; object-src 'none'; base-uri 'self'; form-action 'self'"
    html, count = re.subn(r'<meta http-equiv="Content-Security-Policy"[^>]*>',
                          f'<meta http-equiv="Content-Security-Policy" content="{policy}">', html)
    if count != 1:
        raise RuntimeError('Upstream CSP changed; review offline entrypoint')
    html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n<base href="/upstream/">')
    html = re.sub(r'<link rel="(?:preconnect|manifest)"[^>]*>', '', html)
    # Runs before boot.js and every deferred module, isolating this page's storage.
    html = html.replace('</head>', '<script src="/full_game/offline.js"></script>\n</head>')
    for original, replacement in [('js/core/firebase-config.js', '/full_game/config.js'),
                                  ('js/game.js', '/full_game/game.js'),
                                  ('js/main.js', '/full_game/main.js')]:
        needle = f'src="{original}"'
        if html.count(needle) != 1:
            raise RuntimeError(f'Unexpected upstream script reference: {original}')
        html = html.replace(needle, f'src="{replacement}"')
    html = html.replace('<title>PokéSurvivor – Juego Pokémon gratis online estilo Survivor.io</title>',
                        '<title>PokéSurvivor — full local game</title>')
    (target / 'index.html').write_text(html)
    main = (source / 'js/main.js').read_text()
    guard = "if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {"
    if main.count(guard) != 1:
        raise RuntimeError('Upstream service worker startup changed; review offline entrypoint')
    main = main.replace(guard, 'if (false) { // Local full game: service worker disabled.')
    notice = '// Required Notice: Copyright 2026 JPTER78 (https://github.com/JPTER78/pokesurvivor)\n// https://polyformproject.org/licenses/noncommercial/1.0.0/\n'
    (target / 'main.js').write_text(notice + main)
    game = (source / 'js/game.js').read_text()
    anchor = '    init, startRun, everyone, levelXp, flash, offerBond,'
    if game.count(anchor) != 1:
        raise RuntimeError('Upstream game API changed; review training adapter')
    hook = '''
    // Accessible only in the explicitly requested training page.
    rlTick() {
      if (!window.POKERL_TRAINING) throw new Error('Training mode required');
      if (state === 'playing') update(1 / 60);
      G.Input.endFrame();
    },
'''
    game = game.replace(anchor, anchor + hook)
    (target / 'game.js').write_text(notice + game)

if __name__ == '__main__':
    build()
