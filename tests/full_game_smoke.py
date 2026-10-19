"""Exercise full-game onboarding, real world systems, and local-only storage."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=os.environ.get('POKERL_CHROME'), headless=True)
        context = browser.new_context(viewport={'width':1280, 'height':720})
        context.add_init_script("window.testNativeStorage = window.localStorage; window.localStorage.setItem('ps.test-sentinel', 'unchanged');")
        page = context.new_page()
        errors, external, missing = [], [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: external.append(request.url) if urlsplit(request.url).hostname not in ('127.0.0.1','localhost') and request.url.startswith(('http:', 'https:')) else None)
        page.on('response', lambda response: missing.append(response.url) if response.status >= 400 else None)
        try:
            page.goto('http://127.0.0.1:8000/')
            page.wait_for_function('window.G && G.DB && G.DB.mode === "local" && document.querySelector("#boot").classList.contains("hidden")')
            if page.locator('#tap-start').is_visible():
                page.locator('#tap-start').click()
            page.locator('#btn-guest').click()
            count = page.evaluate('G.PERSONALITY_TEST.length')
            for _ in range(count):
                page.locator('#test-a button').first.click()
            page.locator('#res-ok').click()
            # Dismiss optional upstream notices with their normal buttons if present.
            page.locator('#scr-menu [data-go="play"]').click(timeout=10000)
            page.wait_for_function('G.Game.state === "playing"')
            page.keyboard.down('d'); page.wait_for_timeout(500); page.keyboard.up('d')
            page.wait_for_timeout(3000)
            report = page.evaluate('''() => {
                const props = G.World.visibleProps({left:()=>-2200,top:()=>-2200,w:4400,h:4400,sees:()=>true});
                const terrain = new Set();
                for(let x=-1600;x<=1600;x+=32) for(let y=-1600;y<=1600;y+=32) terrain.add(G.World.kindAt(x,y));
                localStorage.setItem('test-isolation','local');
                return {mode:G.DB.mode, guest:G.DB.guest, state:G.Game.state,
                    terrain:[...terrain], props:[...new Set(props.map(p=>p.type))],
                    enemies:G.EnemyMgr.all().map(e=>({dex:e.dex,hp:e.hp,maxHp:e.maxHp,behavior:e.behavior})),
                    worldTime:G.Game.st.time, firebase:G.FIREBASE_CONFIG,
                    isolated:testNativeStorage.getItem('pokerl.fullgame.test-isolation') === 'local',
                    sentinel:testNativeStorage.getItem('ps.test-sentinel')};
            }''')
            assert report['mode'] == 'local' and report['firebase'] is None
            assert report['isolated'] and report['sentinel'] == 'unchanged'
            assert {'floor','wall','liquid'} <= set(report['terrain']), report
            assert {'tree','grass','rock','chest'} <= set(report['props']), report
            assert report['enemies'], report
            assert not errors, errors
            assert not external, external
            assert not missing, missing
            assert not page.evaluate('navigator.serviceWorker.controller')
            page.screenshot(path='results/full-game.png')
            persisted = page.evaluate('''async () => {
                await G.DB.logout();
                const result = await G.DB.register('pokerl_test', 'local-test-password');
                if (!result.ok) throw new Error(result.error);
                G.DB.save.partner=7; G.DB.grant(7,'starter'); G.DB.save.coins=777; G.DB.commit();
                return G.DB.save.coins;
            }''')
            assert persisted == 777
            page.reload()
            page.wait_for_function('window.G && G.DB && G.DB.save && G.DB.save.partner === 7')
            assert page.evaluate('G.DB.save.coins') == 777
            assert page.evaluate("testNativeStorage.getItem('ps.accounts')") is None
            report['local_save_restored'] = True
            assert not errors, errors
            assert not external, external
            Path('results/full-game-validation.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report, indent=2))
        finally:
            browser.close()

if __name__ == '__main__':main()
