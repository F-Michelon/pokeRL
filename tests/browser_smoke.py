from training.browser_env import BrowserEnv
from training.env import PokeEnv
import numpy as np
import os

def main():
    env=BrowserEnv(executable=os.environ.get('POKERL_CHROME'))
    errors=[];env.page.on('pageerror',lambda e:errors.append(str(e)))
    failed=[]
    env.page.on('response',lambda response:failed.append(response.url) if response.status>=400 else None)
    try:
        obs,_=env.reset(seed=17)
        assert obs.shape==(3,90,160)
        assert env.page.evaluate('window.G.Sprites.sheet(7,"Idle").ready')
        # Teacher-only state access here verifies the renderer's underlying simulation.
        before=env.page.evaluate('window.teacher.observation()')
        for _ in range(5):obs,_,_,_,info=env.step(4)
        after=env.page.evaluate('window.teacher.observation()')
        assert after[0]>before[0]
        assert not errors,errors
        assert not failed,failed
        env.page.locator('#game').screenshot(path='results/browser-smoke.png')
        print({'visual_status':info['visual_status'],'movement_verified':True,'page_errors':errors})
    finally:env.close()
if __name__=='__main__':main()
