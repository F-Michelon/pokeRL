"""Full upstream game, fixed-step Chromium simulation, structured observations.

No screenshots or wall-clock waits in step. A fresh browser context resets all
upstream closures and guest progression between episodes.
"""
import functools
import http.server
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit
import gymnasium as gym
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

class LocalServer:
    def __enter__(self):
        handler=functools.partial(QuietHandler,directory=str(ROOT))
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}/full_game/'
        return self
    def __exit__(self,*_args):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)

INIT = r'''(() => {
  window.POKERL_TRAINING = true;
  window.pokerlClock = 0;
  window.pokerlSeed = seed => {
    let state=seed>>>0;
    window.pokerlRandom=()=>{state=(state+0x6d2b79f5)>>>0;let t=Math.imul(state^(state>>>15),1|state);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296;};
  };
  window.pokerlSeed(0);
  window.requestAnimationFrame=()=>0;
  window.cancelAnimationFrame=()=>{};
  Object.defineProperty(performance,'now',{value:()=>window.pokerlClock});
  const NativeDate=Date, epoch=NativeDate.UTC(2026,9,19);
  window.Date=class extends NativeDate {
    constructor(...args){super(...(args.length?args:[epoch+window.pokerlClock]));}
    static now(){return epoch+window.pokerlClock;}
  };
})();'''

class FullGameEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,url,duration=120,dex=7,executable=None):
        if urlsplit(url).hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('Full-game training is restricted to localhost')
        if not 1<=duration<=3600:raise ValueError('duration must be 1..3600 seconds')
        self.url=url;self.duration=duration;self.dex=dex;self.context=None;self.page=None
        self.action_space=gym.spaces.Discrete(50)
        self.pw=sync_playwright().start()
        try:
            self.browser=self.pw.chromium.launch(headless=True,executable_path=executable or os.environ.get('POKERL_CHROME'),
                args=['--disable-background-timer-throttling'])
            initial=self._reset_game(0)
            self.observation_space=gym.spaces.Box(-10,10,shape=(len(initial['observation']),),dtype=np.float32)
        except BaseException:
            self.close();raise

    def _reset_game(self,seed):
        if self.context:self.context.close()
        self.context=self.browser.new_context(viewport={'width':1280,'height':720},device_scale_factor=1)
        self.context.add_init_script(INIT)
        self.context.route('**/*',lambda route:route.continue_() if urlsplit(route.request.url).hostname in ('127.0.0.1','localhost','::1') and route.request.resource_type not in ('image','font','media') else route.abort())
        self.page=self.context.new_page();self.errors=[]
        self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        self.page.goto(self.url+'?training=1',wait_until='load')
        self.page.wait_for_function('window.G && G.Flow && G.Game && document.querySelector("#boot").classList.contains("hidden")',polling=50)
        # Self-hosted script injection is allowed by the local entrypoint CSP.
        self.page.add_script_tag(url='/full_game/training.js')
        result=self.page.evaluate('(config)=>fullTraining.start(config)',{'seed':int(seed),'duration':self.duration,'dex':self.dex})
        self._check(result);self.latest=result
        return result

    def _check(self,result):
        if self.errors:raise RuntimeError('Full-game browser error: '+'; '.join(self.errors))
        if not np.isfinite(result['observation']).all():raise RuntimeError('Non-finite observation')
        if not any(result['mask']):raise RuntimeError('No valid actions')

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        actual=int(seed) if seed is not None else int(self.np_random.integers(0,1_000_000_000))
        if not 0<=actual<=4294967295:raise ValueError('Seed must be uint32')
        result=self._reset_game(actual)
        return np.asarray(result['observation'],dtype=np.float32),result['info']

    def step(self,action):
        result=self.page.evaluate('(action)=>fullTraining.step(action)',int(action))
        self._check(result);self.latest=result
        return np.asarray(result['observation'],dtype=np.float32),result['reward'],result['terminated'],result['truncated'],result['info']

    def action_masks(self):
        return np.asarray(self.latest['mask'],dtype=bool)

    def close(self):
        if self.context:self.context.close();self.context=None
        if hasattr(self,'browser'):self.browser.close()
        if hasattr(self,'pw'):self.pw.stop()
