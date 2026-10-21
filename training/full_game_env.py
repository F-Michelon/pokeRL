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
from training.browser_setup import configure_browser_cache
from training.full_game_config import LEGACY_REWARD

ROOT = Path(__file__).resolve().parents[1]

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Reset/close can cancel in-flight sprite downloads.
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
    metadata={'render_modes':['human'],'render_fps':10}
    def __init__(self,url,duration=120,dex=7,executable=None,*,render_mode=None,headless=None,video_dir=None,
                 observation_version='v2',reward_weights=None):
        if urlsplit(url).hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('Full-game training is restricted to localhost')
        if not 1<=duration<=3600:raise ValueError('duration must be 1..3600 seconds')
        if render_mode not in (None,'human'):raise ValueError('Unsupported render mode')
        if video_dir and render_mode is None:raise ValueError('Recording requires render_mode="human"')
        self.render_mode=render_mode;self.video_dir=video_dir
        if observation_version not in ('v1','v2-pickups','v2'):raise ValueError('Unknown observation version')
        self.observation_version=observation_version
        self.reward_weights=dict(LEGACY_REWARD if reward_weights is None else reward_weights)
        self.url=url;self.duration=duration;self.dex=dex;self.context=None;self.page=None
        self.action_space=gym.spaces.Discrete(50)
        # The driver resolves both Chromium and recording FFmpeg at startup.
        configure_browser_cache(ROOT)
        self.pw=sync_playwright().start()
        try:
            self.browser=self.pw.chromium.launch(headless=(render_mode is None) if headless is None else headless,executable_path=executable or os.environ.get('POKERL_CHROME'),
                args=['--disable-background-timer-throttling'])
            initial=self._reset_game(0,record=False)
            self.observation_space=gym.spaces.Box(-10,10,shape=(len(initial['observation']),),dtype=np.float32)
        except BaseException:
            self.close();raise

    def _reset_game(self,seed,record=True):
        if self.context:self.context.close()
        video_options={'record_video_dir':str(self.video_dir),'record_video_size':{'width':1280,'height':720}} if self.video_dir and record else {}
        self.context=self.browser.new_context(viewport={'width':1280,'height':720},device_scale_factor=1,**video_options)
        self.context.add_init_script(INIT)
        blocked=('media',) if self.render_mode else ('image','font','media')
        self.context.route('**/*',lambda route:route.continue_() if urlsplit(route.request.url).hostname in ('127.0.0.1','localhost','::1') and route.request.resource_type not in blocked else route.abort())
        self.page=self.context.new_page();self.errors=[]
        self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        self.page.goto(self.url+'?training=1',wait_until='load')
        self.page.wait_for_function('window.G && G.Flow && G.Game && document.querySelector("#boot").classList.contains("hidden")',polling=50)
        if self.render_mode and not self.page.evaluate('typeof G.Game.rlRender === "function"'):
            raise RuntimeError('Rebuild the visual adapter first: python scripts/build_full_game.py')
        # Self-hosted script injection is allowed by the local entrypoint CSP.
        self.page.add_script_tag(url='/full_game/training.js')
        result=self.page.evaluate('(config)=>fullTraining.start(config)',{'seed':int(seed),'duration':self.duration,'dex':self.dex,
            'observationVersion':self.observation_version,'rewardWeights':self.reward_weights})
        self._check(result);self.latest=result
        if self.render_mode:
            self.page.evaluate('document.querySelector("#tap-start")?.classList.add("hidden")')
            # The simulation clock is frozen during loading, so use a real timer
            # to bound preloading even if an image request never completes.
            self.page.evaluate('''dex => Promise.race([
                G.Sprites.preload([dex]),
                new Promise(resolve => setTimeout(resolve, 5000))
            ])''', self.dex)
            self.render()
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

    def render(self):
        if self.render_mode:
            self.page.evaluate('G.Game.rlRender()')

    def close(self):
        if self.context:self.context.close();self.context=None
        if hasattr(self,'browser'):self.browser.close()
        if hasattr(self,'pw'):self.pw.stop()
