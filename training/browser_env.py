"""Normal-speed localhost-only screenshot/WASD interface; no state reads in step."""
import io
import time
from urllib.parse import urlencode, urlsplit
import gymnasium as gym
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

KEYS=[[],['w'],['s'],['a'],['d'],['w','a'],['w','d'],['s','a'],['s','d']]
class BrowserEnv(gym.Env):
    def __init__(self, url='http://127.0.0.1:8000/rl/browser_runner/',config='arena_easy',randomize=False,headless=True,executable=None,graphics='upstream'):
        parts=urlsplit(url)
        if parts.scheme!='http' or parts.hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('Browser training and evaluation are restricted to localhost')
        if graphics not in ('upstream','debug'):raise ValueError('Unknown graphics mode')
        self.url=url;self.config=config;self.randomize=randomize;self.graphics=graphics
        self.action_space=gym.spaces.Discrete(9)
        self.observation_space=gym.spaces.Box(0,255,(3,90,160),dtype=np.uint8)
        self.pw=sync_playwright().start()
        self.browser=self.pw.chromium.launch(headless=headless,executable_path=executable)
        self.context=self.browser.new_context(viewport={'width':1280,'height':720},device_scale_factor=1)
        self.context.route('**/*',lambda route:route.continue_() if urlsplit(route.request.url).hostname in ('127.0.0.1','localhost','::1') else route.abort())
        self.page=self.context.new_page();self.held=set();self.ticks=0
        self.capture_seconds=0
    def capture(self):
        start=time.perf_counter()
        # A fixed canvas clip avoids repeated locator scrolling/stability waits.
        raw=self.page.screenshot(clip=self.canvas_box)
        im=Image.open(io.BytesIO(raw)).convert('RGB');pixel=im.getpixel((5,5))
        status={(255,0,0):'dead',(0,255,0):'complete',(255,0,255):'upgrade',(0,0,255):'ready'}.get(pixel,'playing')
        obs=np.asarray(im.resize((160,90))).transpose(2,0,1).copy()
        self.capture_seconds=time.perf_counter()-start
        return obs,status,raw
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        seed=int(seed) if seed is not None else int(self.np_random.integers(0,1_000_000_000))
        query=urlencode({'seed':seed,'config':self.config,'randomize':int(self.randomize),'graphics':self.graphics})
        self.page.goto(self.url+'?'+query);self.page.wait_for_function('window.arenaReady === true')
        self.canvas_box=self.page.locator('#game').bounding_box()
        self.held=set();self.ticks=0
        obs,status,_=self.capture()
        if status!='ready':raise RuntimeError(f'Unexpected initial visual state: {status}')
        self.page.keyboard.press('Space')
        self.page.wait_for_timeout(20)
        obs,_,_=self.capture()
        return obs,{}
    def step(self,action):
        wanted=set(KEYS[int(action)])
        for key in self.held-wanted:self.page.keyboard.up(key)
        for key in wanted-self.held:self.page.keyboard.down(key)
        self.held=wanted
        # Budget the preceding capture cost into the next ~100 ms control cycle.
        self.page.wait_for_timeout(max(0,100-self.capture_seconds*1000));self.ticks+=1
        obs,status,_=self.capture()
        if status=='upgrade':self.page.keyboard.press('3')
        terminated=status=='dead';truncated=status=='complete' or self.ticks>=1200
        # Screenshot-only survival reward; structured XP/damage deliberately unavailable.
        return obs,(-10 if terminated else .01),terminated,truncated,{'visual_status':status}
    def close(self):
        self.context.close();self.browser.close();self.pw.stop()
