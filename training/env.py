"""Gymnasium bridge: one isolated Node process per environment."""
import json
import select
import subprocess
from pathlib import Path
import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OBS_SIZE = 397

class PokeEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, config=None):
        self.config = config or {}
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), dtype=np.float32)
        self.process = subprocess.Popen(['node', str(ROOT/'rl/server.js')], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.request(op='configure', config=self.config)

    def request(self, **message):
        if self.process.poll() is not None:
            raise RuntimeError('Node environment exited')
        self.process.stdin.write(json.dumps(message)+'\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 30)[0]:
            self.close()
            raise TimeoutError('Node environment timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Node environment closed its output')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Training uses a disjoint range from held-out evaluation (>= 1 billion).
        actual = int(seed) if seed is not None else int(self.np_random.integers(0, 1_000_000_000))
        result = self.request(op='reset', seed=actual)
        return np.asarray(result['observation'], dtype=np.float32), result['info']

    def step(self, action):
        result = self.request(op='step', action=int(action), frames=6)
        while result['observation'][10]:
            # Separate, event-driven strategic baseline; movement policy never chooses upgrades.
            choice = 2 if result['observation'][2] < .6 else 0
            result['observation'] = self.request(op='upgrade', choice=choice)['observation']
        return (np.asarray(result['observation'], dtype=np.float32), result['reward'],
                result['terminated'], result['truncated'], result['info'])

    def close(self):
        if not hasattr(self, 'process'):
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
