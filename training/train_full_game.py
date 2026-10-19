"""Train masked PPO on the full upstream game's structured state."""
import argparse
import functools
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from training.full_game_env import ROOT, FullGameEnv, LocalServer
from training.upstream_revision import read_revision


def make_env(url,duration,dex,log):
    return Monitor(FullGameEnv(url,duration=duration,dex=dex),filename=log,
                   info_keywords=('survival','xp','damage','upgrades','moveSwitches','itemChoices','bondChoices'))

class WallLimit(BaseCallback):
    def __init__(self,seconds):super().__init__();self.seconds=seconds
    def _on_training_start(self):self.started=time.monotonic()
    def _on_step(self):return not self.seconds or time.monotonic()-self.started<self.seconds


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--envs',type=int,default=4)
    p.add_argument('--duration',type=int,default=120);p.add_argument('--dex',type=int,default=7)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--max-seconds',type=int,default=0)
    args=p.parse_args()
    if args.steps<1 or args.envs<1 or args.max_seconds<0:p.error('Invalid steps/envs/time budget')
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.ones(1,device=args.device).add_(1).cpu()  # Detect unsupported GPU architectures early.
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise RuntimeError('Output already contains a run; use a new directory')
    revision=read_revision(ROOT/'upstream')
    lock=json.loads((ROOT/'upstream.lock.json').read_text())
    if revision!=lock['revision']:raise RuntimeError('Upstream revision mismatch')
    source_files=[ROOT/'full_game/training.js',ROOT/'full_game/game.js',ROOT/'training/full_game_env.py',Path(__file__).resolve()]
    metadata={**vars(args),'environment':'full-game-v1','upstream':lock,'python':platform.python_version(),
              'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','gymnasium','stable-baselines3','sb3-contrib','playwright']},
              'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}}
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with LocalServer() as server:
        env=SubprocVecEnv([functools.partial(make_env,server.url,args.duration,args.dex,str(out/f'env-{i}')) for i in range(args.envs)],start_method='spawn')
        try:
            env.seed(args.seed)
            model=MaskablePPO.load(args.resume,env=env,device=args.device) if args.resume else MaskablePPO(
                'MlpPolicy',env,n_steps=128,batch_size=64,n_epochs=4,learning_rate=3e-4,
                policy_kwargs={'net_arch':[256,256]},seed=args.seed,device=args.device,verbose=1)
            callbacks=CallbackList([CheckpointCallback(save_freq=max(1,10000//args.envs),save_path=str(out/'checkpoints'),name_prefix='full'),WallLimit(args.max_seconds)])
            started=time.monotonic();before=model.num_timesteps
            try:
                model.learn(total_timesteps=args.steps,callback=callbacks,reset_num_timesteps=not bool(args.resume))
            except KeyboardInterrupt:
                print('Interrupted: saving current checkpoint',flush=True)
            model.save(out/'model')
            metadata.update({'completed_timesteps':model.num_timesteps,'new_timesteps':model.num_timesteps-(before if args.resume else 0),'training_seconds':time.monotonic()-started})
            (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        finally:env.close()

if __name__=='__main__':main()
