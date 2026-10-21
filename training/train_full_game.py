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
from stable_baselines3.common.logger import configure
from training.full_game_env import ROOT, FullGameEnv, LocalServer
from training.upstream_revision import read_revision
from training.full_game_config import DEFAULTS, read_metadata, resolve_config


def make_env(url,config,log):
    return Monitor(FullGameEnv(url,duration=config['duration'],dex=config['dex'],
                   observation_version=config['observation_version'],reward_weights=config['reward_weights']),filename=log,
                   info_keywords=('survival','xp','damage','kills','level','healed','healingPickups','xpPickups',
                                  'upgrades','moveSwitches','itemChoices','bondChoices',
                                  'reward_survival','reward_xp','reward_kills','reward_damage','reward_death'))

class WallLimit(BaseCallback):
    def __init__(self,seconds):super().__init__();self.seconds=seconds
    def _on_training_start(self):self.started=time.monotonic()
    def _on_step(self):return not self.seconds or time.monotonic()-self.started<self.seconds


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',help='JSON experiment config; explicit CLI settings override it')
    for key in ('steps','envs','duration','dex','seed','n_steps','batch_size','n_epochs','checkpoint_steps'):
        p.add_argument('--'+key.replace('_','-'),type=int)
    for key in ('gamma','ent_coef','learning_rate','target_kl'):
        p.add_argument('--'+key.replace('_','-'),type=float)
    p.add_argument('--net-arch',type=int,nargs='+')
    p.add_argument('--observation-version',choices=['v1','v2-pickups','v2'])
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--max-seconds',type=int,default=0)
    args=p.parse_args()
    if args.max_seconds<0:p.error('Invalid time budget')
    try:
        parent=read_metadata(args.resume) if args.resume else None
        config=resolve_config(args.config,{k:getattr(args,k,None) for k in DEFAULTS},parent)
    except (ValueError,OSError) as error:
        p.error(str(error))
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.ones(1,device=args.device).add_(1).cpu()  # Detect unsupported GPU architectures early.
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise RuntimeError('Output already contains a run; use a new directory')
    revision=read_revision(ROOT/'upstream')
    lock=json.loads((ROOT/'upstream.lock.json').read_text())
    if revision!=lock['revision']:raise RuntimeError('Upstream revision mismatch')
    if parent and parent['upstream']!=lock:raise ValueError('Resume upstream revision differs')
    source_files=[ROOT/'full_game/training.js',ROOT/'full_game/game.js',ROOT/'training/full_game_env.py',
                  ROOT/'training/full_game_config.py',Path(__file__).resolve()]
    metadata={**vars(args),**config,'config':config,'config_file':args.config,'environment':'full-game-v2','upstream':lock,'python':platform.python_version(),
              'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','gymnasium','stable-baselines3','sb3-contrib','playwright']},
              'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}}
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with LocalServer() as server:
        env=SubprocVecEnv([functools.partial(make_env,server.url,config,str(out/f'env-{i}')) for i in range(config['envs'])],start_method='spawn')
        try:
            env.seed(config['seed'])
            model=MaskablePPO.load(args.resume,env=env,device=args.device) if args.resume else MaskablePPO(
                'MlpPolicy',env,**{k:config[k] for k in ('n_steps','batch_size','n_epochs','learning_rate','gamma','ent_coef','target_kl')},
                policy_kwargs={'net_arch':config['net_arch']},seed=config['seed'],device=args.device,verbose=1)
            # Loading a checkpoint restores its old seed and reseeds the env.
            # Honor the explicitly resolved seed for this training segment.
            model.seed=config['seed']
            model.set_random_seed(config['seed'])
            callbacks=CallbackList([CheckpointCallback(save_freq=max(1,config['checkpoint_steps']//config['envs']),save_path=str(out/'checkpoints'),name_prefix='full'),WallLimit(args.max_seconds)])
            model.set_logger(configure(str(out),['stdout','csv']))
            metadata['observation_size']=int(env.observation_space.shape[0])
            (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print('EXPERIMENT:',json.dumps(config,sort_keys=True),flush=True)
            started=time.monotonic();before=model.num_timesteps
            try:
                model.learn(total_timesteps=config['steps'],callback=callbacks,reset_num_timesteps=not bool(args.resume))
            except KeyboardInterrupt:
                print('Interrupted: saving current checkpoint',flush=True)
            model.save(out/'model')
            new_timesteps=model.num_timesteps-(before if args.resume else 0)
            metadata.update({'completed_timesteps':model.num_timesteps,'new_timesteps':new_timesteps,
                             'requested_steps_completed':new_timesteps>=config['steps'],'training_seconds':time.monotonic()-started})
            (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        finally:env.close()

if __name__=='__main__':main()
