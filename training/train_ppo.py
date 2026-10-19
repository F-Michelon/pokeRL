"""python -m training.train_ppo --config configs/arena_easy.json"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from training.env import PokeEnv, ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/arena_easy.json')
    p.add_argument('--steps',type=int,default=1_000_000)
    p.add_argument('--envs',type=int,default=8)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='results/ppo')
    p.add_argument('--resume')
    args=p.parse_args()
    if args.steps<=0 or args.envs<=0: p.error('steps and envs must be positive')
    torch.set_num_threads(1)
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    metadata={**vars(args),'arena':config,'upstream':json.loads((ROOT/'upstream.lock.json').read_text()),
              'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','gymnasium','stable-baselines3']},
              'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((ROOT/'rl').glob('*.js'))}}
    (output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    env=SubprocVecEnv([lambda:Monitor(PokeEnv(config)) for _ in range(args.envs)],start_method='spawn')
    env.seed(args.seed)
    try:
        model=PPO.load(args.resume,env=env,device=args.device) if args.resume else PPO(
            'MlpPolicy',env,n_steps=256,batch_size=256,n_epochs=5,learning_rate=3e-4,
            policy_kwargs={'net_arch':[128,128]},seed=args.seed,device=args.device,verbose=1)
        checkpoint=CheckpointCallback(save_freq=max(1,50_000//args.envs),save_path=str(output/'checkpoints'),name_prefix='ppo')
        model.learn(total_timesteps=args.steps,reset_num_timesteps=not bool(args.resume),callback=checkpoint)
        model.save(output/'model')
    finally:
        env.close()

if __name__=='__main__':main()
