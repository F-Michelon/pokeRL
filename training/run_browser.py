import argparse
import json
import time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from training.browser_env import BrowserEnv

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--config',default='arena_easy')
    p.add_argument('--finetune-steps',type=int,default=0);p.add_argument('--output',default='results/student_finetuned')
    p.add_argument('--device',default='cpu');p.add_argument('--executable');p.add_argument('--visible',action='store_true')
    p.add_argument('--graphics',choices=['upstream','debug'],default='upstream');args=p.parse_args()
    env=BrowserEnv(config=args.config,randomize=bool(args.finetune_steps),headless=not args.visible,executable=args.executable,graphics=args.graphics)
    try:
        model=PPO.load(args.model,env=env,device=args.device)
        if args.finetune_steps:
            model.learn(total_timesteps=args.finetune_steps);model.save(args.output)
        else:
            obs,_=env.reset(seed=1_000_000_000);done=False;start=time.perf_counter();steps=0
            while not done:
                action,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=env.step(action);done=term or trunc;steps+=1
            report={**info,'wall_seconds':time.perf_counter()-start,'steps':steps,'seed':1_000_000_000,
                    'model':args.model,'config':args.config,'device':args.device,'graphics':args.graphics}
            output=Path(args.output+'.json');output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(report,indent=2)+'\n');print(report)
    finally:env.close()
if __name__=='__main__':main()
