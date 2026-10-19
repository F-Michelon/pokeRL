import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.browser_env import BrowserEnv

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--config',default='arena_easy');p.add_argument('--output',default='results/teacher')
    p.add_argument('--executable');p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--graphics',choices=['upstream','debug'],default='upstream');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    model=PPO.load(args.model,device='cpu');env=BrowserEnv(config=args.config,randomize=True,executable=args.executable,graphics=args.graphics)
    summaries=[]
    try:
        for episode in range(args.episodes):
            image,_=env.reset(seed=args.seed+episode);images=[];actions=[];done=False;start=time.perf_counter()
            while not done:
                obs=np.asarray(env.page.evaluate('window.teacher.observation()'),dtype=np.float32)
                action=int(model.predict(obs,deterministic=True)[0]);images.append(image);actions.append(action)
                image,_,term,trunc,_=env.step(action);done=term or trunc
            np.savez_compressed(out/f'{episode:05d}.npz',images=np.stack(images),actions=np.array(actions,dtype=np.int64))
            summaries.append({'episode':episode,'samples':len(actions),'wall_seconds':time.perf_counter()-start,
                              'metrics':env.page.evaluate('window.teacher.metrics()')})
            print(summaries[-1],flush=True)
        (out/'metadata.json').write_text(json.dumps({**vars(args),'episodes_recorded':summaries},indent=2)+'\n')
    finally:env.close()
if __name__=='__main__':main()
