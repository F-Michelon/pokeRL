"""Held-out full-game episodes with the same horizon and partner as training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.full_game_env import FullGameEnv, LocalServer


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-start',type=int,default=1_000_000_000);p.add_argument('--output',required=True)
    p.add_argument('--baselines',action='store_true');args=p.parse_args()
    if args.episodes<1 or args.seed_start<1_000_000_000:p.error('Use positive episode count and held-out seed range >= 1 billion')
    metadata=json.loads((Path(args.model).parent/'run.json').read_text())
    if metadata['environment']!='full-game-v1':raise ValueError('Not a full-game checkpoint')
    torch.set_num_threads(1);model=MaskablePPO.load(args.model,device='cpu')
    report={'model':args.model,'training':metadata,'episodes':args.episodes,'seed_start':args.seed_start,'policies':{}}
    with LocalServer() as server:
        env=FullGameEnv(server.url,duration=metadata['duration'],dex=metadata['dex'])
        try:
            for policy in ['ppo']+(['stand_first','circle_first','random_valid'] if args.baselines else []):
                runs=[]
                for seed in range(args.seed_start,args.seed_start+args.episodes):
                    obs,_=env.reset(seed=seed);rng=np.random.default_rng(seed);total=0;done=False
                    while not done:
                        mask=env.action_masks();info=env.latest['info']
                        if policy=='ppo':action=int(model.predict(obs,deterministic=True,action_masks=mask)[0])
                        elif policy=='random_valid':action=int(rng.choice(np.flatnonzero(mask)))
                        elif not mask[0]:action=int(np.flatnonzero(mask)[0])
                        elif policy=='stand_first':action=0
                        else:action=[4,8,2,7,3,5,1,6][int(info['survival']/2)%8]
                        obs,reward,terminated,truncated,info=env.step(action);total+=reward;done=terminated or truncated
                    runs.append({**info,'return':total,'truncated':truncated})
                keys=['survival','xp','damage','dead','return','upgrades','moveSwitches','itemChoices','bondChoices']
                report['policies'][policy]={'mean':{k:float(np.mean([r[k] for r in runs])) for k in keys},'runs':runs}
                print(policy,report['policies'][policy]['mean'],flush=True)
            out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
        finally:env.close()
if __name__=='__main__':main()
