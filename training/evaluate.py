"""Paired held-out seeds for PPO and three deterministic movement baselines."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.env import PokeEnv

ACTIONS=np.array([[0,0],[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]],dtype=float)
ACTIONS[5:]/=np.sqrt(2)
def baseline(name,obs,t):
    if name=='stand':return 0
    if name=='circle':return [4,8,2,7,3,5,1,6][int(t/2)%8]
    if name=='flee':
        enemies=obs[12:172].reshape(20,8)
        if not enemies[0,0]:return 0
        direction=-enemies[0,1:3]*[1200,660]
        return int(np.argmax(ACTIONS@direction))
    raise ValueError(name)

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/arena_easy.json')
    p.add_argument('--model');p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--seed-start',type=int,default=1_000_000_000)
    p.add_argument('--output',default='results/baselines.json');args=p.parse_args()
    if args.episodes<1:p.error('episodes must be positive')
    config=json.loads(Path(args.config).read_text());env=PokeEnv(config)
    model=PPO.load(args.model,device='cpu') if args.model else None
    report={'config':config,'seed_start':args.seed_start,'episodes':args.episodes,'policies':{}}
    try:
        for policy in ['stand','circle','flee']+(['ppo'] if model else []):
            runs=[]
            for seed in range(args.seed_start,args.seed_start+args.episodes):
                obs,_=env.reset(seed=seed);done=False;total=0;t=0
                while not done:
                    action=model.predict(obs,deterministic=True)[0] if policy=='ppo' else baseline(policy,obs,t)
                    obs,r,terminated,truncated,info=env.step(action);total+=r;t=info['survival'];done=terminated or truncated
                runs.append({**info,'return':total})
            report['policies'][policy]={'mean':{k:float(np.mean([r[k] for r in runs])) for k in ['survival','xpPerMinute','damagePerMinute','dead','return']},'runs':runs}
        if model:
            teacher=report['policies']['ppo'];comparisons={}
            for name in ['stand','circle','flee']:
                differences=np.array([a['return']-b['return'] for a,b in zip(teacher['runs'],report['policies'][name]['runs'])])
                sem=float(differences.std(ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else None
                comparisons[name]={'paired_return_difference':float(differences.mean()),'lower_95_normal':float(differences.mean()-1.96*sem) if sem is not None else None}
            report['movement_gate']={'passed':args.episodes>=30 and all(x['lower_95_normal']>0 for x in comparisons.values()),'comparisons':comparisons,'criterion':'30+ paired seeds; positive lower 95% normal CI of return difference against every baseline'}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v['mean'] for k,v in report['policies'].items()},indent=2))
        if model:print(report['movement_gate'])
    finally:env.close()
if __name__=='__main__':main()
