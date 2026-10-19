"""Measure checkpoint inference on the machine that will control the browser."""
import argparse
import json
import time
import numpy as np
import torch
from stable_baselines3 import PPO

def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--iterations',type=int,default=500);args=p.parse_args()
    torch.set_num_threads(1);model=PPO.load(args.model,device='cpu')
    obs=np.zeros(model.observation_space.shape,dtype=model.observation_space.dtype)
    for _ in range(20):model.predict(obs,deterministic=True)
    times=[]
    for _ in range(args.iterations):
        start=time.perf_counter();model.predict(obs,deterministic=True);times.append((time.perf_counter()-start)*1000)
    print(json.dumps({'device':'cpu','parameters':sum(p.numel() for p in model.policy.parameters()),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'includes_browser_capture':False},indent=2))
if __name__=='__main__':main()
