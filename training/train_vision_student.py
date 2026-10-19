"""Distill a PPO CNN policy, preserving its checkpoint format for browser RL."""
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader,TensorDataset

class ImageSpace(gym.Env):
    observation_space=gym.spaces.Box(0,255,(3,90,160),dtype=np.uint8)
    action_space=gym.spaces.Discrete(9)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='results/teacher');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--device',default='cpu');p.add_argument('--output',default='results/student');args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(123)
    files=sorted(Path(args.data).glob('*.npz'))
    if len(files)<2:p.error('Record at least two episodes for an episode-disjoint validation split')
    def dataset(paths):
        chunks=[]
        for path in paths:
            with np.load(path) as f:chunks.append((f['images'],f['actions']))
        return TensorDataset(torch.from_numpy(np.concatenate([c[0] for c in chunks])),torch.from_numpy(np.concatenate([c[1] for c in chunks])))
    cut=max(1,int(len(files)*.8));train=DataLoader(dataset(files[:cut]),batch_size=64,shuffle=True);valid=DataLoader(dataset(files[cut:]),batch_size=64)
    model=PPO('CnnPolicy',ImageSpace(),device=args.device,n_steps=128,batch_size=64,seed=123)
    for epoch in range(args.epochs):
        model.policy.set_training_mode(True)
        for x,y in train:
            x=x.to(model.device);y=y.to(model.device)
            loss=torch.nn.functional.cross_entropy(model.policy.get_distribution(x).distribution.logits,y)
            model.policy.optimizer.zero_grad();loss.backward();model.policy.optimizer.step()
        model.policy.set_training_mode(False);correct=total=0
        with torch.no_grad():
            for x,y in valid:
                pred=model.policy.get_distribution(x.to(model.device)).distribution.logits.argmax(1).cpu()
                correct+=int((pred==y).sum());total+=len(y)
        print(f'epoch={epoch+1} held_out_accuracy={correct/total:.3f}',flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);model.save(args.output)
if __name__=='__main__':main()
