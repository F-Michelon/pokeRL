"""Advance only when held-out comparisons pass; rerun with --start after inspection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--steps',default=1000000,type=int)
    p.add_argument('--envs',default=8,type=int);p.add_argument('--start',default=0,type=int);p.add_argument('--resume');args=p.parse_args()
    stages=['easy','dense','projectiles','terrain','varied'];resume=args.resume
    for i in range(args.start,len(stages)):
        config=f'configs/arena_{stages[i]}.json';out=Path(f'results/curriculum/{i}-{stages[i]}')
        command=[sys.executable,'-m','training.train_ppo','--config',config,'--steps',str(args.steps),'--envs',str(args.envs),'--device',args.device,'--output',str(out)]
        if resume:command+=['--resume',resume]
        subprocess.run(command,check=True);resume=str(out/'model.zip')
        report=out/'evaluation.json'
        subprocess.run([sys.executable,'-m','training.evaluate','--config',config,'--model',resume,'--output',str(report)],check=True)
        if not json.loads(report.read_text())['movement_gate']['passed']:
            print(f'Stopping at stage {i}: movement gate failed. Inspect {report}.');return
if __name__=='__main__':main()
