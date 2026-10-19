"""Measure canvas capture latency on the target laptop with the local server up."""
import argparse
import json
import time
import numpy as np
from training.browser_env import BrowserEnv

def main():
    p=argparse.ArgumentParser();p.add_argument('--executable');p.add_argument('--samples',type=int,default=20);args=p.parse_args()
    env=BrowserEnv(executable=args.executable)
    try:
        env.reset(seed=10);durations=[]
        for _ in range(args.samples):
            start=time.perf_counter();env.capture();durations.append(1000*(time.perf_counter()-start))
        print(json.dumps({'canvas_capture_median_ms':float(np.median(durations)),
                          'canvas_capture_p95_ms':float(np.percentile(durations,95))},indent=2))
    finally:env.close()
if __name__=='__main__':main()
