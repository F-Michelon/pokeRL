"""Paired held-out evaluation of final models and optional learning checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.full_game_env import FullGameEnv, LocalServer
from training.full_game_config import LEGACY_REWARD, read_metadata, env_kwargs


def evaluate(model_path, args, server, baselines=False):
    metadata = read_metadata(model_path)
    model = MaskablePPO.load(model_path, device='cpu')
    settings = env_kwargs(metadata)
    if args.duration is not None:
        settings['duration'] = args.duration
    report = dict(model=str(model_path), training=metadata, model_timesteps=model.num_timesteps,
                  episodes=args.episodes, seed_start=args.seed_start,
                  duration=settings['duration'], dex=settings['dex'], policies={})
    env = FullGameEnv(server.url, **settings)
    try:
        if model.observation_space.shape != env.observation_space.shape:
            raise ValueError('Checkpoint observation shape does not match its metadata')
        for policy in ['ppo'] + (['stand_first', 'circle_first', 'random_valid'] if baselines else []):
            runs = []
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                obs, _ = env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                total = 0
                done = False
                while not done:
                    mask = env.action_masks()
                    info = env.latest['info']
                    if policy == 'ppo':
                        action = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
                    elif policy == 'random_valid':
                        action = int(rng.choice(np.flatnonzero(mask)))
                    elif not mask[0]:
                        action = int(np.flatnonzero(mask)[0])
                    elif policy == 'stand_first':
                        action = 0
                    else:
                        action = [4, 8, 2, 7, 3, 5, 1, 6][int(info['survival'] / 2) % 8]
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    done = terminated or truncated
                # Fixed reference score remains comparable when reward weights differ.
                reference = sum(LEGACY_REWARD[k] * info[k] for k in ('survival', 'xp', 'kills'))
                reference -= LEGACY_REWARD['damage'] * info['damage'] + LEGACY_REWARD['death'] * info['dead']
                runs.append({**info, 'return': total, 'referenceReturn': reference,
                             'horizonReached': not info['dead'] and info['survival'] >= settings['duration'],
                             'truncated': truncated})
            keys = ['survival', 'xp', 'damage', 'dead', 'return', 'referenceReturn', 'horizonReached',
                    'kills', 'level', 'hpFraction', 'healed', 'healingPickups', 'xpPickups',
                    'upgrades', 'moveSwitches', 'itemChoices', 'bondChoices']
            means = {k: float(np.mean([r[k] for r in runs])) for k in keys}
            sem = {k: float(np.std([r[k] for r in runs], ddof=1) / np.sqrt(len(runs)))
                   if len(runs) > 1 else None for k in keys}
            components = {k: float(np.mean([r['rewardComponents'][k] for r in runs])) for k in LEGACY_REWARD}
            report['policies'][policy] = dict(mean=means, sem=sem, reward_components=components, runs=runs)
            print(str(model_path), policy, means, flush=True)
        return report
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed-start', type=int, default=1_000_000_000)
    p.add_argument('--duration', type=int, help='Override horizon for an explicitly longer evaluation')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--baselines', action='store_true')
    p.add_argument('--checkpoints', action='store_true', help='Also evaluate checkpoints/; baselines run once for the final model')
    args = p.parse_args()
    if args.episodes < 1 or not 1_000_000_000 <= args.seed_start <= 4294967295 - (args.episodes - 1):
        p.error('Use positive episode count and held-out uint32 seeds >= 1 billion')
    if args.duration is not None and not 1 <= args.duration <= 3600:
        p.error('duration must be 1..3600')
    if args.output.exists():
        p.error('Evaluation output already exists; use a new path')
    torch.set_num_threads(1)
    paths = []
    if args.checkpoints:
        checkpoint_dir = args.model.parent if args.model.parent.name == 'checkpoints' else args.model.parent / 'checkpoints'
        paths = sorted((path for path in checkpoint_dir.glob('full_*_steps.zip') if path != args.model),
                       key=lambda path: int(path.stem.split('_')[1]))
    paths.append(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    with LocalServer() as server:
        for path in paths:
            reports.append(evaluate(path, args, server, baselines=args.baselines and path == args.model))
            payload = {'evaluations': reports} if args.checkpoints else reports[0]
            # Preserve completed models if a subsequent evaluation hits a time limit.
            temp = args.output.with_suffix(args.output.suffix + '.tmp')
            temp.write_text(json.dumps(payload, indent=2) + '\n')
            temp.replace(args.output)


if __name__ == '__main__':
    main()
