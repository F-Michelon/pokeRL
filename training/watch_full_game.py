"""Watch a saved full-game policy locally, or record it without a display."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from playwright.sync_api import Error as PlaywrightError

from training.full_game_env import FullGameEnv, LocalServer
from training.full_game_config import read_metadata, env_kwargs


def run_metadata(model_path):
    """Support final models and the trainer's checkpoints/ subdirectory."""
    return read_metadata(model_path)


def show_status(env, episode, episodes, speed, total, action=None, finished=False):
    info = env.latest['info']
    status = (
        f"Model playback · game {episode}/{episodes} · seed {info['seed']} · {speed:g}×\n"
        f"Survival {info['survival']:.1f}s · HP {info['hp']:.0f} · "
        f"Level {info['level']} · Kills {info['kills']} · XP {info['xp']:.0f}\n"
        f"Return {total:.1f} · {info['decision']}"
    )
    if action is not None:
        if action == 49:
            choice = 'Reroll upgrades'
        elif action >= 45:
            choice = f'Choose option {action - 44}'
        else:
            direction = ['Stay still', 'North', 'South', 'West', 'East',
                         'Northwest', 'Northeast', 'Southwest', 'Southeast'][action % 9]
            choice = direction + (f' / move slot {action // 9}' if action >= 9 else '')
        status += f" · {choice}"
    if finished:
        status += ' · ' + ('Died' if info['dead'] else 'Episode limit reached')
    env.page.evaluate('''text => {
        let box = document.getElementById('policy-status');
        if (!box) {
            box = document.createElement('div'); box.id = 'policy-status';
            box.style.cssText = 'position:fixed;bottom:12px;left:12px;z-index:99999;'
                + 'background:rgba(0,0,0,.8);color:white;padding:10px 14px;border-radius:8px;'
                + 'font:13px/1.5 monospace;max-width:560px;pointer-events:none';
            document.body.appendChild(box);
        }
        box.replaceChildren(...text.split(String.fromCharCode(10)).map(line => {
            const row = document.createElement('div');
            row.textContent = line; row.style.display = 'block'; return row;
        }));
    }''', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1_000_000_000)
    parser.add_argument('--speed', type=float, default=1, help='Playback speed; default: real time')
    parser.add_argument('--record', type=Path, help='Directory for WebM videos and a JSON report')
    parser.add_argument('--headless', action='store_true', help='Record without opening a window')
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.seed <= 4294967295 - (args.episodes - 1):
        parser.error('Use a positive episode count and uint32 episode seeds')
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    if args.headless and not args.record:
        parser.error('--headless requires --record')
    metadata = run_metadata(args.model)
    torch.set_num_threads(1)
    model = MaskablePPO.load(args.model, device='cpu')
    if args.record:
        # Keep each recording self-contained; never replace an earlier report.
        args.record.mkdir(parents=True, exist_ok=False)
    report = {'model': str(args.model), 'training': metadata, 'speed': args.speed, 'runs': []}
    interrupted = False
    with LocalServer() as server:
        env = FullGameEnv(server.url, **env_kwargs(metadata),
                          render_mode='human', headless=args.headless, video_dir=args.record)
        try:
            for episode in range(1, args.episodes + 1):
                obs, _ = env.reset(seed=args.seed + episode - 1)
                video = env.page.video
                total = 0.0
                show_status(env, episode, args.episodes, args.speed, total)
                # Let newly requested sprites arrive and show the initial position.
                env.page.wait_for_timeout(500)
                env.render()
                while True:
                    started = time.monotonic()
                    before = env.latest['info']['survival']
                    action = int(model.predict(obs, deterministic=True, action_masks=env.action_masks())[0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    env.render()
                    done = terminated or truncated
                    show_status(env, episode, args.episodes, args.speed, total, action, done)
                    # Menu choices take no simulation time; briefly show their outcome.
                    delay = max(info['survival'] - before, 0.3 if action >= 45 else 0) / args.speed
                    env.page.wait_for_timeout(max(0, delay - (time.monotonic() - started)) * 1000)
                    if done:
                        break
                result = {**info, 'return': total, 'truncated': truncated}
                print(json.dumps(result), flush=True)
                env.page.wait_for_timeout(1500)
                env.context.close()
                env.context = None
                if video:
                    path = args.record / f"episode-{episode:03d}-seed-{info['seed']}.webm"
                    video.save_as(path)
                    video.delete()
                    result['video'] = str(path)
                    print(f'Video saved: {path}', flush=True)
                report['runs'].append(result)
        except KeyboardInterrupt:
            interrupted = True
            print('Playback stopped.', flush=True)
        except PlaywrightError:
            if env.page.is_closed() or not env.browser.is_connected():
                interrupted = True
                print('Playback window closed.', flush=True)
            else:
                raise
        finally:
            env.close()
            if args.record:
                report['interrupted'] = interrupted
                (args.record / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
