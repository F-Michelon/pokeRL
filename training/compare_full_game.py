"""Compare saved evaluations on identical episode seeds, partner and horizon."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.reports:
        report = json.loads(path.read_text())
        rows.extend(report.get('evaluations', [report]))
    contracts = {(r['seed_start'], r['episodes'], r.get('duration', r['training']['duration']),
                  r.get('dex', r['training']['dex'])) for r in rows}
    if len(contracts) != 1:
        parser.error('Reports must use identical evaluation seeds, episode counts, duration and partner')
    print('| Model | Steps | Survival (s ± SE) | Horizon reached | XP | Damage | Healed | Healing pickups |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for report in rows:
        policy = report['policies']['ppo']
        m = policy['mean']
        sem = policy.get('sem', {}).get('survival')
        uncertainty = f' ± {sem:.1f}' if sem is not None else ''
        print(f"| {report['model']} | {report.get('model_timesteps', '?')} | "
              f"{m['survival']:.1f}{uncertainty} | {m.get('horizonReached', 1-m['dead']):.0%} | "
              f"{m['xp']:.1f} | {m['damage']:.1f} | {m.get('healed', 0):.1f} | {m.get('healingPickups', 0):.2f} |")


if __name__ == '__main__':
    main()
