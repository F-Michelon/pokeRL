"""Check batch argument forwarding without submitting jobs or requiring a GPU."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SlurmArgumentsTests(unittest.TestCase):
    def run_batch(self, args, overrides=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('upstream/js', '.venv/bin', 'bin', 'full_game', 'rl', 'run/checkpoints'):
                (root / name).mkdir(parents=True, exist_ok=True)
            for name in ('rl/upstream-source.js', 'full_game/index.html', 'full_game/main.js',
                         'full_game/game.js', 'run/model.zip', 'run/run.json',
                         'run/checkpoints/full_500000_steps.zip'):
                (root / name).touch()
            (root / '.venv/bin/activate').write_text(
                f'export PATH={shlex.quote(str(root / "bin"))}:"$PATH"\n')
            for name, body in {
                'python': '#!/bin/bash\nexit 0\n',
                'srun': '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n',
            }.items():
                path = root / 'bin' / name
                path.write_text(body)
                path.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith('POKERL_')}
            env.update(SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID='12345', CAPTURE=str(root / 'args'))
            env.update(overrides or {})
            subprocess.run(['bash', str(ROOT / 'cluster/full-game.slurm'), *args],
                           env=env, check=True, capture_output=True, text=True)
            return (root / 'args').read_text().splitlines()

    def test_config_is_forwarded_without_silently_overriding_it(self):
        args = self.run_batch(['train', 'configs/full_large.json'])
        self.assertIn('configs/full_large.json', args)
        for flag in ('--steps', '--duration', '--envs', '--gamma'):
            self.assertNotIn(flag, args)

    def test_resume_and_explicit_overrides(self):
        args = self.run_batch(['train'], {
            'POKERL_RESUME': '/tmp/prior model.zip', 'POKERL_STEPS': '2000000',
            'POKERL_SEED': '456', 'POKERL_ENT_COEF': '0.01',
        })
        for flag, value in (('--resume', '/tmp/prior model.zip'), ('--steps', '2000000'),
                            ('--seed', '456'), ('--ent-coef', '0.01')):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn('--config', args)

    def test_paired_checkpoint_evaluation_options(self):
        args = self.run_batch(['eval', 'run'], {
            'POKERL_EVAL_MODEL': 'run/checkpoints/full_500000_steps.zip',
            'POKERL_BASELINES': '0', 'POKERL_EVAL_CHECKPOINTS': '1',
            'POKERL_EPISODES': '50', 'POKERL_EVAL_SEED': '2000000000',
            'POKERL_EVAL_DURATION': '300',
        })
        self.assertNotIn('--baselines', args)
        self.assertIn('--checkpoints', args)
        self.assertEqual(args[args.index('--seed-start') + 1], '2000000000')
        self.assertEqual(args[args.index('--duration') + 1], '300')
        self.assertEqual(args[args.index('--model') + 1], 'run/checkpoints/full_500000_steps.zip')


if __name__ == '__main__':
    unittest.main()
