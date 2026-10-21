import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.full_game_config import (
    DEFAULTS, config_from_metadata, env_kwargs, read_metadata, resolve_config,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_presets_change_one_experimental_factor_at_a_time(self):
        names = ['baseline', 'pickups', 'danger', 'survival', 'horizon', 'large']
        expected = ['observation_version', 'observation_version', 'reward_weights', 'gamma', 'net_arch']
        configs = [resolve_config(Path('configs') / f'full_{name}.json') for name in names]
        for before, after, key in zip(configs, configs[1:], expected):
            self.assertEqual([k for k in before if before[k] != after[k]], [key])

    def test_cli_overrides_and_typo_rejection(self):
        config = resolve_config('configs/full_large.json', {'steps': 512, 'gamma': None})
        self.assertEqual(config['steps'], 512)
        self.assertEqual(config['net_arch'], [512, 512])
        for override in ({'gama': 0.9}, {'gamma': float('nan')}, {'batch_size': 63},
                         {'net_arch': [0]}, {'reward_weights': {'survival': 1}}):
            with self.assertRaises(ValueError):
                resolve_config(overrides=override)

    def test_old_checkpoints_select_legacy_observations(self):
        old = dict(environment='full-game-v1', duration=120, dex=7, seed=123, steps=100000, envs=4)
        self.assertEqual(config_from_metadata(old)['observation_version'], 'v1')
        self.assertEqual(env_kwargs(old)['observation_version'], 'v1')
        resumed = resolve_config(overrides={'steps': 200000}, resume_metadata=old)
        self.assertEqual(resumed['observation_version'], 'v1')
        self.assertEqual(resumed['steps'], 200000)

    def test_resume_inherits_settings_but_rejects_experiment_changes(self):
        metadata = dict(environment='full-game-v2', config=resolve_config('configs/full_large.json'))
        config = resolve_config(overrides={'steps': 2000000, 'seed': 456}, resume_metadata=metadata)
        self.assertEqual(config['net_arch'], [512, 512])
        self.assertEqual(config['seed'], 456)
        for override in ({'gamma': 0.9}, {'observation_version': 'v1'}, {'net_arch': [256, 256]}):
            with self.assertRaisesRegex(ValueError, 'start a fresh run'):
                resolve_config(overrides=override, resume_metadata=metadata)

    def test_checkpoint_subdirectory_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = dict(environment='full-game-v2', config=copy.deepcopy(DEFAULTS))
            (root / 'run.json').write_text(json.dumps(data))
            self.assertEqual(read_metadata(root / 'checkpoints' / 'full_100_steps.zip'), data)


if __name__ == '__main__':
    unittest.main()
