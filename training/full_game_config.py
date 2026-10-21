"""Versioned experiment settings shared by training, evaluation and playback."""
import copy
import json
import math
from pathlib import Path

LEGACY_REWARD = dict(survival=0.1, xp=0.1, kills=0.5, damage=0.1, death=10.0)
DEFAULTS = dict(
    steps=1_000_000, envs=4, duration=120, dex=7, seed=123,
    observation_version='v2', reward_weights=LEGACY_REWARD,
    net_arch=[256, 256], gamma=0.99, ent_coef=0.0,
    n_steps=128, batch_size=64, n_epochs=4, learning_rate=3e-4,
    target_kl=None, checkpoint_steps=100_000,
)
# Changing these while resuming would silently change the experiment.
RESUME_KEYS = (
    'duration', 'dex', 'observation_version', 'reward_weights', 'net_arch',
    'gamma', 'ent_coef', 'n_steps', 'batch_size', 'n_epochs', 'learning_rate',
    'target_kl',
)


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown experiment settings: {sorted(unknown)}')
    for key in ('steps', 'envs', 'duration', 'dex', 'n_steps', 'batch_size',
                'n_epochs', 'checkpoint_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 1 <= config['duration'] <= 3600:
        raise ValueError('duration must be 1..3600')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 1_000_000_000 - config['envs']:
        raise ValueError('Training seed range must stay below held-out seeds (1 billion)')
    if config['observation_version'] not in ('v1', 'v2-pickups', 'v2'):
        raise ValueError('Unknown observation_version')
    if not isinstance(config['net_arch'], list) or not config['net_arch'] or any(
        type(n) is not int or n < 1 for n in config['net_arch']
    ):
        raise ValueError('net_arch must be a nonempty list of positive integers')
    for key in ('gamma', 'ent_coef', 'learning_rate'):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(f'{key} must be finite')
    if not 0 < config['gamma'] < 1 or config['ent_coef'] < 0 or config['learning_rate'] <= 0:
        raise ValueError('Require 0 < gamma < 1, ent_coef >= 0 and learning_rate > 0')
    if config['target_kl'] is not None and (
        not isinstance(config['target_kl'], (int, float)) or
        not math.isfinite(config['target_kl']) or config['target_kl'] <= 0
    ):
        raise ValueError('target_kl must be null or finite and positive')
    weights = config['reward_weights']
    if not isinstance(weights, dict) or set(weights) != set(LEGACY_REWARD) or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()
    ):
        raise ValueError('reward_weights must contain finite nonnegative survival/xp/kills/damage/death weights')
    rollout = config['envs'] * config['n_steps']
    if config['batch_size'] < 2 or rollout < config['batch_size'] or rollout % config['batch_size']:
        raise ValueError('batch_size must be >= 2 and divide envs * n_steps')
    return config


def read_metadata(model_path):
    model_path = Path(model_path)
    path = model_path.parent / 'run.json'
    if not path.exists() and model_path.parent.name == 'checkpoints':
        path = model_path.parent.parent / 'run.json'
    metadata = json.loads(path.read_text())
    if metadata.get('environment') not in ('full-game-v1', 'full-game-v2'):
        raise ValueError('Expected a full-game checkpoint')
    return metadata


def config_from_metadata(metadata):
    config = copy.deepcopy(DEFAULTS)
    if 'config' in metadata:
        config.update(metadata['config'])
    else:
        # Old checkpoints used fixed PPO defaults and the original observation.
        config.update({k: metadata[k] for k in DEFAULTS if k in metadata})
        config['observation_version'] = 'v1'
    return validate_config(config)


def resolve_config(path=None, overrides=None, resume_metadata=None):
    base = config_from_metadata(resume_metadata) if resume_metadata else copy.deepcopy(DEFAULTS)
    config = copy.deepcopy(base)
    if path:
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError('Experiment config must be a JSON object')
        config.update(values)
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    validate_config(config)
    if resume_metadata:
        changed = [k for k in RESUME_KEYS if config[k] != base[k]]
        if changed:
            raise ValueError(f'Resume settings differ ({", ".join(changed)}); start a fresh run for this experiment')
    return config


def env_kwargs(metadata):
    config = config_from_metadata(metadata)
    return dict(duration=config['duration'], dex=config['dex'],
                observation_version=config['observation_version'],
                reward_weights=config['reward_weights'])
