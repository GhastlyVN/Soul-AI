"""G3.1 metrics — per-episode then equal-weight aggregate.

No long-episode dominance; no invented threshold.
"""
import numpy as np


def compute_metrics(pred, target):
    diff = pred - target
    if not np.all(np.isfinite(diff)):
        raise ValueError('non-finite prediction or target values are not scoreable')
    result = {
        'mse': float(np.mean(diff**2)),
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff**2))),
        'bias': float(np.mean(diff)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('non-finite per-episode metric')
    return result


def aggregate_episode_metrics(ep_metrics):
    # ep_metrics: list of dicts per episode
    keys = ['mse', 'mae', 'rmse', 'bias']
    out = {}
    for k in keys:
        vals = [m[k] for m in ep_metrics]
        if not all(np.isfinite(value) for value in vals):
            raise ValueError(f'non-finite per-episode metric: {k}')
        out[k] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'dispersion': float(np.std(vals)),
            'count': len(vals),
        }
        if not all(np.isfinite(value) for value in out[k].values()):
            raise ValueError(f'non-finite aggregate metric: {k}')
    return out
