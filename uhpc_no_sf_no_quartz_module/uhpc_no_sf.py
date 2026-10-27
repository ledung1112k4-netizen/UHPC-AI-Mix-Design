"""Constrained UHPC search; uses ONLY the caller's trained-model prediction function.
No fabricated model or predictions. All quantities in kg/m3.
"""
from dataclasses import dataclass
from typing import Callable, Mapping
import numpy as np

MATERIALS = ('cement', 'fly_ash', 'ggbfs', 'quartz_powder', 'quartz_sand', 'water', 'superplasticizer', 'steel_fiber', 'silica_fume')

@dataclass(frozen=True)
class Bounds:
    low: float
    high: float


def optimize_no_sf_no_quartz(predict: Callable[[Mapping[str, float]], float],
                   bounds: Mapping[str, Bounds], target_mpa: float = 120.0,
                   batch_kg: float = 50.0, n_trials: int = 3000,
                   seed: int = 42, validate=None):
    """Random constrained search. predict must call the EXISTING model pipeline.

    validate(mix) can reject candidates outside verified material/domain limits.
    Returns None when no feasible solution is found. This does not establish
    laboratory strength or guarantee model generalization.
    """
    if not callable(predict):
        raise TypeError('A real trained-model prediction callback is required')
    if target_mpa <= 0 or batch_kg <= 0 or n_trials < 1:
        raise ValueError('Target, batch and trials must be positive')
    missing = set(MATERIALS) - set(bounds)
    if missing:
        raise ValueError(f'Missing bounds: {sorted(missing)}')
    for excluded in ('silica_fume', 'quartz_powder'):
        if not (bounds[excluded].low == bounds[excluded].high == 0):
            raise ValueError(f'{excluded} bounds must be exactly 0')
    for name in MATERIALS:
        b = bounds[name]
        if not (np.isfinite(b.low) and np.isfinite(b.high) and 0 <= b.low <= b.high):
            raise ValueError(f'Invalid bounds for {name}')
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n_trials):
        mix = {k: float(rng.uniform(bounds[k].low, bounds[k].high)) for k in MATERIALS}
        mix['silica_fume'] = 0.0
        mix['quartz_powder'] = 0.0
        if validate is not None and not validate(mix):
            continue
        strength = float(predict(mix))
        if not np.isfinite(strength):
            continue
        if strength >= target_mpa:
            candidates.append((strength, mix))
    if not candidates:
        return None
    strength, mix = min(candidates, key=lambda x: x[0] - target_mpa)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError('Zero mass candidate')
    scale = batch_kg / total
    return {
        'predicted_mpa': strength,
        'mix_kg_m3': mix,
        'batch_kg': {k: v * scale for k, v in mix.items()},
        'batch_total_kg': batch_kg,
        'sf_kg_m3': 0.0,
        'sf_batch_kg': 0.0,
        'quartz_powder_kg_m3': 0.0,
        'quartz_powder_batch_kg': 0.0,
        'note': 'Model estimate only; requires experimental verification.',
    }

# Backward-compatible alias for existing integrations.
optimize_no_sf = optimize_no_sf_no_quartz
