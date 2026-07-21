"""
Per-cell precipitation reference for the 'simple' event extraction.

The simple method defines events and their quantile features (``*_q``) relative
to the per-cell distribution of the *loaded* period: the q98 detection threshold
and every percentile rank are estimated on whatever years are extracted. When
the test events are extracted over the test years alone, those references differ
from the training ones, so a given physical rainfall maps to a different
``p_1h_q`` at train and test time (a covariate shift that penalises the simple
method on independent data).

This module lets the training extraction persist, per cell, the quantities
needed to reproduce the training normalisation:
    - the q98 detection threshold, and
    - a compact CDF (value grid at fixed percentile levels) for each ranked
      column (raw precip for ``i_max_q``, each accumulation window for
      ``p_*_q``, and the daily API for ``api_q``).

The test extraction then detects and ranks events against these stored training
references instead of re-deriving them, so the feature space is consistent
across the train/test boundary.

The CDF is sampled on a tail-dense level grid because event features live in the
upper tail of the distribution (events are, by construction, q98 exceedances).
"""

import pickle
from pathlib import Path

import numpy as np

# Percentile levels at which each per-cell CDF is stored. Dense in the upper
# tail because the ranked event values (maxima, accumulations) fall there.
REF_LEVELS = np.unique(np.concatenate([
    np.linspace(0.0, 0.99, 100),
    np.linspace(0.99, 0.999, 30),
    np.linspace(0.999, 1.0, 20),
])).astype('float64')


def build_cdf(values):
    """
    Build a compact CDF for a 1-D sample: the value at each level of REF_LEVELS.

    Parameters
    ----------
    values: np.ndarray
        The sample (non-finite values are ignored).

    Returns
    -------
    np.ndarray|None
        The value grid (float32, same length as REF_LEVELS), or None if the
        sample has no finite value.
    """
    values = np.asarray(values, dtype='float64')
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return np.quantile(finite, REF_LEVELS).astype('float32')


def cdf_percentile(cdf_values, x):
    """
    Percentile of ``x`` within a stored CDF (inverse of build_cdf), matching the
    convention of ``Precipitation._pct_rank``: non-finite inputs map to NaN, and
    values below/above the reference map to 0/1.

    Parameters
    ----------
    cdf_values: np.ndarray
        The value grid returned by build_cdf (aligned with REF_LEVELS).
    x: float|np.ndarray
        The value(s) to rank.

    Returns
    -------
    np.ndarray
        The percentile(s) in [0, 1] (NaN where the input is not finite).
    """
    x = np.asarray(x, dtype='float64')
    out = np.full(x.shape, np.nan)
    if cdf_values is None:
        return out
    ok = np.isfinite(x)
    if ok.any():
        # np.interp needs a non-decreasing xp; the CDF grid is monotonic by
        # construction (quantiles of a sample).
        out[ok] = np.interp(x[ok], np.asarray(cdf_values, dtype='float64'),
                            REF_LEVELS, left=0.0, right=1.0)
    return out


def save_reference(reference, path):
    """
    Persist a per-cell reference mapping (cid -> reference dict) to a pickle file.

    Parameters
    ----------
    reference: dict
        Mapping cid -> {'q98': float, 'precip': cdf, 'windows': {name: cdf},
        'api': cdf}.
    path: str|Path
        The output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(reference, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_reference(path):
    """
    Load a per-cell reference mapping saved by save_reference.

    Parameters
    ----------
    path: str|Path
        The reference file path.

    Returns
    -------
    dict
        Mapping cid -> reference dict.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
