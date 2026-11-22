"""Twelve-class building-resolved BFMI."""
# SPDX-License-Identifier: MIT


def bfmi(volumes):
    """Return normalized Shannon entropy for one grid's 12 class totals.

    Parameters
    ----------
    volumes : array-like of length 12
        Building-volume totals in this order:
        RR, II, PP, TT, RC, EE, IC, CC, PR, AA, MM, OO.
        OO contains combinations outside the preceding 11 classes. All totals
        must use the same volume unit. An absent class has volume zero;
        unknown quantities remain missing rather than being replaced by zero.

    Returns
    -------
    float
        BFMI = -sum(p * log2(p)) / log2(12), omitting zero shares.
        A single occupied class gives 0; equal shares across all 12 give 1.
        Missing, negative, non-finite or zero-total inputs return NaN.
    """
    import numpy as np

    v = np.asarray(volumes, dtype=float)
    if v.shape != (12,):
        raise ValueError("volumes must contain exactly 12 class totals")
    if not np.isfinite(v).all() or (v < 0).any() or not (v > 0).any():
        return float("nan")
    v = v / v.max()
    p = v[v > 0] / v.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(12))
