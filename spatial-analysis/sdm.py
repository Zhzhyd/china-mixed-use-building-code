"""M5c: heteroskedastic GS2SLS spatial Durbin model with spatial error.

Run: python sdm.py final_sample.csv queen_weights.npz
Requires Python 3.10+, numpy, pandas, scipy and spreg==1.8.2.
Tested with numpy 1.25.0, pandas 2.2.3 and scipy 1.14.1.

Input CSV: the final 91,019 grids in weight-matrix row order, with grid_id,
city_code, ring_b, BFMI_raw, carbin_avg and the 19 CONTROLS listed below.
carbin_avg is annual territorial emissions in t C per grid. BFMI_raw is the
normalized 12-class index. ring_b is the existing shared 5-km band code 1..8
(band 8: >=35 km). Controls are the prepared M5c fields, not raw raster values:
GDP_log1p is already log-transformed; roadL and transit distance are in km;
EAD_idx is already normalized; pop_avg2 is the existing population proxy;
slope_std is the existing composite terrain covariate. Keep their input scales.

Input NPZ: scipy sparse, final within-city Queen W in exactly the CSV row order,
with nonzero rows summing to one and 216 isolated zero rows retained.
The fixed Legendre domain [0, 0.9087302537621148] is not the observed range.
The estimator performs the specified single GS2SLS update (max_iter=1).
Printed roots are zeros of the homogeneous-neighbourhood approximate total
marginal association, searched on the fixed Legendre domain as in M5c.
"""
# SPDX-License-Identifier: MIT

import argparse
import json

import numpy as np
import pandas as pd
from numpy.polynomial import Legendre
from numpy.polynomial.legendre import legvander
from scipy import linalg, sparse
import spreg
from spreg.error_sp_het import BaseGM_Combo_Het
from spreg.twosls import BaseTSLS


DOMAIN = (0.0, 0.9087302537621148)
CONTROLS = [
    "Vol_R", "Vol_I", "Vol_E", "Vol_P", "Vol_C", "Vol_M", "Vol_A", "Vol_T",
    "pop_avg2", "viirs_sd", "GDP_log1p", "roadL", "DistNearestTransit_km_poi",
    "UrbanDevelopmentAge", "dem_avg", "slope_avg", "slope_std", "EAD_idx", "JSD",
]


def _within_scale(x, groups):
    """Frozen numerical scaling: std(ddof=0) after city/ring absorption."""
    r = x.copy()
    aggregators = [sparse.csr_matrix((np.ones(len(g)), (g, np.arange(len(g))))) for g in groups]
    counts = [np.bincount(g) for g in groups]
    scale = np.maximum(x.std(axis=0), 1.0)
    for _ in range(2000):
        previous = r.copy()
        for g, a, count in zip(groups, aggregators, counts):
            r -= ((a @ r) / count[:, None])[g]
        if np.max(np.abs(r - previous) / scale) < 1e-11:
            sd = r.std(axis=0, ddof=0)
            if (sd <= np.finfo(float).eps).any():
                raise ValueError("A control is constant after fixed-effect absorption")
            return sd
    raise RuntimeError("Fixed-effect scaling did not converge")


def fit_m5c(frame, weights):
    """Return (spreg model in numerical fitting units, compact result dict).

    frame is the prepared final-sample pandas DataFrame; weights is its aligned
    row-standardized scipy sparse Queen matrix. Rows and values are retained.
    """
    if spreg.__version__ != "1.8.2":
        raise RuntimeError("This specification uses spreg==1.8.2")
    required = ["grid_id", "city_code", "ring_b", "BFMI_raw", "carbin_avg"] + CONTROLS
    if len(frame) != 91019 or frame.city_code.nunique() != 344 or frame.grid_id.duplicated().any():
        raise ValueError("Expected the final 91,019 unique grids across 344 cities")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Inputs must be complete and finite; no rows are dropped")
    b = frame.BFMI_raw.to_numpy(dtype=float)
    if ((b < 0) | (b > 1)).any() or not frame.ring_b.isin(range(1, 9)).all():
        raise ValueError("Invalid normalized BFMI or shared distance-band code")
    w = sparse.csr_matrix(weights, dtype=float)
    rows, columns = w.nonzero()
    totals = np.asarray(w.sum(axis=1)).ravel()
    city = frame.city_code.to_numpy()
    if (w.shape != (len(frame), len(frame)) or not np.isfinite(w.data).all()
            or (w.data < 0).any() or np.any(w.diagonal() != 0)
            or np.count_nonzero(totals == 0) != 216
            or not np.allclose(totals[totals != 0], 1)
            or np.any(city[rows] != city[columns])):
        raise ValueError("Expected aligned within-city Queen W with normalized rows and 216 isolates")

    basis = legvander(2 * (b - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) - 1, 3)[:, 1:]
    x = np.column_stack([frame[CONTROLS].to_numpy(dtype=float), basis])
    cities, ci = np.unique(city, return_inverse=True)
    rings, ri = np.unique(frame.ring_b.to_numpy(dtype=int), return_inverse=True)
    n, k = x.shape
    fe = sparse.csr_matrix((np.ones(2 * n), (np.tile(np.arange(n), 2),
                           np.r_[ci, len(cities) + ri])), shape=(n, len(cities) + len(rings)))
    gram = (fe.T @ fe).toarray()
    _, _, pivots = linalg.qr(gram, pivoting=True)
    rank = np.linalg.matrix_rank(gram)
    f = fe[:, np.sort(pivots[:rank])].toarray()
    f /= np.sqrt(np.mean(f * f, axis=0))
    sd = _within_scale(x, [ci, ri])
    ys = float(frame.carbin_avg.std(ddof=0))
    if ys <= 0:
        raise ValueError("Emissions must have positive variance")
    y, xs = frame.carbin_avg.to_numpy(dtype=float)[:, None] / ys, x / sd
    wx, wy = w @ xs, w @ y
    exog, instruments = np.column_stack([xs, wx, f]), w @ wx
    initial = BaseTSLS(y, exog, yend=wy, q=instruments)
    if abs(float(initial.betas[-1, 0])) >= 0.9:
        raise RuntimeError("Initial rho reaches the estimator's clipping boundary")
    del initial
    model = BaseGM_Combo_Het(y, exog, yend=wy, q=instruments, w=w,
                            max_iter=1, epsilon=1e-5, step1c=False,
                            inv_method="power_exp", hard_bound=True)
    coefficients = model.betas.ravel()
    rho, lam = map(float, coefficients[-2:])
    if (not np.isfinite(coefficients).all() or not np.isfinite(model.vm).all()
            or abs(rho) >= 0.89999999 or abs(lam) >= 0.99):
        raise RuntimeError("Non-finite, boundary or clipped estimate")

    # Restore fitting scales; lambda is not a multiplier of the mean association.
    combined = (coefficients[:k] + coefficients[k:2*k]) * ys / sd / (1 - rho)
    marginal = Legendre(np.r_[0.0, combined[-3:]], domain=DOMAIN).deriv()
    roots = sorted(float(r.real) for r in marginal.roots()
                   if abs(r.imag) < 1e-9 and DOMAIN[0] <= r.real <= DOMAIN[1])
    return model, {
        "N": n, "cities": len(cities), "rho": rho, "lambda": lam,
        "BFMI_observed_range": [float(b.min()), float(b.max())],
        "zero_crossings": roots,
        "roots_within_observed_range": [bool(b.min() <= r <= b.max()) for r in roots],
        "negative_between_two_roots": bool(marginal(np.mean(roots)) < 0) if len(roots) == 2 else None,
        "MA_at_0_5_kt_C_per_year_grid_per_BFMI": float(marginal(0.5) / 1000),
        "estimator_stop": model.iter_stop,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the final M5c sample and report BFMI zero crossings")
    parser.add_argument("sample_csv")
    parser.add_argument("weights_npz")
    args = parser.parse_args()
    _, result = fit_m5c(pd.read_csv(args.sample_csv, float_precision="round_trip"),
                             sparse.load_npz(args.weights_npz))
    print(json.dumps(result, indent=2))
