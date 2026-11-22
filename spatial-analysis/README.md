# Spatial analysis

- `sdm.py`: M5c spatial Durbin model with spatial error, estimated by heteroskedastic GS2SLS. Includes city and distance-band fixed effects and reports BFMI marginal-association zero crossings.
- `bfmi.py`: computes normalized 12-class BFMI from one grid's class-level building-volume totals. Input order and missing-value handling are described in the function docstring.

## Run

Tested with Python 3.10.11:

```bash
pip install numpy==1.25.0 pandas==2.2.3 scipy==1.14.1 spreg==1.8.2
python sdm.py final_sample.csv queen_weights.npz
```

Provide the prepared final sample of 91,019 grids across 344 cities and its row-standardized, within-city Queen matrix in the same row order, retaining isolated zero rows. Required CSV fields and their scales are documented in `sdm.py`.

The model uses the fixed cubic Legendre transformation and one GS2SLS update. With the study inputs, the approximate total marginal association has zero crossings at 0.196019 and 0.634771.

License: MIT.
