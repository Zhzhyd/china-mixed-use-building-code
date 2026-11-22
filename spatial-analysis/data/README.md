# Analysis grids

`analysis_grids.geojson` contains the 91,019 final analysis grid polygons
across 344 cities, in WGS84 longitude/latitude coordinates (EPSG:4326).

Each feature has 24 properties: the grid and city identifiers, distance-band
code, 12-class BFMI, annual carbon emissions, and the 19 prepared control
variables listed in [`sdm.py`](../sdm.py). Emissions (`carbin_avg`) are in
t C yr⁻¹ grid⁻¹. Model values and feature order match the final-sample CSV
and its aligned Queen matrix; polygon coordinates retain their source precision.
