"""Identify historical and nighttime-light-assisted international city centres.

Inputs
------
1. A city CSV with city_id, city_name and boundary_file. Each boundary_file
   contains the existing study boundary for one city, with a defined CRS.
   Relative paths are resolved against the CSV directory.
2. GUB_Global_1990.shp: historical built-up polygons in WGS84 (EPSG:4326).
   Original feature IDs identify patches consistently across city queries.
3. Optional 2020 nighttime lights: SRUNet_NPP_VIIRS_V2_Like_2020.tif,
   WGS84, north-up, with nonnegative radiance values in band 1.
   Product reference: https://doi.org/10.1038/s41597-024-04228-6

Method
------
Repair geometries and use a city-local Lambert azimuthal equal-area projection.
Assign each historical patch to the city with the largest overlap fraction
(intersection area / full patch area), provided that fraction is at least 0.5.
Ties are resolved by ascending string city_id. Clip assigned patches to that
city's study boundary and take the centroid of their union in the local
projection. The centroid is retained even when it falls outside the union.
A city with no selected patch uses its study-boundary centroid and is flagged.

With nighttime lights, also calculate C3_NTL. Its search area is the intersection
of a 3-km buffer around C0, a 1-km buffer around the historical union and the
study boundary. Weight pixel-intersection centroids by their projected area
times log1p(radiance capped at the search area's 99th percentile). Move halfway
from C0 toward that weighted location, capped at 2 km; retain C0 if the proposal
is outside the search area or no positive weight is available.

Dependencies: numpy, pandas, geopandas, shapely>=2, pyproj, pyogrio, rasterio.

Example
-------
python identify_city_centres.py cities.csv GUB_Global_1990.shp --output centres.geojson
python identify_city_centres.py cities.csv GUB_Global_1990.shp --ntl SRUNet_NPP_VIIRS_V2_Like_2020.tif --output centres.geojson

The output contains C0 and, when --ntl is supplied, C3_NTL as separate WGS84
points for every city, with coordinates, matching flags and shift distances.
"""
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import Window, from_bounds
import shapely
from shapely.ops import transform


def _local_projection(boundary):
    """Use the study-boundary lon/lat centroid as the LAEA projection origin."""
    origin = boundary.centroid
    crs = (f"+proj=laea +lon_0={origin.x} +lat_0={origin.y} "
           "+datum=WGS84 +units=m")
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    return forward, inverse


def _read_boundary(path):
    frame = gpd.read_file(path, engine="pyogrio")
    if frame.crs is None:
        raise ValueError(f"Missing boundary CRS: {path}")
    frame = frame.to_crs(4326)
    if frame.empty or frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(f"Missing or empty boundary geometry: {path}")
    boundary = shapely.union_all(shapely.make_valid(frame.geometry))
    if boundary.is_empty or boundary.area <= 0:
        raise ValueError(f"Boundary has no polygonal area: {path}")
    return boundary


def _ntl_pixels(src, geometry):
    """Read intersecting native pixels; zero radiance remains a valid value."""
    window = from_bounds(*geometry.bounds, src.transform)
    col = max(0, int(np.floor(window.col_off)))
    row = max(0, int(np.floor(window.row_off)))
    stop_col = min(src.width, int(np.ceil(window.col_off + window.width)))
    stop_row = min(src.height, int(np.ceil(window.row_off + window.height)))
    if stop_col <= col or stop_row <= row:
        return [], np.array([])
    values = src.read(1, window=Window(col, row, stop_col-col, stop_row-row)).astype(float)
    rr, cc = np.indices(values.shape)
    affine = src.transform
    x = affine.c + (cc + col) * affine.a
    y = affine.f + (rr + row) * affine.e
    pixels = shapely.box(x, y + affine.e, x + affine.a, y).ravel()
    values = values.ravel()
    valid = np.isfinite(values) & (values >= 0)
    # In this product, zero is also the background code: keep zero radiance.
    if src.nodata is not None and src.nodata != 0:
        valid &= values != src.nodata
    pixels, values = pixels[valid], values[valid]
    hit = shapely.intersects(pixels, geometry)
    return shapely.intersection(pixels[hit], geometry), values[hit]


def _ntl_centre(src, anchor, historical_union, boundary, forward, inverse):
    """Apply the fixed nighttime-light correction in city-local metres."""
    search = anchor.buffer(3000).intersection(historical_union.buffer(1000)).intersection(boundary)
    if search.is_empty:
        return anchor, 0, "empty_search_area"
    pieces, values = _ntl_pixels(src, transform(inverse.transform, search))
    if not len(values):
        return anchor, 0, "no_valid_pixels"
    cap = np.percentile(values, 99)
    weights, coordinates = [], []
    for piece, value in zip(pieces, values):
        projected = transform(forward.transform, piece)
        weights.append(np.log1p(min(value, cap)) * projected.area)
        coordinates.append([projected.centroid.x, projected.centroid.y])
    weights = np.asarray(weights)
    if weights.sum() <= 0:
        return anchor, len(values), "no_positive_weight"
    target = np.average(coordinates, axis=0, weights=weights)
    shift = 0.5 * (target - np.array([anchor.x, anchor.y]))
    length = np.linalg.norm(shift)
    if length > 2000:
        shift *= 2000 / length
    proposed = shapely.Point(anchor.x + shift[0], anchor.y + shift[1])
    if search.covers(proposed):
        return proposed, len(values), "accepted"
    return anchor, len(values), "outside_search_area_anchor_retained"


def identify_centres(city_csv, historical_file, ntl_file=None):
    """Return (centres, patch_matches), adding C3_NTL when ntl_file is supplied.

    centres is an EPSG:4326 GeoDataFrame. patch_matches records every positive
    city/patch intersection, its overlap fraction and whether it was selected.
    Inputs are read-only; this function writes no files.
    """
    city_csv, historical_file = Path(city_csv), Path(historical_file)
    cities = pd.read_csv(city_csv, dtype={"city_id": str})
    fields = ["city_id", "city_name", "boundary_file"]
    if not set(fields).issubset(cities.columns):
        raise ValueError(f"City CSV requires columns: {', '.join(fields)}")
    if cities.empty or cities[fields].isna().any().any() or cities.city_id.duplicated().any():
        raise ValueError("City metadata must be complete with unique city_id values")
    historical_crs = pyogrio.read_info(historical_file)["crs"]
    if historical_crs is None or CRS.from_user_input(historical_crs).to_epsg() != 4326:
        raise ValueError("Historical GUB input must use EPSG:4326")

    boundaries, candidates = {}, []
    for city in cities.itertuples(index=False):
        path = Path(city.boundary_file)
        if not path.is_absolute():
            path = city_csv.parent / path
        boundary = _read_boundary(path)
        forward, inverse = _local_projection(boundary)
        projected_boundary = transform(forward.transform, boundary)
        boundaries[city.city_id] = (projected_boundary, forward, inverse)
        patches = gpd.read_file(historical_file, bbox=boundary.bounds,
                                engine="pyogrio", fid_as_index=True)
        for patch_id, row in patches.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                raise ValueError(f"Missing historical geometry at feature {patch_id}")
            patch = transform(forward.transform, shapely.make_valid(row.geometry))
            if patch.area <= 0:
                raise ValueError(f"Historical feature {patch_id} has no polygonal area")
            intersection = patch.intersection(projected_boundary)
            if intersection.area > 0:
                candidates.append({
                    "city_id": city.city_id,
                    "patch_id": int(patch_id),
                    "patch_area_km2": patch.area / 1e6,
                    "intersection_km2": intersection.area / 1e6,
                    "overlap_fraction": intersection.area / patch.area,
                    # Preserve the geographic intermediate used in the analysis.
                    "geometry": transform(inverse.transform, intersection),
                })

    columns = ["city_id", "patch_id", "patch_area_km2", "intersection_km2",
               "overlap_fraction", "geometry"]
    matches = pd.DataFrame(candidates, columns=columns)
    matches["selected"] = False
    for _, group in matches.groupby("patch_id"):
        best = group.sort_values(["overlap_fraction", "city_id"],
                                  ascending=[False, True]).index[0]
        if matches.loc[best, "overlap_fraction"] >= 0.5:
            matches.loc[best, "selected"] = True

    records = []
    raster = nullcontext(None) if ntl_file is None else rasterio.open(ntl_file)
    with raster as src:
        if src is not None and (src.crs is None or src.crs.to_epsg() != 4326 or
                                src.transform.b != 0 or src.transform.d != 0 or
                                src.transform.a <= 0 or src.transform.e >= 0):
            raise ValueError("Nighttime lights must be a north-up EPSG:4326 raster")
        for city in cities.itertuples(index=False):
            boundary, forward, inverse = boundaries[city.city_id]
            selected = matches.loc[(matches.city_id == city.city_id) & matches.selected]
            selected = selected.sort_values(["intersection_km2", "patch_id"],
                                            ascending=[False, True])
            fallback = selected.empty
            patches = ([boundary] if fallback else
                       [transform(forward.transform, p) for p in selected.geometry])
            historical_union = shapely.union_all(patches)
            anchor = historical_union.centroid
            points = [("C0", anchor, 0, "historical_union_centroid" if not fallback
                       else "study_boundary_fallback")]
            if src is not None:
                point, count, status = _ntl_centre(src, anchor, historical_union,
                                                  boundary, forward, inverse)
                points.append(("C3_NTL", point, count, status))
            for version, projected, count, status in points:
                centre = transform(inverse.transform, projected)
                records.append({
                    "city_id": city.city_id,
                    "city_name": city.city_name,
                    "centre_id": f"{city.city_id}_{version}",
                    "version": version,
                    "longitude": centre.x,
                    "latitude": centre.y,
                    "historical_source_year": None if fallback else 1990,
                    "historical_patch_N": len(selected),
                    "current_boundary_fallback": fallback,
                    "outside_historical_patch": None if fallback else not historical_union.covers(projected),
                    "shift_from_C0_m": projected.distance(anchor),
                    "ntl_year": 2020 if version == "C3_NTL" else None,
                    "ntl_pixel_N": count,
                    "status": status,
                    "geometry": centre,
                })
    centres = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
    return centres, matches.drop(columns="geometry")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cities_csv", type=Path)
    parser.add_argument("historical_polygons", type=Path)
    parser.add_argument("--ntl", type=Path, help="2020 nighttime-light raster; also output C3_NTL")
    parser.add_argument("--output", type=Path, required=True, help="New output GeoJSON file")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".geojson":
        parser.error("--output must end in .geojson")
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    centres, _ = identify_centres(args.cities_csv, args.historical_polygons, args.ntl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    centres.to_file(args.output, driver="GeoJSON", engine="pyogrio")
    print(f"Saved {len(centres)} centre records for {centres.city_id.nunique()} cities to {args.output}")


if __name__ == "__main__":
    main()
