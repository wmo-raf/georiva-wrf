#!/usr/bin/env python3
"""
wrf_to_georiva.py

Convert a raw WRF output file (wrfout_d01_*) into a compact NetCDF file for
GeoRiva. Exactly one wrfout file in, one NetCDF file out.

Output: one file per model run, named by the run start time in UTC:
    wrf_YYYYMMDDTHHZ.nc

Variables written (native units, GeoRiva handles conversions):
    t2     2 m air temperature                       K
    d2m    2 m dew point temperature                 K     (capped at t2)
    msl    mean sea level pressure                   Pa
    u10    10 m eastward wind                        m s-1 (earth-relative)
    v10    10 m northward wind                       m s-1 (earth-relative)
    tp     total precipitation, accumulated from
           the start of the run                      mm
    tcc    total cloud cover (maximum overlap)       1     (0 to 1)
    fg10   10 m max wind speed since previous step   m s-1 (only if WRF was
           run with nwp_diagnostics = 1)
    psfc   surface pressure                          Pa
    pblh   boundary layer height                     m
    vegfra vegetation fraction                       %     (0 to 100)
    rainnc, rainc, rainsh  grid-scale, convective and shallow convective
           precipitation, accumulated from run start  mm
           (only if CU_PHYSICS or SHCU_PHYSICS is nonzero; tp is always
           written and is their sum)

Global attribute georiva_wrf_schema = 1 marks the variable set above. Renaming
or removing a variable breaks configured GeoRiva catalogs, so it bumps the
schema and is a breaking release. Adding a variable does not.

Sending is not done here: use georiva_push.py (see README.md).

Requirements: Python 3.6+, numpy, netCDF4.

Example:
    python3 wrf_to_georiva.py /path/to/wrfout_d01_2026-10-27_06:00:00 --outdir /data/georiva
"""

import argparse
import datetime as dt
import glob
import os
import sys

try:
    import numpy as np
    import netCDF4
except ImportError as exc:  # pragma: no cover
    sys.exit("ERROR: missing Python package ({}). Install with: "
             "conda install -c conda-forge numpy netcdf4".format(exc))

REQUIRED_VARS = ["Times", "XLAT", "XLONG", "T2", "Q2", "PSFC", "HGT",
                 "U10", "V10", "RAINNC", "CLDFRA", "PBLH", "VEGFRA"]
OPTIONAL_RAIN = ["RAINC", "RAINSH"]
GUST_VAR = "WSPD10MAX"
FILE_PREFIX = "wrf_"
SCHEMA_VERSION = 1

# netCDF keeps a chunk cache per output variable (64 MB each by default), which
# grows to several hundred MB when every variable is written one timestep at a
# time. Each step is a single chunk written once, so a small cache costs nothing.
CHUNK_CACHE_BYTES = 4 * 1024 * 1024
# WRF MAP_PROJ codes: 1 Lambert, 2 polar stereographic, 3 Mercator, 6 lat-lon
SUPPORTED_PROJ = {3: "Mercator", 6: "regular lat-lon"}
UNSUPPORTED_PROJ = {1: "Lambert conformal", 2: "polar stereographic"}


def log(msg):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print("[{}] {}".format(stamp, msg), flush=True)


def parse_times(ds):
    raw = ds.variables["Times"][:]
    times = []
    for row in raw:
        s = b"".join(row.tolist()).decode() if row.dtype.kind == "S" else "".join(row)
        times.append(dt.datetime.strptime(s.strip(), "%Y-%m-%d_%H:%M:%S"))
    return times


def run_start(ds, times):
    attr = getattr(ds, "SIMULATION_START_DATE", None)
    if attr:
        try:
            return dt.datetime.strptime(attr.strip(), "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            pass
    return times[0]


def check_projection(ds):
    """Refuse grids that cannot be written as regular 1D lat/lon axes."""
    code = getattr(ds, "MAP_PROJ", None)
    if code is not None and int(code) not in SUPPORTED_PROJ:
        name = UNSUPPORTED_PROJ.get(int(code),
                                    getattr(ds, "MAP_PROJ_CHAR", "unknown"))
        sys.exit("ERROR: grid projection is {} (MAP_PROJ={}). Only Mercator "
                 "and regular lat-lon grids are supported.".format(name, int(code)))


def grid_axes(ds):
    """1D lat/lon from WRF's 2D coordinates. Only valid for grids whose rows
    have constant latitude and columns constant longitude (Mercator, lat-lon)."""
    xlat = ds.variables["XLAT"][0, :, :]
    xlon = ds.variables["XLONG"][0, :, :]
    lat = xlat[:, 0].astype("f8")
    lon = xlon[0, :].astype("f8")
    lat_dev = float(np.abs(xlat - lat[:, None]).max())
    lon_dev = float(np.abs(xlon - lon[None, :]).max())
    if lat_dev > 1e-3 or lon_dev > 1e-3:
        proj = getattr(ds, "MAP_PROJ_CHAR", "unknown")
        sys.exit("ERROR: grid is not separable into 1D lat/lon (projection: {}, "
                 "max deviation lat {:.4f}, lon {:.4f}). This script supports "
                 "Mercator and lat-lon WRF grids.".format(proj, lat_dev, lon_dev))
    return lat, lon


def lat_resampler(lat, tol=1e-6):
    """Plan to move rows onto a uniform latitude axis, or None if already uniform.

    Mercator rows are evenly spaced in projected y, not in latitude. Returns
    (new_lat, i0, i1, w): new row k = old[i0[k]] * (1 - w[k]) + old[i1[k]] * w[k],
    the same weights for every variable and step.
    """
    ny = len(lat)
    new_lat = np.linspace(lat[0], lat[-1], ny)
    if ny < 3 or np.abs(new_lat - lat).max() <= tol:
        return None
    sign = 1.0 if lat[-1] > lat[0] else -1.0
    i0 = np.clip(np.searchsorted(sign * lat, sign * new_lat, side="right") - 1, 0, ny - 2)
    w = (new_lat - lat[i0]) / (lat[i0 + 1] - lat[i0])
    return new_lat, i0, i0 + 1, np.clip(w, 0.0, 1.0)


def apply_resampler(plan, field):
    """Resample ``field`` (rows on axis -2) with a plan from lat_resampler."""
    if plan is None:
        return field
    _, i0, i1, w = plan
    w = w[:, None]
    return field[..., i0, :] * (1.0 - w) + field[..., i1, :] * w


def dewpoint(t2, q2, psfc):
    """2 m dew point (K) from mixing ratio and surface pressure, capped at t2."""
    q = np.maximum(q2, 1e-10)
    e = q * psfc / (0.622 + q)                     # vapour pressure, Pa
    x = np.log(e / 611.2)
    td = 243.5 * x / (17.67 - x) + 273.15          # Magnus, back to K
    return np.minimum(td, t2)


def sea_level_pressure(psfc, hgt, t2):
    """Reduce surface pressure to sea level with a standard lapse rate (Pa)."""
    return psfc * (1.0 - 0.0065 * hgt / (t2 + 0.0065 * hgt)) ** (-5.257)


def convert(src, outdir):
    netCDF4.set_chunk_cache(CHUNK_CACHE_BYTES, 101, 1.0)
    ds = netCDF4.Dataset(src)
    ds.set_auto_mask(False)
    missing = [v for v in REQUIRED_VARS if v not in ds.variables]
    if missing:
        sys.exit("ERROR: {} is missing variables: {}".format(src, ", ".join(missing)))

    check_projection(ds)
    times = parse_times(ds)
    start = run_start(ds, times)
    steps = sorted(set((b - a).total_seconds() / 3600.0 for a, b in zip(times, times[1:])))
    log("Run start {}Z, {} steps, step spacing (h): {}".format(
        start.strftime("%Y-%m-%d %H:%M"), len(times), steps))
    if len(steps) > 1:
        log("WARNING: steps are not evenly spaced")

    lat, lon = grid_axes(ds)
    plan = lat_resampler(lat)
    if plan is not None:
        log("Mercator rows are not uniform in latitude: resampling onto a uniform axis")
        lat = plan[0]
    rain_vars = ["RAINNC"] + [v for v in OPTIONAL_RAIN if v in ds.variables]
    has_gust = GUST_VAR in ds.variables
    has_rot = "COSALPHA" in ds.variables and "SINALPHA" in ds.variables
    if not has_gust:
        log("NOTE: {} not in file, gust (fg10) will not be written. "
            "Enable nwp_diagnostics = 1 in namelist.input to add it.".format(GUST_VAR))

    name = "{}{}.nc".format(FILE_PREFIX, start.strftime("%Y%m%dT%HZ"))
    os.makedirs(outdir, exist_ok=True)
    final_path = os.path.join(outdir, name)
    tmp_path = final_path + ".part"

    ny, nx = len(lat), len(lon)
    out = netCDF4.Dataset(tmp_path, "w", format="NETCDF4")
    out.Conventions = "CF-1.8"
    out.title = "WRF forecast subset for GeoRiva"
    out.source = (getattr(ds, "TITLE", "WRF").strip()
                  + ", " + getattr(ds, "MAP_PROJ_CHAR", "") + " grid")
    out.georiva_wrf_schema = np.int32(SCHEMA_VERSION)
    out.forecast_reference_time = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    # No wall-clock time here: the same input must give byte-identical output, so
    # georiva_push.py's checksum marker skips an unchanged file on a repeat run.
    out.history = "wrf_to_georiva.py from {}".format(os.path.basename(src))
    if plan is not None:
        out.history += ("; latitude rows linearly resampled onto a uniform axis "
                        "over the same extent and row count")

    out.createDimension("time", None)
    out.createDimension("lat", ny)
    out.createDimension("lon", nx)

    vt = out.createVariable("time", "f8", ("time",))
    vt.standard_name = "time"
    vt.units = "hours since 1970-01-01 00:00:00"
    vt.calendar = "standard"
    vt.axis = "T"
    vt[:] = netCDF4.date2num(times, vt.units, vt.calendar)

    vref = out.createVariable("forecast_reference_time", "f8")
    vref.standard_name = "forecast_reference_time"
    vref.units = vt.units
    vref.calendar = vt.calendar
    vref.assignValue(netCDF4.date2num(start, vt.units, vt.calendar))

    vlat = out.createVariable("lat", "f8", ("lat",))
    vlat.standard_name, vlat.long_name, vlat.units, vlat.axis = "latitude", "latitude", "degrees_north", "Y"
    vlat[:] = lat
    vlon = out.createVariable("lon", "f8", ("lon",))
    vlon.standard_name, vlon.long_name, vlon.units, vlon.axis = "longitude", "longitude", "degrees_east", "X"
    vlon[:] = lon

    def make(vname, std, long, units):
        v = out.createVariable(vname, "f4", ("time", "lat", "lon"), zlib=True,
                               complevel=4, chunksizes=(1, ny, nx), fill_value=np.float32(1e20))
        if std:
            v.standard_name = std
        v.long_name = long
        v.units = units
        return v

    vars_out = {
        "t2": make("t2", "air_temperature", "2 m air temperature", "K"),
        "d2m": make("d2m", "dew_point_temperature", "2 m dew point temperature", "K"),
        "msl": make("msl", "air_pressure_at_mean_sea_level", "mean sea level pressure", "Pa"),
        "u10": make("u10", "eastward_wind", "10 m eastward wind", "m s-1"),
        "v10": make("v10", "northward_wind", "10 m northward wind", "m s-1"),
        "tp": make("tp", None, "total precipitation accumulated from start of run", "mm"),
        "tcc": make("tcc", "cloud_area_fraction", "total cloud cover (maximum overlap)", "1"),
    }
    if has_gust:
        vars_out["fg10"] = make("fg10", "wind_speed_of_gust",
                                "10 m maximum wind speed since previous output step", "m s-1")

    # Direct copies: written exactly as WRF stores them, apart from the latitude
    # resampling every field gets. Output name -> wrfout name.
    direct = {
        "psfc": ("PSFC", make("psfc", "surface_air_pressure", "surface pressure", "Pa")),
        "pblh": ("PBLH", make("pblh", "atmosphere_boundary_layer_thickness",
                              "planetary boundary layer height", "m")),
        "vegfra": ("VEGFRA", make("vegfra", "vegetation_area_fraction",
                                  "vegetation fraction", "%")),
    }
    # Rain components only when a cumulus scheme can produce a non-trivial split;
    # otherwise tp already equals RAINNC. A missing attribute counts as 0.
    if any(int(getattr(ds, a, 0) or 0) != 0 for a in ("CU_PHYSICS", "SHCU_PHYSICS")):
        parts = {"rainnc": ("RAINNC", "grid-scale"), "rainc": ("RAINC", "convective"),
                 "rainsh": ("RAINSH", "shallow convective")}
        for name_out, (wname, label) in parts.items():
            if wname in ds.variables:
                direct[name_out] = (wname, make(
                    name_out, None,
                    "{} precipitation accumulated from start of run".format(label), "mm"))

    V = ds.variables
    hgt = V["HGT"][0].astype("f8")
    if has_rot:
        cosa = V["COSALPHA"][0].astype("f8")
        sina = V["SINALPHA"][0].astype("f8")

    for i in range(len(times)):
        t2 = V["T2"][i].astype("f8")
        q2 = V["Q2"][i].astype("f8")
        psfc = V["PSFC"][i].astype("f8")
        u, v = V["U10"][i].astype("f8"), V["V10"][i].astype("f8")
        if has_rot:
            u, v = u * cosa - v * sina, v * cosa + u * sina

        R = lambda f: apply_resampler(plan, f)
        vars_out["t2"][i] = R(t2)
        vars_out["d2m"][i] = R(dewpoint(t2, q2, psfc))
        vars_out["msl"][i] = R(sea_level_pressure(psfc, hgt, t2))
        vars_out["u10"][i] = R(u)
        vars_out["v10"][i] = R(v)
        vars_out["tp"][i] = R(sum(V[r][i].astype("f8") for r in rain_vars))
        vars_out["tcc"][i] = R(np.clip(V["CLDFRA"][i].max(axis=0), 0.0, 1.0))
        if has_gust:
            vars_out["fg10"][i] = R(V[GUST_VAR][i].astype("f8"))
        for src_name, var_out in direct.values():
            var_out[i] = R(V[src_name][i].astype("f8"))

    out.close()
    ds.close()
    os.replace(tmp_path, final_path)
    log("Wrote {} ({:.1f} MB)".format(final_path, os.path.getsize(final_path) / 1e6))
    return final_path


def prune_local(outdir, keep, protect=None):
    files = sorted(glob.glob(os.path.join(outdir, FILE_PREFIX + "*.nc")))
    for old in files[:-keep] if keep > 0 else []:
        if protect and os.path.abspath(old) == os.path.abspath(protect):
            continue
        os.remove(old)
        if os.path.exists(old + ".sent"):  # georiva_push.py marker
            os.remove(old + ".sent")
        log("Removed local {}".format(os.path.basename(old)))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert one wrfout file to a GeoRiva NetCDF file.")
    p.add_argument("wrfout", nargs="*", help="exactly one wrfout_d01_* file")
    p.add_argument("--outdir", default=".", help="local folder for output files (default: current)")
    p.add_argument("--keep", type=int, default=5,
                   help="converted files to keep in --outdir (default: 5, 0 = keep all)")
    a = p.parse_args(argv)

    if len(a.wrfout) != 1:
        p.error("expected exactly one wrfout file, got {}".format(len(a.wrfout)))
    src = a.wrfout[0]
    if not os.path.isfile(src):
        sys.exit("ERROR: {} not found".format(src))
    log("Input {}".format(src))

    path = convert(src, a.outdir)
    prune_local(a.outdir, a.keep, protect=path)
    log("Done")


if __name__ == "__main__":
    main()
