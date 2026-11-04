#!/usr/bin/env python3
"""
wrfout_tidy.py

Author: Erick Otenyo (WMO Regional Office for Africa) Github @erick-otenyo

Convert a raw WRF output file (wrfout_d01_*) into a compact, CF-conventions
NetCDF file with standard variable names on a regular latitude/longitude axis,
optionally reduced to daily or weekly values. One wrfout file in; one NetCDF
file out per requested product.

Products (--product, comma separated, default hourly). Each is written to
wrf_<product>_YYYYMMDDTHHZ.nc, named by the run start time in UTC:
    hourly   every WRF output step
    daily    one step per complete UTC day, stamped 00Z, with time_bnds
    weekly   one step for the 7 complete UTC days from the first 00Z in the
             run, stamped with the start of the window, with time_bnds

Hourly variables (native units):
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

Daily and weekly variables are named <variable>_<stat> and carry cell_methods:
    t2_mean t2_min t2_max, d2m_mean, msl_mean, tcc_mean, psfc_mean, pblh_mean,
    vegfra_mean, fg10_max, and tp_sum (total mm over the period, plus
    rainnc_sum, rainc_sum, rainsh_sum when present). The daily product also has
    u10_mean and v10_mean (vector means); the weekly product does not.
A period is written only when the 00Z step that opens it and the 00Z step that
closes it are both present and the steps between are evenly spaced. If a
requested product cannot be built, nothing is written. Fields are resampled in
latitude before they are aggregated.

Global attribute wrfout_tidy_schema = 1 marks the variable set above; renaming
or removing a variable bumps it and is a breaking release. Adding a variable
does not. wrfout_tidy_product records which product a file is.

Requirements: Python 3.6+, numpy, netCDF4.

Example:
    python3 wrfout_tidy.py /path/to/wrfout_d01_2026-10-27_00:00:00 \\
        --product daily,weekly --outdir /data/out
"""

import argparse
import datetime as dt
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
PRODUCTS = ("hourly", "daily", "weekly")
FILE_PREFIX = "wrf_"
SCHEMA_VERSION = 1
WEEK_DAYS = 7

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




# Statistics each aggregated product gives each hourly variable. Output names
# are <variable>_<stat>. "sum" is the difference of an accumulated field between
# the end and the start of the period; every other stat is taken over the hourly
# steps of the period.
DAILY_STATS = {
    "t2": ["mean", "min", "max"],
    "d2m": ["mean"],
    "msl": ["mean"],
    "u10": ["mean"],
    "v10": ["mean"],
    "tp": ["sum"],
    "tcc": ["mean"],
    "fg10": ["max"],
    "psfc": ["mean"],
    "pblh": ["mean"],
    "vegfra": ["mean"],
    "rainnc": ["sum"],
    "rainc": ["sum"],
    "rainsh": ["sum"],
}
# A 7 day vector mean of wind is close to zero almost everywhere, so the weekly
# product leaves the components out.
WEEKLY_STATS = {k: v for k, v in DAILY_STATS.items() if k not in ("u10", "v10")}
PRODUCT_STATS = {"daily": DAILY_STATS, "weekly": WEEKLY_STATS}
STAT_WORD = {"mean": "mean", "min": "minimum", "max": "maximum"}
CELL_METHODS = {"mean": "time: mean", "min": "time: minimum",
                "max": "time: maximum", "sum": "time: sum"}
ACCUMULATED = ("tp", "rainnc", "rainc", "rainsh")


def _evenly_spaced(times, i0, i1):
    gaps = set(round((b - a).total_seconds())
               for a, b in zip(times[i0:i1], times[i0 + 1:i1 + 1]))
    return len(gaps) == 1


def complete_days(times):
    """UTC days fully covered by the run, as (start_index, end_index) pairs.

    A day is complete when the 00Z step that opens it and the 00Z step that
    closes it are both present and every step between them is evenly spaced.
    Returns (days, skipped) where skipped lists the dates of days left out.
    """
    index = {t: i for i, t in enumerate(times)}
    first = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
    last = times[-1].replace(hour=0, minute=0, second=0, microsecond=0)
    days, skipped = [], []
    d = first
    while d <= last:
        nxt = d + dt.timedelta(days=1)
        i0, i1 = index.get(d), index.get(nxt)
        if i0 is not None and i1 is not None and i1 > i0 and _evenly_spaced(times, i0, i1):
            days.append((i0, i1))
        else:
            skipped.append(d.strftime("%Y-%m-%d"))
        d = nxt
    return days, skipped


def weekly_window(times, ndays=WEEK_DAYS):
    """(start_index, end_index) of the ndays complete UTC days that begin at the
    first 00Z step in the run, or None if the run does not cover them."""
    index = {t: i for i, t in enumerate(times)}
    midnights = [t for t in times if t.hour == 0 and t.minute == 0 and t.second == 0]
    if not midnights:
        return None
    t0 = midnights[0]
    i0 = index[t0]
    i1 = index.get(t0 + dt.timedelta(days=ndays))
    if i1 is None or not _evenly_spaced(times, i0, i1):
        return None
    return i0, i1


def parse_products(text):
    """'hourly,daily' -> ['hourly', 'daily'] (order kept, duplicates dropped)."""
    out = []
    for p in (x.strip() for x in text.split(",")):
        if p not in PRODUCTS:
            raise argparse.ArgumentTypeError(
                "unknown product {!r} (choose from {})".format(p, ", ".join(PRODUCTS)))
        if p not in out:
            out.append(p)
    if not out:
        raise argparse.ArgumentTypeError("no product given")
    return out


def convert(src, outdir, products=("hourly",)):
    """Convert one wrfout file; returns the list of files written, in the order
    of ``products``. Everything is validated before any file is created, and the
    outputs are only moved into place once all of them are written."""
    products = list(parse_products(products) if isinstance(products, str) else products)
    for p in products:
        if p not in PRODUCTS:
            sys.exit("ERROR: unknown product {!r}".format(p))
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

    # Periods each product covers, checked before anything is written.
    windows, skipped = {}, []
    span = "steps run from {}Z to {}Z".format(times[0].strftime("%Y-%m-%d %H:%M"),
                                              times[-1].strftime("%Y-%m-%d %H:%M"))
    if "daily" in products:
        days, skipped = complete_days(times)
        if not days:
            sys.exit("ERROR: no complete UTC day in {} ({}). The daily product needs "
                     "the 00Z step that opens a day and the 00Z step that closes "
                     "it.".format(src, span))
        windows["daily"] = days
        if skipped:
            log("Daily: skipping incomplete days: {}".format(", ".join(skipped)))
    if "weekly" in products:
        win = weekly_window(times)
        if win is None:
            sys.exit("ERROR: {} does not cover {} complete UTC days from its first "
                     "00Z step ({}). The weekly product needs the 00Z step that "
                     "opens the window and the one {} days later.".format(
                         src, WEEK_DAYS, span, WEEK_DAYS))
        windows["weekly"] = [win]
        if win[1] < len(times) - 1:
            log("Weekly: run is longer than {} days, later steps are not used".format(WEEK_DAYS))

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

    # Every hourly variable: name -> (standard_name, long_name, units).
    defs = {
        "t2": ("air_temperature", "2 m air temperature", "K"),
        "d2m": ("dew_point_temperature", "2 m dew point temperature", "K"),
        "msl": ("air_pressure_at_mean_sea_level", "mean sea level pressure", "Pa"),
        "u10": ("eastward_wind", "10 m eastward wind", "m s-1"),
        "v10": ("northward_wind", "10 m northward wind", "m s-1"),
        "tp": (None, "total precipitation accumulated from start of run", "mm"),
        "tcc": ("cloud_area_fraction", "total cloud cover (maximum overlap)", "1"),
    }
    if has_gust:
        defs["fg10"] = ("wind_speed_of_gust",
                        "10 m maximum wind speed since previous output step", "m s-1")
    defs["psfc"] = ("surface_air_pressure", "surface pressure", "Pa")
    defs["pblh"] = ("atmosphere_boundary_layer_thickness",
                    "planetary boundary layer height", "m")
    defs["vegfra"] = ("vegetation_area_fraction", "vegetation fraction", "%")
    # Rain components only when a cumulus scheme can produce a non-trivial split;
    # otherwise tp already equals RAINNC. A missing attribute counts as 0.
    rain_parts = {}
    if any(int(getattr(ds, a, 0) or 0) != 0 for a in ("CU_PHYSICS", "SHCU_PHYSICS")):
        parts = {"rainnc": ("RAINNC", "grid-scale"), "rainc": ("RAINC", "convective"),
                 "rainsh": ("RAINSH", "shallow convective")}
        for name_out, (wname, label) in parts.items():
            if wname in ds.variables:
                rain_parts[name_out] = wname
                defs[name_out] = (None, "{} precipitation accumulated from start "
                                        "of run".format(label), "mm")
    direct = {"psfc": "PSFC", "pblh": "PBLH", "vegfra": "VEGFRA"}

    V = ds.variables
    hgt = V["HGT"][0].astype("f8")
    if has_rot:
        cosa = V["COSALPHA"][0].astype("f8")
        sina = V["SINALPHA"][0].astype("f8")

    def fields(i):
        """All hourly fields of step i, already on the output latitude axis."""
        t2 = V["T2"][i].astype("f8")
        q2 = V["Q2"][i].astype("f8")
        psfc = V["PSFC"][i].astype("f8")
        u, v = V["U10"][i].astype("f8"), V["V10"][i].astype("f8")
        if has_rot:
            u, v = u * cosa - v * sina, v * cosa + u * sina
        f = {
            "t2": t2,
            "d2m": dewpoint(t2, q2, psfc),
            "msl": sea_level_pressure(psfc, hgt, t2),
            "u10": u,
            "v10": v,
            "tp": sum(V[r][i].astype("f8") for r in rain_vars),
            "tcc": np.clip(V["CLDFRA"][i].max(axis=0), 0.0, 1.0),
        }
        if has_gust:
            f["fg10"] = V[GUST_VAR][i].astype("f8")
        for n, wname in direct.items():
            f[n] = V[wname][i].astype("f8")
        for n, wname in rain_parts.items():
            f[n] = V[wname][i].astype("f8")
        return {n: apply_resampler(plan, a) for n, a in f.items()}

    ny, nx = len(lat), len(lon)

    def write_product(product, tmp_path):
        out = netCDF4.Dataset(tmp_path, "w", format="NETCDF4")
        out.Conventions = "CF-1.8"
        out.title = "WRF output subset"
        out.source = (getattr(ds, "TITLE", "WRF").strip()
                      + ", " + getattr(ds, "MAP_PROJ_CHAR", "") + " grid")
        out.wrfout_tidy_schema = np.int32(SCHEMA_VERSION)
        out.wrfout_tidy_product = product
        out.forecast_reference_time = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        # No wall-clock time here: the same input must give byte-identical
        # output, so repeat runs on the same input give the same file.
        out.history = "wrfout_tidy.py from {}".format(os.path.basename(src))
        if plan is not None:
            out.history += ("; latitude rows linearly resampled onto a uniform axis "
                            "over the same extent and row count")
        if product == "daily":
            out.history += ("; daily aggregates over complete UTC days, each "
                            "stamped with the start of its day")
            if skipped:
                out.history += "; incomplete days skipped: {}".format(", ".join(skipped))
        elif product == "weekly":
            i0, i1 = windows["weekly"][0]
            out.history += ("; aggregates over the {} complete UTC days from {}Z to "
                            "{}Z, stamped with the start of the window".format(
                                WEEK_DAYS, times[i0].strftime("%Y-%m-%d %H:%M"),
                                times[i1].strftime("%Y-%m-%d %H:%M")))

        out.createDimension("time", None)
        out.createDimension("lat", ny)
        out.createDimension("lon", nx)

        vt = out.createVariable("time", "f8", ("time",))
        vt.standard_name = "time"
        vt.units = "hours since 1970-01-01 00:00:00"
        vt.calendar = "standard"
        vt.axis = "T"
        vtb = None
        if product == "hourly":
            vt[:] = netCDF4.date2num(times, vt.units, vt.calendar)
        else:
            out.createDimension("nv", 2)
            vt.bounds = "time_bnds"
            vtb = out.createVariable("time_bnds", "f8", ("time", "nv"))
            vtb.units = vt.units
            vtb.calendar = vt.calendar

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

        def make(vname, std, long, units, cell_methods=None):
            v = out.createVariable(vname, "f4", ("time", "lat", "lon"), zlib=True,
                                   complevel=4, chunksizes=(1, ny, nx),
                                   fill_value=np.float32(1e20))
            if std:
                v.standard_name = std
            v.long_name = long
            v.units = units
            if cell_methods:
                v.cell_methods = cell_methods
            return v

        if product == "hourly":
            vars_out = {n: make(n, std, long, units) for n, (std, long, units) in defs.items()}
            for i in range(len(times)):
                f = fields(i)
                for n, var_out in vars_out.items():
                    var_out[i] = f[n]
        else:
            table = PRODUCT_STATS[product]
            stats = [(base, s) for base in defs for s in table.get(base, [])]
            vars_out = {}
            for base, s in stats:
                std, long, units = defs[base]
                if s == "sum":
                    std = None
                    if base == "tp":
                        long = "{} precipitation total".format(product)
                    else:
                        long = "{} total {}".format(product, long.replace(" accumulated from start of run", ""))
                else:
                    long = "{} {} of {}".format(product, STAT_WORD[s], long)
                vars_out[(base, s)] = make("{}_{}".format(base, s), std, long, units,
                                           CELL_METHODS[s])
            for k, (i0, i1) in enumerate(windows[product]):
                acc, n_inst = {}, 0
                first_acc = last_acc = None
                for i in range(i0, i1 + 1):
                    f = fields(i)
                    if i == i0:
                        first_acc = {b: f[b] for b in ACCUMULATED if b in f}
                    if i == i1:
                        last_acc = {b: f[b] for b in ACCUMULATED if b in f}
                    if i < i1:
                        n_inst += 1
                    for base, s in stats:
                        if s == "sum":
                            continue
                        # fg10 is a max since the previous step, so a period owns
                        # the steps after its opening 00Z up to and including its
                        # closing 00Z. Everything else is sampled at the steps
                        # from the opening 00Z up to, not including, the closing.
                        if base == "fg10":
                            if i == i0:
                                continue
                        elif i == i1:
                            continue
                        key = (base, s)
                        a = f[base]
                        if key not in acc:
                            acc[key] = a.copy()
                        elif s == "min":
                            acc[key] = np.minimum(acc[key], a)
                        elif s == "max":
                            acc[key] = np.maximum(acc[key], a)
                        else:
                            acc[key] = acc[key] + a
                for base, s in stats:
                    if s == "sum":
                        vars_out[(base, s)][k] = last_acc[base] - first_acc[base]
                    elif s == "mean":
                        vars_out[(base, s)][k] = acc[(base, s)] / n_inst
                    else:
                        vars_out[(base, s)][k] = acc[(base, s)]
                vt[k] = netCDF4.date2num(times[i0], vt.units, vt.calendar)
                vtb[k, :] = netCDF4.date2num([times[i0], times[i1]], vt.units, vt.calendar)
        out.close()

    os.makedirs(outdir, exist_ok=True)
    pending = []  # (tmp_path, final_path)
    try:
        for product in products:
            name = "{}{}_{}.nc".format(FILE_PREFIX, product, start.strftime("%Y%m%dT%HZ"))
            final_path = os.path.join(outdir, name)
            tmp_path = final_path + ".part"
            pending.append((tmp_path, final_path))
            write_product(product, tmp_path)
    except BaseException:
        for tmp_path, _ in pending:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        raise
    finally:
        ds.close()

    written = []
    for tmp_path, final_path in pending:
        os.replace(tmp_path, final_path)
        log("Wrote {} ({:.1f} MB)".format(final_path, os.path.getsize(final_path) / 1e6))
        written.append(final_path)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert one wrfout file to compact, CF-conventions NetCDF.")
    p.add_argument("wrfout", nargs="*", help="exactly one wrfout_d01_* file")
    p.add_argument("--product", type=parse_products, default=["hourly"],
                   metavar="LIST",
                   help="comma separated products to write: hourly (every output "
                        "step), daily (one step per complete UTC day), weekly (one "
                        "step for {} days from the first 00Z). Default: hourly. "
                        "If one cannot be built, none are written.".format(WEEK_DAYS))
    p.add_argument("--outdir", default=".", help="local folder for output files (default: current)")
    a = p.parse_args(argv)

    if len(a.wrfout) != 1:
        p.error("expected exactly one wrfout file, got {}".format(len(a.wrfout)))
    src = a.wrfout[0]
    if not os.path.isfile(src):
        sys.exit("ERROR: {} not found".format(src))
    log("Input {}".format(src))

    convert(src, a.outdir, a.product)
    log("Done")


if __name__ == "__main__":
    main()
