#!/usr/bin/env python3
"""
wrf_to_georiva.py

Convert a raw WRF output file (wrfout_d01_*) into a compact NetCDF file for
GeoRiva, and optionally upload it to an FTP server.

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

Upload (optional): the file is sent as <name>.tmp and renamed once complete,
so readers never see a half-written file. Older runs beyond --keep are removed.

Requirements: Python 3.6+, numpy, netCDF4. FTP uses the standard library.

Examples:
    # Convert only
    python3 wrf_to_georiva.py /path/to/wrfout_d01_2026-10-27_06:00:00 --outdir /data/georiva

    # Convert and upload (password from environment)
    export GEORIVA_FTP_PASSWORD='...'
    python3 wrf_to_georiva.py /path/to/wrfout_d01_* --outdir /data/georiva \\
        --ftp-host ftp.example.org --ftp-user kmd --ftp-dir wrf --keep 5
"""

import argparse
import datetime as dt
import ftplib
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
                 "U10", "V10", "RAINNC", "CLDFRA"]
OPTIONAL_RAIN = ["RAINC", "RAINSH"]
GUST_VAR = "WSPD10MAX"
FILE_PREFIX = "wrf_"


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
    ds = netCDF4.Dataset(src)
    ds.set_auto_mask(False)
    missing = [v for v in REQUIRED_VARS if v not in ds.variables]
    if missing:
        sys.exit("ERROR: {} is missing variables: {}".format(src, ", ".join(missing)))

    times = parse_times(ds)
    start = run_start(ds, times)
    steps = sorted(set((b - a).total_seconds() / 3600.0 for a, b in zip(times, times[1:])))
    log("Run start {}Z, {} steps, step spacing (h): {}".format(
        start.strftime("%Y-%m-%d %H:%M"), len(times), steps))
    if len(steps) > 1:
        log("WARNING: steps are not evenly spaced")

    lat, lon = grid_axes(ds)
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
    out.forecast_reference_time = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.history = "{} wrf_to_georiva.py from {}".format(
        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), os.path.basename(src))

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

        vars_out["t2"][i] = t2
        vars_out["d2m"][i] = dewpoint(t2, q2, psfc)
        vars_out["msl"][i] = sea_level_pressure(psfc, hgt, t2)
        vars_out["u10"][i] = u
        vars_out["v10"][i] = v
        vars_out["tp"][i] = sum(V[r][i].astype("f8") for r in rain_vars)
        vars_out["tcc"][i] = np.clip(V["CLDFRA"][i].max(axis=0), 0.0, 1.0)
        if has_gust:
            vars_out["fg10"][i] = V[GUST_VAR][i]

    out.close()
    ds.close()
    os.replace(tmp_path, final_path)
    log("Wrote {} ({:.1f} MB)".format(final_path, os.path.getsize(final_path) / 1e6))
    return final_path


def prune_local(outdir, keep):
    files = sorted(glob.glob(os.path.join(outdir, FILE_PREFIX + "*.nc")))
    for old in files[:-keep] if keep > 0 else []:
        os.remove(old)
        log("Removed local {}".format(os.path.basename(old)))


def upload(path, host, user, password, remote_dir, keep, port, tls, passive):
    name = os.path.basename(path)
    ftp = ftplib.FTP_TLS() if tls else ftplib.FTP()
    ftp.connect(host, port, timeout=120)
    ftp.login(user, password)
    if tls:
        ftp.prot_p()
    ftp.set_pasv(passive)
    if remote_dir:
        try:
            ftp.cwd(remote_dir)
        except ftplib.error_perm:
            ftp.mkd(remote_dir)
            ftp.cwd(remote_dir)

    tmp = name + ".tmp"
    log("Uploading {} to {}:{}/{}".format(name, host, remote_dir, tmp))
    with open(path, "rb") as fh:
        ftp.storbinary("STOR " + tmp, fh, blocksize=1024 * 1024)
    try:
        ftp.delete(name)  # replace an earlier upload of the same run
    except ftplib.error_perm:
        pass
    ftp.rename(tmp, name)
    log("Renamed {} -> {}".format(tmp, name))

    if keep > 0:
        runs = sorted(f for f in ftp.nlst() if f.startswith(FILE_PREFIX) and f.endswith(".nc"))
        for old in runs[:-keep]:
            ftp.delete(old)
            log("Removed remote {}".format(old))
    ftp.quit()


def main():
    p = argparse.ArgumentParser(description="Convert wrfout to a GeoRiva NetCDF and optionally upload via FTP.")
    p.add_argument("wrfout", nargs="+", help="wrfout_d01_* file(s); the latest by name is used")
    p.add_argument("--outdir", default=".", help="local folder for output files (default: current)")
    p.add_argument("--keep", type=int, default=5, help="runs to keep locally and on the FTP (default: 5, 0 = keep all)")
    p.add_argument("--ftp-host", help="FTP host; omit to skip upload")
    p.add_argument("--ftp-port", type=int, default=21)
    p.add_argument("--ftp-user", default=os.environ.get("GEORIVA_FTP_USER"))
    p.add_argument("--ftp-password", default=os.environ.get("GEORIVA_FTP_PASSWORD"),
                   help="prefer the GEORIVA_FTP_PASSWORD environment variable")
    p.add_argument("--ftp-dir", default="wrf", help="remote folder (default: wrf)")
    p.add_argument("--ftp-tls", action="store_true", help="use explicit FTPS")
    p.add_argument("--ftp-active", action="store_true", help="use active mode instead of passive")
    a = p.parse_args()

    src = sorted(a.wrfout)[-1]
    if not os.path.isfile(src):
        sys.exit("ERROR: {} not found".format(src))
    log("Input {}".format(src))

    path = convert(src, a.outdir)
    prune_local(a.outdir, a.keep)

    if a.ftp_host:
        if not a.ftp_user or not a.ftp_password:
            sys.exit("ERROR: FTP user/password missing (use --ftp-user and GEORIVA_FTP_PASSWORD)")
        last = None
        for attempt in range(1, 4):
            try:
                upload(path, a.ftp_host, a.ftp_user, a.ftp_password, a.ftp_dir,
                       a.keep, a.ftp_port, a.ftp_tls, not a.ftp_active)
                break
            except (ftplib.all_errors) as exc:
                last = exc
                log("Upload attempt {} failed: {}".format(attempt, exc))
        else:
            sys.exit("ERROR: upload failed after 3 attempts: {}".format(last))
    else:
        log("No --ftp-host given, skipping upload")
    log("Done")


if __name__ == "__main__":
    main()
