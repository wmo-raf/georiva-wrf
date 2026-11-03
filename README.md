# georiva-wrf

`wrf_to_georiva.py` converts **one** WRF output file (`wrfout_d01_*`) into **one**
GeoRiva NetCDF file. It does not send anything. Sending is done by
`georiva_push.py`, the push-feed sender served by your GeoRiva instance.

## What it produces

`wrf_YYYYMMDDTHHZ.nc`, named by the run start time (UTC), with global attribute
`georiva_wrf_schema = 1`.

| Variable | Meaning | Units |
|---|---|---|
| `t2` | 2 m air temperature | K |
| `d2m` | 2 m dew point (capped at `t2`) | K |
| `msl` | mean sea level pressure | Pa |
| `u10`, `v10` | 10 m wind, earth-relative | m s-1 |
| `tp` | precipitation accumulated from run start | mm |
| `tcc` | total cloud cover | 1 (0 to 1) |
| `fg10` | 10 m max wind since previous step (only with `nwp_diagnostics = 1`) | m s-1 |

Renaming or removing a variable breaks a configured catalog's Variable sources,
so it bumps `georiva_wrf_schema` and is a breaking release.

On a Mercator grid the latitude rows are unevenly spaced, so every field is
linearly resampled along latitude onto a uniform axis (same extent, same number
of rows) and `history` says so. Regular lat-lon grids pass through unchanged.

Only **Mercator** and **regular lat-lon** grids are supported. Lambert and
polar-stereographic files are refused with a message naming the projection.

## Requirements

Python 3.6+, numpy, netCDF4 (`conda install -c conda-forge numpy netcdf4`).

## Usage

```
python3 wrf_to_georiva.py WRFOUT_FILE [--outdir DIR] [--keep N]
```

- Exactly one input file. Passing several is an error.
- `--keep N` keeps the newest N converted files in `--outdir` (default 5, 0 = all).
  Retention of *sent* files belongs to `georiva_push.py send --dir --keep`,
  which only removes files it has sent. Note `--keep` here counts files whether
  or not they were sent, so after a long outage raise it (or use 0) to avoid
  deleting output that has not been sent yet.

## Sending to GeoRiva

Set `GEO` to the folder holding the scripts and `sender.conf` (the rclone block
from the push feed page), and `RUN` to the WRF run directory.

**1. Hook at the end of your run script** (send as soon as the run finishes):

```
python3 $GEO/wrf_to_georiva.py $RUN/wrfout_d01_<run> --outdir $GEO/OUT \
  && python3 $GEO/georiva_push.py send $GEO/OUT/wrf_<run>.nc --config $GEO/sender.conf
```

**2. Cron, converting the newest run on a schedule.** The script takes one file,
so pick it in the shell. Output is byte-identical for the same input, so the `.sent` marker makes repeat
runs skip an unchanged file. `--dir` leaves files modified in the last 120 s for
the next pass:

```
30 * * * * f=$(ls -1 $RUN/wrfout_d01_* | tail -1) && python3 $GEO/wrf_to_georiva.py "$f" --outdir $GEO/OUT && python3 $GEO/georiva_push.py send --dir $GEO/OUT --keep 5 --config $GEO/sender.conf
```

**3. Run in progress, re-sending as output grows** (a wrfout file is rewritten
while WRF runs; GeoRiva re-ingests the same key on resend):

```
*/20 * * * * python3 $GEO/wrf_to_georiva.py $RUN/wrfout_d01_<run> --outdir $GEO/OUT && python3 $GEO/georiva_push.py send $GEO/OUT/wrf_<run>.nc --config $GEO/sender.conf
```

Do not use `frames_per_outfile` splitting with this script: each wrfout file
would convert to a single-step file with the same name. Keep one file per run.

## Tests

```
pip install pytest numpy netCDF4 && pytest
```
