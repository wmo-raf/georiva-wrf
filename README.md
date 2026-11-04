# wrfout-tidy

`wrfout_tidy.py` turns a raw WRF output file (`wrfout_d01_*`) into a compact,
CF-conventions NetCDF file: standard variable names and units, a regular
latitude/longitude axis, only the surface fields most users want, and optionally
daily or weekly values instead of every output step.

One wrfout file in, one NetCDF file out per requested product. The tool only
converts.

## Products

Choose with `--product`, a comma separated list (default `hourly`). Each product
is written to its own file, named by the run start time (UTC):

| `--product`        | File                         | Time axis                                                               |
|--------------------|------------------------------|-------------------------------------------------------------------------|
| `hourly` (default) | `wrf_hourly_YYYYMMDDTHHZ.nc` | every WRF output step                                                   |
| `daily`            | `wrf_daily_YYYYMMDDTHHZ.nc`  | one step per complete UTC day, stamped 00Z                              |
| `weekly`           | `wrf_weekly_YYYYMMDDTHHZ.nc` | one step covering the 7 complete UTC days from the first 00Z in the run |

Every file has the global attributes `wrfout_tidy_schema = 1` and
`wrfout_tidy_product` (`hourly`, `daily` or `weekly`).

If any requested product cannot be built, the run fails and **none** are written.

### Hourly variables

| Variable     | Meaning                                                             | Units        |
|--------------|---------------------------------------------------------------------|--------------|
| `t2`         | 2 m air temperature                                                 | K            |
| `d2m`        | 2 m dew point (capped at `t2`)                                      | K            |
| `msl`        | mean sea level pressure                                             | Pa           |
| `u10`, `v10` | 10 m wind, earth-relative                                           | m s-1        |
| `tp`         | precipitation accumulated from run start                            | mm           |
| `tcc`        | total cloud cover                                                   | 1 (0 to 1)   |
| `fg10`       | 10 m max wind since previous step (only with `nwp_diagnostics = 1`) | m s-1        |
| `psfc`       | surface pressure                                                    | Pa           |
| `pblh`       | planetary boundary layer height                                     | m            |
| `vegfra`     | vegetation fraction                                                 | % (0 to 100) |

`tp` is the sum of WRF's rain components (`RAINNC`, `RAINC`, `RAINSH`). With
`cu_physics = 0` and `shcu_physics = 0` it equals `RAINNC` and only `tp` is written.
If either setting is nonzero, `rainnc`, `rainc` and `rainsh` (mm, accumulated from
run start, those present in the wrfout) are written as well.

Renaming or removing a variable bumps `wrfout_tidy_schema` and is a breaking
release. Adding a variable does not.

### Daily and weekly variables

Each hourly variable is reduced to one value per period and renamed
`<variable>_<stat>`, with `cell_methods` set. There is no bare `t2` in a daily or
weekly file.

| Output                                                                        | Statistic over the period                                          |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| `t2_mean`, `t2_min`, `t2_max`                                                 | mean, minimum, maximum of the hourly steps                         |
| `d2m_mean`, `msl_mean`, `psfc_mean`, `tcc_mean`, `pblh_mean`, `vegfra_mean`   | mean                                                               |
| `u10_mean`, `v10_mean` (**daily only**)                                       | mean of the components (vector mean, not mean speed)               |
| `fg10_max`                                                                    | maximum (only with `nwp_diagnostics = 1`)                          |
| `tp_sum` (and `rainnc_sum`, `rainc_sum`, `rainsh_sum` when those are written) | total mm: accumulation at the end of the period minus at the start |

The weekly product leaves out the wind components: a 7 day vector mean of wind is
close to zero almost everywhere.

- **Complete UTC periods only.** A period is written when the 00Z step that opens
  it and the 00Z step that closes it are both present and the steps between are
  evenly spaced.
    - `daily`: incomplete days are skipped and named in `history`. A run with no
      complete day is an error.
    - `weekly`: the window starts at the first 00Z step in the run and covers 7 days,
      so it needs the 00Z step 168 hours later. A shorter run is an error. Steps
      after the window are ignored.
- Each period is stamped with its start and `time_bnds` gives `[start, end]`.
- Means and extremes use the steps from the opening 00Z up to, not including, the
  closing 00Z. `fg10` is a maximum since the previous step, so a period uses the
  steps after its opening 00Z up to and including its closing 00Z.
- On a Mercator grid, fields are resampled in latitude **before** aggregating, so
  `t2_max` is the true maximum of the delivered field.

### Grids

Memory stays at about 140 MB whatever the run length: input is read one step
at a time and the netCDF chunk cache is capped at 4 MB per variable.

On a Mercator grid the latitude rows are unevenly spaced, so every field is
linearly resampled along latitude onto a uniform axis (same extent, same number
of rows) and `history` says so. Regular lat-lon grids pass through unchanged.

Only **Mercator** and **regular lat-lon** grids are supported. Lambert and
polar-stereographic files are refused with a message naming the projection.

Output is byte-identical for the same input.

## Requirements

Python 3.6+, numpy, netCDF4 (`conda install -c conda-forge numpy netcdf4`).

## Usage

```
python3 wrfout_tidy.py WRFOUT_FILE [--product LIST] [--outdir DIR]
```

- Exactly one input file. Passing several is an error.
- `--product daily,weekly` writes both files from the one wrfout file.
- Do not use `frames_per_outfile` splitting: each wrfout file would convert to a
  single-step file with the same name. Keep one file per run.

Examples:

```
# a short run, every output step
python3 wrfout_tidy.py wrfout_d01_2026-10-27_00:00:00 --outdir OUT

# a week long run: daily values and the 7 day summary
python3 wrfout_tidy.py wrfout_d01_2026-10-27_00:00:00 --product daily,weekly --outdir OUT
```

## Running it

The tool takes one wrfout file, so pick the file in the shell. Output is
byte-identical for the same input and each file is moved into place only once
it is complete, so re-running on an unchanged file is harmless and a reader
never sees a half-written file.

**At the end of your run script:**

```
python3 wrfout_tidy.py $RUN/wrfout_d01_<run> --outdir $OUT
```

**Cron, converting the newest run on a schedule:**

```
30 * * * * f=$(ls -1 $RUN/wrfout_d01_* | tail -1) && python3 wrfout_tidy.py "$f" --outdir $OUT
```

**While a run is still writing** (a wrfout file grows as WRF runs; each pass
converts what is there so far, replacing the previous file of the same name):

```
*/20 * * * * python3 wrfout_tidy.py $RUN/wrfout_d01_<run> --outdir $OUT
```

While a run is in progress, `daily` and `weekly` only contain the periods that
are already complete, and `weekly` fails until 7 days exist.

What happens to `$OUT` afterwards (copying, publishing, cleaning up) is up to the
caller; the tool never moves or deletes anything but its own `.part` files.

## Tests

```
pip install pytest numpy netCDF4 && pytest
```
