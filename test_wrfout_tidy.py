import numpy as np
import netCDF4
import pytest

import wrfout_tidy as w

NY, NX, NT = 4, 5, 2


def conv(src, outdir, product="hourly"):
    """Convert with a single product and return its path."""
    return w.convert(src, outdir, [product])[0]


def make_wrfout(path, map_proj=3, gust=False, skew=False, lat_axis=None, t2_of_lat=False,
                rainc=True, cu=0, shcu=0, times=None, hourly_fn=None):
    """hourly_fn(i) -> dict of overrides {"T2": scalar, "RAINNC": scalar, ...}"""
    if times is None:
        times = ["2026-10-27_06:00:00", "2026-10-27_07:00:00"]
    NT = len(times)
    ds = netCDF4.Dataset(str(path), "w")
    ds.CU_PHYSICS = cu
    ds.SHCU_PHYSICS = shcu
    ds.createDimension("Time", None)
    ds.createDimension("DateStrLen", 19)
    ds.createDimension("south_north", NY)
    ds.createDimension("west_east", NX)
    ds.createDimension("bottom_top", 3)
    ds.MAP_PROJ = np.int32(map_proj)
    ds.MAP_PROJ_CHAR = {1: "Lambert Conformal", 3: "Mercator"}.get(map_proj, "other")
    ds.SIMULATION_START_DATE = times[0]
    ds.TITLE = "OUTPUT FROM WRF"
    tvar = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
    for i, s in enumerate(times):
        tvar[i, :] = netCDF4.stringtochar(np.array([s], "S19"))[0]
    ax = np.linspace(-1, 1, NY) if lat_axis is None else np.asarray(lat_axis, float)
    lat = ax[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * np.linspace(36, 37, NX)[None, :]
    if skew:
        lon = lon + lat * 0.5
    def var(name, dims, val):
        v = ds.createVariable(name, "f4", dims)
        v[:] = val
        if hourly_fn is not None:
            for i in range(NT):
                o = hourly_fn(i)
                if name in o:
                    v[i] = o[name]
    d2 = ("Time", "south_north", "west_east")
    var("XLAT", d2, np.broadcast_to(lat, (NT, NY, NX)))
    var("XLONG", d2, np.broadcast_to(lon, (NT, NY, NX)))
    var("T2", d2, np.broadcast_to(250.0 + lat, (NT, NY, NX)) if t2_of_lat
        else np.full((NT, NY, NX), 300.0))
    var("Q2", d2, np.full((NT, NY, NX), 0.012))
    var("PSFC", d2, np.full((NT, NY, NX), 90000.0))
    var("HGT", d2, np.full((NT, NY, NX), 1000.0))
    var("U10", d2, np.full((NT, NY, NX), 3.0))
    var("V10", d2, np.full((NT, NY, NX), 4.0))
    var("RAINNC", d2, np.full((NT, NY, NX), 2.0))
    if rainc:
        var("RAINC", d2, np.full((NT, NY, NX), 1.0))
        var("RAINSH", d2, np.full((NT, NY, NX), 0.5))
    var("PBLH", d2, np.full((NT, NY, NX), 850.0))
    var("VEGFRA", d2, np.full((NT, NY, NX), 62.5))
    var("CLDFRA", ("Time", "bottom_top", "south_north", "west_east"),
        np.full((NT, 3, NY, NX), 0.4))
    if gust:
        var("WSPD10MAX", d2, np.full((NT, NY, NX), 9.0))
    ds.close()
    return str(path)


def test_output_variables_units_and_schema(tmp_path):
    src = make_wrfout(tmp_path / "wrfout_d01", gust=True)
    out = conv(src, str(tmp_path / "out"))
    assert out.endswith("wrf_hourly_20261027T06Z.nc")
    with netCDF4.Dataset(out) as ds:
        assert ds.wrfout_tidy_schema == 1
        units = {n: ds[n].units for n in ["t2", "d2m", "msl", "u10", "v10", "tp", "tcc", "fg10"]}
        assert units == {"t2": "K", "d2m": "K", "msl": "Pa", "u10": "m s-1",
                         "v10": "m s-1", "tp": "mm", "tcc": "1", "fg10": "m s-1"}
        assert ds["t2"].shape == (NT, NY, NX)
        assert float(ds["tp"][0, 0, 0]) == pytest.approx(3.5)
        assert float(ds["tcc"][0, 0, 0]) == pytest.approx(0.4)
        assert float(ds["d2m"][0, 0, 0]) <= 300.0


def test_direct_copies(tmp_path):
    src = make_wrfout(tmp_path / "f")
    out = conv(src, str(tmp_path / "o"))
    expected = {"psfc": (90000.0, "Pa"), "pblh": (850.0, "m"), "vegfra": (62.5, "%")}
    with netCDF4.Dataset(out) as ds:
        assert ds.wrfout_tidy_schema == 1
        for name, (value, units) in expected.items():
            assert ds[name].units == units
            assert ds[name].shape == (NT, NY, NX)
            assert np.allclose(ds[name][:], value)
        # with no cumulus physics, rain is only written as tp
        assert not {"rainc", "rainnc", "rainsh"} & set(ds.variables)


def test_tp_without_rainc(tmp_path):
    out = conv(make_wrfout(tmp_path / "f", rainc=False), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert float(ds["tp"][0, 0, 0]) == pytest.approx(2.0)


def test_direct_copies_resampled_with_mercator_rows(tmp_path):
    src = make_wrfout(tmp_path / "f", lat_axis=MERC_LAT)
    out = conv(src, str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        # constant fields stay constant through the row resampling
        assert np.allclose(ds["pblh"][:], 850.0)
        assert ds["pblh"].shape == ds["t2"].shape


def test_rain_components_with_cumulus(tmp_path):
    out = conv(make_wrfout(tmp_path / "f", cu=1), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert np.allclose(ds["rainnc"][:], 2.0)
        assert np.allclose(ds["rainc"][:], 1.0)
        assert np.allclose(ds["rainsh"][:], 0.5)
        assert np.allclose(ds["tp"][:], 3.5)


def test_rain_components_with_shallow_cumulus_only(tmp_path):
    out = conv(make_wrfout(tmp_path / "f", shcu=1), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert "rainsh" in ds.variables and "rainnc" in ds.variables


def test_rain_components_missing_wrfout_var(tmp_path):
    out = conv(make_wrfout(tmp_path / "f", cu=1, rainc=False), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert "rainnc" in ds.variables
        assert "rainc" not in ds.variables and "rainsh" not in ds.variables


def test_rain_components_resampled(tmp_path):
    out = conv(make_wrfout(tmp_path / "f", lat_axis=MERC_LAT, cu=1), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert ds["rainc"].shape == ds["t2"].shape
        assert np.allclose(ds["rainc"][:], 1.0)


def test_gust_optional(tmp_path):
    out = conv(make_wrfout(tmp_path / "f"), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert "fg10" not in ds.variables


def test_more_than_one_file_is_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        w.main(["a", "b", "--outdir", str(tmp_path)])
    assert e.value.code != 0
    assert "exactly one" in capsys.readouterr().err


def test_lambert_refused_naming_projection(tmp_path):
    src = make_wrfout(tmp_path / "f", map_proj=1)
    with pytest.raises(SystemExit) as e:
        conv(src, str(tmp_path / "o"))
    msg = str(e.value)
    assert "Lambert" in msg and "Mercator" in msg and "lat-lon" in msg


def test_no_ftp_flags():
    with pytest.raises(SystemExit):
        w.main(["x", "--ftp-host", "h"])




def test_same_input_gives_identical_bytes(tmp_path):
    import hashlib
    src = make_wrfout(tmp_path / "f")
    a = conv(src, str(tmp_path / "a"))
    b = conv(src, str(tmp_path / "b"))
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h(a) == h(b)


MERC_LAT = np.degrees(np.arctan(np.sinh(np.linspace(-0.6, 0.2, NY))))


def test_mercator_rows_resampled_to_uniform_axis(tmp_path):
    src = make_wrfout(tmp_path / "f", lat_axis=MERC_LAT, t2_of_lat=True)
    out = conv(src, str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        lat = ds["lat"][:]
        assert np.ptp(np.diff(lat)) < 1e-9
        assert lat[0] == pytest.approx(MERC_LAT[0]) and lat[-1] == pytest.approx(MERC_LAT[-1])
        assert len(lat) == NY
        # T2 is linear in lat, so linear interpolation reproduces it exactly
        assert np.allclose(ds["t2"][0, :, 0], 250.0 + lat, atol=1e-3)
        assert "resampl" in ds.history.lower()
        # end rows coincide with original rows: unchanged
        assert float(ds["t2"][0, 0, 0]) == pytest.approx(250.0 + MERC_LAT[0], abs=1e-3)


def test_regular_grid_passes_through(tmp_path):
    src = make_wrfout(tmp_path / "f", t2_of_lat=True)
    out = conv(src, str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert np.allclose(ds["lat"][:], np.linspace(-1, 1, NY))
        assert "resampl" not in ds.history.lower()


def test_descending_latitude_resampled(tmp_path):
    src = make_wrfout(tmp_path / "f", lat_axis=MERC_LAT[::-1], t2_of_lat=True)
    out = conv(src, str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        lat = ds["lat"][:]
        assert np.ptp(np.diff(lat)) < 1e-9 and lat[0] > lat[-1]
        assert np.allclose(ds["t2"][0, :, 0], 250.0 + lat, atol=1e-3)


# ---- daily product -------------------------------------------------------

def hours(start, n):
    import datetime as dt
    t0 = dt.datetime.strptime(start, "%Y-%m-%d_%H:%M:%S")
    return [(t0 + dt.timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S") for h in range(n)]


def daily(tmp_path, start="2026-10-27_00:00:00", n=49, **kw):
    src = make_wrfout(tmp_path / "f", times=hours(start, n), **kw)
    return conv(src, str(tmp_path / "o"), "daily")


def test_daily_filename_and_attrs(tmp_path):
    out = daily(tmp_path)
    assert out.endswith("wrf_daily_20261027T00Z.nc")
    with netCDF4.Dataset(out) as ds:
        assert ds.wrfout_tidy_schema == 1
        assert ds.wrfout_tidy_product == "daily"
        assert ds["time"].shape == (2,)  # 48 h = two complete days
        assert "t2" not in ds.variables  # no bare hourly names


def test_daily_time_axis_and_bounds(tmp_path):
    import datetime as dt
    out = daily(tmp_path)
    with netCDF4.Dataset(out) as ds:
        t = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False)
        assert [x.hour for x in t] == [0, 0] and (t[1] - t[0]) == dt.timedelta(days=1)
        b = ds["time_bnds"][:]
        assert np.allclose(b[:, 0], ds["time"][:])
        assert np.allclose(b[:, 1] - b[:, 0], 24.0)
        assert ds["time"].bounds == "time_bnds"


def test_daily_statistics(tmp_path):
    # T2 follows 280 + hour, so day 0 (hours 0..23) has mean 291.5, min 280, max 303;
    # day 1 (hours 24..47) has mean 315.5, min 304, max 327.
    # RAINNC accumulates 1 mm per hour, so each day totals 24 mm; WSPD10MAX is the
    # step index, so day 0 owns steps 1..24 and day 1 steps 25..48.
    out = daily(tmp_path, gust=True, hourly_fn=lambda i: {
        "T2": 280.0 + i, "RAINNC": float(i), "WSPD10MAX": float(i)})
    with netCDF4.Dataset(out) as ds:
        at = lambda n, d: float(ds[n][d, 0, 0])
        assert at("t2_mean", 0) == pytest.approx(291.5, abs=1e-3)
        assert at("t2_min", 0) == pytest.approx(280.0)
        assert at("t2_max", 0) == pytest.approx(303.0)
        assert at("t2_mean", 1) == pytest.approx(315.5, abs=1e-3)
        assert at("t2_min", 1) == pytest.approx(304.0)
        assert at("t2_max", 1) == pytest.approx(327.0)
        assert at("tp_sum", 0) == pytest.approx(24.0 + 0.0 * 0)  # RAINC/RAINSH constant
        assert at("tp_sum", 1) == pytest.approx(24.0)
        assert at("fg10_max", 0) == pytest.approx(24.0)
        assert at("fg10_max", 1) == pytest.approx(48.0)
        assert ds["t2_max"].cell_methods == "time: maximum"
        assert ds["tp_sum"].cell_methods == "time: sum"
        assert ds["t2_mean"].units == "K" and ds["tp_sum"].units == "mm"


def test_daily_variable_set(tmp_path):
    out = daily(tmp_path, gust=True, cu=1)
    expected = {"t2_mean", "t2_min", "t2_max", "d2m_mean", "msl_mean", "u10_mean",
                "v10_mean", "tp_sum", "tcc_mean", "fg10_max", "psfc_mean",
                "pblh_mean", "vegfra_mean", "rainnc_sum", "rainc_sum", "rainsh_sum"}
    with netCDF4.Dataset(out) as ds:
        got = {n for n in ds.variables if ds[n].ndim == 3}
        assert got == expected
        assert np.allclose(ds["pblh_mean"][:], 850.0)
        assert np.allclose(ds["rainc_sum"][:], 0.0)


def test_daily_without_gust_or_cumulus(tmp_path):
    out = daily(tmp_path)
    with netCDF4.Dataset(out) as ds:
        assert "fg10_max" not in ds.variables
        assert "rainc_sum" not in ds.variables


def test_daily_skips_partial_days(tmp_path):
    # Starts 06Z: the first (partial) day is skipped; 06Z..06Z+48h holds one
    # complete day (the 27th->28th needs 00Z on the 27th, which is absent, so
    # the only complete day is the 28th).
    out = daily(tmp_path, start="2026-10-27_06:00:00", n=50)
    with netCDF4.Dataset(out) as ds:
        assert ds["time"].shape == (1,)
        t = netCDF4.num2date(ds["time"][0], ds["time"].units, only_use_cftime_datetimes=False)
        assert (t.day, t.hour) == (28, 0)
        assert "2026-10-27" in ds.history and "skipped" in ds.history


def test_daily_error_without_complete_day(tmp_path, capsys):
    src = make_wrfout(tmp_path / "f", times=hours("2026-10-27_00:00:00", 12))
    with pytest.raises(SystemExit) as e:
        conv(src, str(tmp_path / "o"), "daily")
    msg = str(e.value)
    assert "no complete UTC day" in msg and "2026-10-27 00:00" in msg and "2026-10-27 11:00" in msg
    assert not list((tmp_path / "o").glob("*.nc")) if (tmp_path / "o").exists() else True


def test_daily_needs_closing_midnight(tmp_path):
    # 24 steps (00Z..23Z) lack the closing 00Z, so there is no complete day
    src = make_wrfout(tmp_path / "f", times=hours("2026-10-27_00:00:00", 24))
    with pytest.raises(SystemExit):
        conv(src, str(tmp_path / "o"), "daily")


def test_daily_min_max_use_resampled_field(tmp_path):
    out = daily(tmp_path, lat_axis=MERC_LAT, t2_of_lat=True)
    with netCDF4.Dataset(out) as ds:
        lat = ds["lat"][:]
        assert np.ptp(np.diff(lat)) < 1e-9
        assert np.allclose(ds["t2_max"][0, :, 0], 250.0 + lat, atol=1e-3)
        assert "resampl" in ds.history.lower()


def test_daily_identical_bytes(tmp_path):
    import hashlib
    src = make_wrfout(tmp_path / "f", times=hours("2026-10-27_00:00:00", 25))
    a = conv(src, str(tmp_path / "a"), product="daily")
    b = conv(src, str(tmp_path / "b"), product="daily")
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h(a) == h(b)


# ---- weekly product (7 days from the first 00Z) ------------------------------

def week(tmp_path, start="2026-10-27_00:00:00", n=7 * 24 + 1, **kw):
    src = make_wrfout(tmp_path / "f", times=hours(start, n), **kw)
    return conv(src, str(tmp_path / "o"), "weekly")


def test_weekly_single_window_and_bounds(tmp_path):
    out = week(tmp_path)
    assert out.endswith("wrf_weekly_20261027T00Z.nc")
    with netCDF4.Dataset(out) as ds:
        assert ds.wrfout_tidy_product == "weekly"
        assert ds["time"].shape == (1,)
        b = ds["time_bnds"][0]
        assert float(b[1] - b[0]) == pytest.approx(168.0)
        assert float(ds["time"][0]) == pytest.approx(float(b[0]))
        assert "7 complete UTC days" in ds.history


def test_weekly_statistics(tmp_path):
    # T2 = 280 + step, RAINNC = 1 mm/h: over 168 steps the mean is 280 + 83.5,
    # min 280, max 280 + 167 and the total rain is 168 mm. WSPD10MAX = step, so the
    # window owns steps 1..168 and the max is 168.
    out = week(tmp_path, gust=True, hourly_fn=lambda i: {
        "T2": 280.0 + i, "RAINNC": float(i), "WSPD10MAX": float(i)})
    with netCDF4.Dataset(out) as ds:
        at = lambda n: float(ds[n][0, 0, 0])
        assert at("t2_mean") == pytest.approx(363.5, abs=1e-3)
        assert at("t2_min") == pytest.approx(280.0)
        assert at("t2_max") == pytest.approx(447.0)
        assert at("tp_sum") == pytest.approx(168.0)
        assert at("fg10_max") == pytest.approx(168.0)
        assert ds["tp_sum"].cell_methods == "time: sum"


def test_weekly_variable_set_has_no_wind_components(tmp_path):
    out = week(tmp_path, gust=True, cu=1)
    expected = {"t2_mean", "t2_min", "t2_max", "d2m_mean", "msl_mean", "tp_sum",
                "tcc_mean", "fg10_max", "psfc_mean", "pblh_mean", "vegfra_mean",
                "rainnc_sum", "rainc_sum", "rainsh_sum"}
    with netCDF4.Dataset(out) as ds:
        assert {n for n in ds.variables if ds[n].ndim == 3} == expected


def test_weekly_uses_only_first_seven_days(tmp_path):
    # 10 days of data: rain keeps accumulating, but the window is days 1..7
    out = week(tmp_path, n=10 * 24 + 1, hourly_fn=lambda i: {"RAINNC": float(i)})
    with netCDF4.Dataset(out) as ds:
        assert float(ds["tp_sum"][0, 0, 0]) == pytest.approx(168.0)


def test_weekly_starts_at_first_midnight_not_run_start(tmp_path):
    # run starts 06Z: the window starts at the next 00Z and needs 7 days from there
    out = week(tmp_path, start="2026-10-27_06:00:00", n=18 + 7 * 24 + 1)
    with netCDF4.Dataset(out) as ds:
        assert out.endswith("wrf_weekly_20261027T06Z.nc")
        t = netCDF4.num2date(ds["time"][0], ds["time"].units, only_use_cftime_datetimes=False)
        assert (t.day, t.hour) == (28, 0)


def test_weekly_error_when_run_shorter_than_seven_days(tmp_path):
    src = make_wrfout(tmp_path / "f", times=hours("2026-10-27_00:00:00", 3 * 24 + 1))
    with pytest.raises(SystemExit) as e:
        conv(src, str(tmp_path / "o"), "weekly")
    assert "7 complete UTC days" in str(e.value)


def test_weekly_error_missing_closing_step(tmp_path):
    # 7 days of steps without the closing 00Z (hour 168)
    src = make_wrfout(tmp_path / "f", times=hours("2026-10-27_00:00:00", 7 * 24))
    with pytest.raises(SystemExit):
        conv(src, str(tmp_path / "o"), "weekly")


# ---- several products in one run ------------------------------------------------

def test_default_is_one_hourly_file(tmp_path):
    src = make_wrfout(tmp_path / "f")
    w.main([src, "--outdir", str(tmp_path / "o")])
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == ["wrf_hourly_20261027T06Z.nc"]


def test_daily_and_weekly_in_one_run(tmp_path):
    src = make_wrfout(tmp_path / "f", times=hours("2026-10-27_00:00:00", 7 * 24 + 1),
                      hourly_fn=lambda i: {"RAINNC": float(i)})
    w.main([src, "--product", "daily,weekly", "--outdir", str(tmp_path / "o")])
    names = sorted(p.name for p in (tmp_path / "o").iterdir())
    assert names == ["wrf_daily_20261027T00Z.nc", "wrf_weekly_20261027T00Z.nc"]
    with netCDF4.Dataset(tmp_path / "o" / names[0]) as d, \
            netCDF4.Dataset(tmp_path / "o" / names[1]) as wk:
        assert d["time"].shape == (7,)
        assert float(d["tp_sum"][:, 0, 0].sum()) == pytest.approx(float(wk["tp_sum"][0, 0, 0]))
        assert d.wrfout_tidy_product == "daily" and wk.wrfout_tidy_product == "weekly"


def test_nothing_written_when_one_product_cannot_be_built(tmp_path):
    # 3 days: daily is fine, weekly is impossible, so neither file may appear
    src = make_wrfout(tmp_path / "f", times=hours("2026-10-27_00:00:00", 3 * 24 + 1))
    out = tmp_path / "o"
    with pytest.raises(SystemExit):
        w.main([src, "--product", "daily,weekly", "--outdir", str(out)])
    assert not out.exists() or list(out.iterdir()) == []


def test_unknown_product_and_duplicates(tmp_path):
    src = make_wrfout(tmp_path / "f")
    with pytest.raises(SystemExit):
        w.main([src, "--product", "monthly", "--outdir", str(tmp_path)])
    assert w.parse_products("hourly, hourly") == ["hourly"]


def test_no_keep_flag_and_nothing_deleted(tmp_path):
    with pytest.raises(SystemExit):
        w.main(["x", "--keep", "3"])
    out = tmp_path / "o"
    out.mkdir()
    for n in ["wrf_hourly_20250101T00Z.nc", "notes.txt"]:
        (out / n).write_bytes(b"x")
    w.main([make_wrfout(tmp_path / "f"), "--outdir", str(out)])
    assert (out / "wrf_hourly_20250101T00Z.nc").exists()
    assert (out / "notes.txt").exists()
