import numpy as np
import netCDF4
import pytest

import wrf_to_georiva as w

NY, NX, NT = 4, 5, 2


def make_wrfout(path, map_proj=3, gust=False, skew=False, lat_axis=None, t2_of_lat=False,
                rainc=True, cu=0, shcu=0):
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
    ds.SIMULATION_START_DATE = "2026-10-27_06:00:00"
    ds.TITLE = "OUTPUT FROM WRF"
    times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
    for i, s in enumerate(["2026-10-27_06:00:00", "2026-10-27_07:00:00"]):
        times[i, :] = netCDF4.stringtochar(np.array([s], "S19"))[0]
    ax = np.linspace(-1, 1, NY) if lat_axis is None else np.asarray(lat_axis, float)
    lat = ax[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * np.linspace(36, 37, NX)[None, :]
    if skew:
        lon = lon + lat * 0.5
    def var(name, dims, val):
        v = ds.createVariable(name, "f4", dims)
        v[:] = val
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
    out = w.convert(src, str(tmp_path / "out"))
    assert out.endswith("wrf_20261027T06Z.nc")
    with netCDF4.Dataset(out) as ds:
        assert ds.georiva_wrf_schema == 1
        units = {n: ds[n].units for n in ["t2", "d2m", "msl", "u10", "v10", "tp", "tcc", "fg10"]}
        assert units == {"t2": "K", "d2m": "K", "msl": "Pa", "u10": "m s-1",
                         "v10": "m s-1", "tp": "mm", "tcc": "1", "fg10": "m s-1"}
        assert ds["t2"].shape == (NT, NY, NX)
        assert float(ds["tp"][0, 0, 0]) == pytest.approx(3.5)
        assert float(ds["tcc"][0, 0, 0]) == pytest.approx(0.4)
        assert float(ds["d2m"][0, 0, 0]) <= 300.0


def test_direct_copies(tmp_path):
    src = make_wrfout(tmp_path / "f")
    out = w.convert(src, str(tmp_path / "o"))
    expected = {"psfc": (90000.0, "Pa"), "pblh": (850.0, "m"), "vegfra": (62.5, "%")}
    with netCDF4.Dataset(out) as ds:
        assert ds.georiva_wrf_schema == 1
        for name, (value, units) in expected.items():
            assert ds[name].units == units
            assert ds[name].shape == (NT, NY, NX)
            assert np.allclose(ds[name][:], value)
        # with no cumulus physics, rain is only written as tp
        assert not {"rainc", "rainnc", "rainsh"} & set(ds.variables)


def test_tp_without_rainc(tmp_path):
    out = w.convert(make_wrfout(tmp_path / "f", rainc=False), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert float(ds["tp"][0, 0, 0]) == pytest.approx(2.0)


def test_direct_copies_resampled_with_mercator_rows(tmp_path):
    src = make_wrfout(tmp_path / "f", lat_axis=MERC_LAT)
    out = w.convert(src, str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        # constant fields stay constant through the row resampling
        assert np.allclose(ds["pblh"][:], 850.0)
        assert ds["pblh"].shape == ds["t2"].shape


def test_rain_components_with_cumulus(tmp_path):
    out = w.convert(make_wrfout(tmp_path / "f", cu=1), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert np.allclose(ds["rainnc"][:], 2.0)
        assert np.allclose(ds["rainc"][:], 1.0)
        assert np.allclose(ds["rainsh"][:], 0.5)
        assert np.allclose(ds["tp"][:], 3.5)


def test_rain_components_with_shallow_cumulus_only(tmp_path):
    out = w.convert(make_wrfout(tmp_path / "f", shcu=1), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert "rainsh" in ds.variables and "rainnc" in ds.variables


def test_rain_components_missing_wrfout_var(tmp_path):
    out = w.convert(make_wrfout(tmp_path / "f", cu=1, rainc=False), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert "rainnc" in ds.variables
        assert "rainc" not in ds.variables and "rainsh" not in ds.variables


def test_rain_components_resampled(tmp_path):
    out = w.convert(make_wrfout(tmp_path / "f", lat_axis=MERC_LAT, cu=1), str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert ds["rainc"].shape == ds["t2"].shape
        assert np.allclose(ds["rainc"][:], 1.0)


def test_gust_optional(tmp_path):
    out = w.convert(make_wrfout(tmp_path / "f"), str(tmp_path / "o"))
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
        w.convert(src, str(tmp_path / "o"))
    msg = str(e.value)
    assert "Lambert" in msg and "Mercator" in msg and "lat-lon" in msg


def test_no_ftp_flags():
    with pytest.raises(SystemExit):
        w.main(["x", "--ftp-host", "h"])


def test_prune_never_removes_file_just_written(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    for d in ["20270101T00Z", "20270102T00Z"]:
        (out / ("wrf_%s.nc" % d)).write_bytes(b"x")
    new = out / "wrf_20261027T06Z.nc"
    new.write_bytes(b"x")
    w.prune_local(str(out), 1, protect=str(new))
    assert new.exists()


def test_same_input_gives_identical_bytes(tmp_path):
    import hashlib
    src = make_wrfout(tmp_path / "f")
    a = w.convert(src, str(tmp_path / "a"))
    b = w.convert(src, str(tmp_path / "b"))
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h(a) == h(b)


MERC_LAT = np.degrees(np.arctan(np.sinh(np.linspace(-0.6, 0.2, NY))))


def test_mercator_rows_resampled_to_uniform_axis(tmp_path):
    src = make_wrfout(tmp_path / "f", lat_axis=MERC_LAT, t2_of_lat=True)
    out = w.convert(src, str(tmp_path / "o"))
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
    out = w.convert(src, str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        assert np.allclose(ds["lat"][:], np.linspace(-1, 1, NY))
        assert "resampl" not in ds.history.lower()


def test_descending_latitude_resampled(tmp_path):
    src = make_wrfout(tmp_path / "f", lat_axis=MERC_LAT[::-1], t2_of_lat=True)
    out = w.convert(src, str(tmp_path / "o"))
    with netCDF4.Dataset(out) as ds:
        lat = ds["lat"][:]
        assert np.ptp(np.diff(lat)) < 1e-9 and lat[0] > lat[-1]
        assert np.allclose(ds["t2"][0, :, 0], 250.0 + lat, atol=1e-3)
