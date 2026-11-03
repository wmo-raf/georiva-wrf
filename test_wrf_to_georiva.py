import numpy as np
import netCDF4
import pytest

import wrf_to_georiva as w

NY, NX, NT = 4, 5, 2


def make_wrfout(path, map_proj=3, gust=False, skew=False):
    ds = netCDF4.Dataset(str(path), "w")
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
    lat = np.linspace(-1, 1, NY)[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * np.linspace(36, 37, NX)[None, :]
    if skew:
        lon = lon + lat * 0.5
    def var(name, dims, val):
        v = ds.createVariable(name, "f4", dims)
        v[:] = val
    d2 = ("Time", "south_north", "west_east")
    var("XLAT", d2, np.broadcast_to(lat, (NT, NY, NX)))
    var("XLONG", d2, np.broadcast_to(lon, (NT, NY, NX)))
    var("T2", d2, np.full((NT, NY, NX), 300.0))
    var("Q2", d2, np.full((NT, NY, NX), 0.012))
    var("PSFC", d2, np.full((NT, NY, NX), 90000.0))
    var("HGT", d2, np.full((NT, NY, NX), 1000.0))
    var("U10", d2, np.full((NT, NY, NX), 3.0))
    var("V10", d2, np.full((NT, NY, NX), 4.0))
    var("RAINNC", d2, np.full((NT, NY, NX), 2.0))
    var("RAINC", d2, np.full((NT, NY, NX), 1.0))
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
        assert float(ds["tp"][0, 0, 0]) == pytest.approx(3.0)
        assert float(ds["tcc"][0, 0, 0]) == pytest.approx(0.4)
        assert float(ds["d2m"][0, 0, 0]) <= 300.0


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
