from config import GRID, SIMULATION


def test_config_coherence():
    assert SIMULATION["n_steps"] > 0
    assert SIMULATION["dt"] > 0
    assert GRID["lon_min"] < GRID["lon_max"]
    assert GRID["lat_min"] < GRID["lat_max"]
    assert GRID["nlon"] > 0
    assert GRID["nlat"] > 0
