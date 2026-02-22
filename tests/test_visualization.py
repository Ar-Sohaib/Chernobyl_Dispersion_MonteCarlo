import numpy as np

import visualization


def test_filter_points_in_extent_removes_out_of_bounds(monkeypatch):
    monkeypatch.setitem(visualization.GRID, "lon_min", 0.0)
    monkeypatch.setitem(visualization.GRID, "lon_max", 1.0)
    monkeypatch.setitem(visualization.GRID, "lat_min", 0.0)
    monkeypatch.setitem(visualization.GRID, "lat_max", 1.0)

    lons = np.array([-0.1, 0.3, 1.0, 1.2])
    lats = np.array([0.5, 0.4, 1.0, 0.3])

    lon_in, lat_in = visualization._filter_points_in_extent(lons, lats)

    assert np.array_equal(lon_in, np.array([0.3, 1.0]))
    assert np.array_equal(lat_in, np.array([0.4, 1.0]))
