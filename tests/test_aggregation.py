import numpy as np

import aggregation
from config import GRID


def test_make_grid_dimensions():
    lon_edges, lat_edges, lon_centers, lat_centers = aggregation.make_grid()
    assert lon_edges.shape[0] == GRID["nlon"] + 1
    assert lat_edges.shape[0] == GRID["nlat"] + 1
    assert lon_centers.shape[0] == GRID["nlon"]
    assert lat_centers.shape[0] == GRID["nlat"]


def test_compute_probability_map_range():
    traj_lon = np.array([[0.0, 1.0], [0.0, 1.0]])
    traj_lat = np.array([[50.0, 51.0], [50.0, 51.0]])
    active = np.array([[True, True], [True, False]])
    prob_map = aggregation.compute_probability_map([(traj_lon, traj_lat, active)])
    assert np.all(prob_map >= 0.0)
    assert np.all(prob_map <= 1.0)


def test_compute_density_map_normalized():
    traj_lon = np.array([[0.0, 0.0], [0.0, 0.0]])
    traj_lat = np.array([[50.0, 50.0], [50.0, 50.0]])
    active = np.array([[True, True], [True, True]])
    density = aggregation.compute_density_map(traj_lon, traj_lat, active)
    assert density.max() == 1.0
