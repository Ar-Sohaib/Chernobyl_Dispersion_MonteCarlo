import numpy as np

import engine


def test_run_single_simulation_reproducible(monkeypatch):
    monkeypatch.setattr(engine, "get_wind", lambda lons, lats, t_hours, rng=None: (np.zeros_like(lons), np.zeros_like(lats)))
    traj_lon_1, traj_lat_1, active_1 = engine.run_single_simulation(seed=42)
    traj_lon_2, traj_lat_2, active_2 = engine.run_single_simulation(seed=42)
    assert np.array_equal(traj_lon_1, traj_lon_2)
    assert np.array_equal(traj_lat_1, traj_lat_2)
    assert np.array_equal(active_1, active_2)


def test_emission_progressive_increases_active_particles(monkeypatch):
    monkeypatch.setattr(engine, "get_wind", lambda lons, lats, t_hours, rng=None: (np.zeros_like(lons), np.zeros_like(lats)))
    _, _, active = engine.run_single_simulation(seed=1)
    assert np.sum(active[0]) <= np.sum(active[1]) <= np.sum(active[2])


def test_out_of_bounds_particles_are_deactivated(monkeypatch):
    monkeypatch.setattr(engine, "get_wind", lambda lons, lats, t_hours, rng=None: (np.full_like(lons, 1e6), np.full_like(lats, 1e6)))
    _, _, active = engine.run_single_simulation(seed=1)
    released_step0 = int(np.sum(active[0]))
    assert not np.any(active[1, :released_step0])


def test_zero_particles_does_not_crash(monkeypatch):
    monkeypatch.setitem(engine.SIMULATION, "n_particles", 0)
    monkeypatch.setattr(engine, "get_wind", lambda lons, lats, t_hours, rng=None: (np.zeros_like(lons), np.zeros_like(lats)))
    traj_lon, traj_lat, active = engine.run_single_simulation(seed=1)
    assert traj_lon.shape[1] == 0
    assert traj_lat.shape[1] == 0
    assert active.shape[1] == 0
