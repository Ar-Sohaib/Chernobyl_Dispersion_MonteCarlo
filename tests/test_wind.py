import numpy as np

import wind
import wind_era5


def test_ms_to_deg_per_hour_conversion(monkeypatch):
    monkeypatch.setattr(wind_era5, "_load", lambda _: {"t_hours": np.array([0.0, 1.0]), "lats": np.array([40.0, 60.0]), "lons": np.array([0.0, 2.0]), "u": None, "v": None})
    monkeypatch.setattr(wind_era5, "_interpolator", lambda t_idx, component: (lambda pts: np.full(pts.shape[0], 10.0 if component == "u" else 0.0)))
    u, v = wind_era5.get_wind(np.array([1.0]), np.array([50.0]), t_hours=0.0, turbulence=0.0)
    expected_u = 10.0 * 3600.0 / (111_000.0 * np.cos(np.radians(50.0)))
    assert np.isclose(u[0], expected_u)
    assert np.isclose(v[0], 0.0)


def test_temporal_interpolation_alpha_edges(monkeypatch):
    monkeypatch.setattr(wind_era5, "_load", lambda _: {"t_hours": np.array([0.0, 1.0]), "lats": np.array([40.0, 60.0]), "lons": np.array([0.0, 2.0]), "u": None, "v": None})

    def fake_interpolator(t_idx, component):
        value = 2.0 if t_idx == 0 else 6.0
        return lambda pts: np.full(pts.shape[0], value if component == "u" else 0.0)

    monkeypatch.setattr(wind_era5, "_interpolator", fake_interpolator)
    u0, _ = wind_era5.get_wind(np.array([1.0]), np.array([50.0]), t_hours=0.0, turbulence=0.0)
    u1, _ = wind_era5.get_wind(np.array([1.0]), np.array([50.0]), t_hours=1.0, turbulence=0.0)
    assert u1[0] > u0[0]


def test_wind_phase_transition_blending(monkeypatch):
    phases = [
        {"t_start": 0, "t_end": 10, "u": 1.0, "v": 1.0, "u_std": 0.0, "v_std": 0.0},
        {"t_start": 10, "t_end": 20, "u": 3.0, "v": 3.0, "u_std": 0.0, "v_std": 0.0},
    ]
    monkeypatch.setattr(wind, "WIND_PHASES", phases)
    u, v = wind.get_wind(np.array([0.0]), np.array([0.0]), t_hours=13.0)
    assert np.isclose(u[0], 2.0)
    assert np.isclose(v[0], 2.0)
