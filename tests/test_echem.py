import numpy as np
import pytest

import echem as ec

CV = dict(E_start=0.3, E_switch=-0.3, E0=0.0, n=1, conc=1.0, D=1e-9, area=1e-6)


def test_thermal_voltage():
    assert ec.thermal_voltage() == pytest.approx(0.025693, abs=1e-6)


def test_nernst_copper():
    # Cu2+ + 2e- -> Cu with [Cu2+] = 0.01 M: E = 0.34 - (0.05916/2) * 2
    E = ec.nernst(0.34, 2, Q=1 / 0.01)
    assert E == pytest.approx(0.34 - 0.05916, abs=2e-5)


def test_daniell_cell():
    E = ec.cell_potential("Cu2+/Cu", "Zn2+/Zn")
    assert E == pytest.approx(1.10)
    assert ec.delta_g(2, E) / 1000 == pytest.approx(-212.3, abs=0.1)
    assert np.log10(ec.equilibrium_constant(2, E)) == pytest.approx(37.2, abs=0.1)


def test_nernst_concentration_cell_slope():
    # 59.16 mV per decade for n = 1 at 25 C
    E1, E2 = ec.nernst(0.0, 1, 1.0), ec.nernst(0.0, 1, 10.0)
    assert E1 - E2 == pytest.approx(0.05916, abs=1e-5)


def test_faraday_copper_plating():
    m = ec.mass_deposited(2.0, 3600.0, 63.546, 2)
    assert m == pytest.approx(2.3710, abs=1e-4)  # 7200 C / (2 F) * 63.546 g/mol
    assert ec.mass_deposited(2.0, 3600.0, 63.546, 2, efficiency=0.9) == pytest.approx(0.9 * m)


def test_butler_volmer_limits():
    i0 = 1e-3
    eta = np.linspace(-0.3, 0.3, 601)
    i = ec.butler_volmer(eta, i0)
    np.testing.assert_allclose(i, -i[::-1], atol=1e-15)  # symmetric for alpha = 0.5
    # near equilibrium: linear, eta = i * R_ct
    small = 1e-4
    assert small / ec.butler_volmer(small, i0) == pytest.approx(ec.charge_transfer_resistance(i0), rel=1e-6)


def test_tafel_slope_and_fit():
    assert ec.tafel_slope(0.5) == pytest.approx(0.1183, abs=1e-4)
    eta = np.linspace(-0.4, 0.4, 161)
    for alpha in (0.3, 0.5, 0.7):
        i = ec.butler_volmer(eta, 2e-4, alpha_a=alpha)
        fa = ec.fit_tafel(eta, i, "anodic", min_eta=0.15)
        assert fa["alpha"] == pytest.approx(alpha, rel=1e-3)
        assert fa["i0"] == pytest.approx(2e-4, rel=2e-2)
        fc = ec.fit_tafel(eta, i, "cathodic", min_eta=0.15)
        assert fc["alpha"] == pytest.approx(1 - alpha, rel=1e-3)


def test_koutecky_levich_is_linear():
    omega = 2 * np.pi * np.array([400, 900, 1600, 2500]) / 60  # rpm -> rad/s
    iL = ec.levich_current(1, 0.196e-4, 1e-9, omega, 1e-6, 1.0)
    ik = 5e-5
    i = ec.koutecky_levich(ik, iL)
    m, c = np.polyfit(omega**-0.5, 1 / i, 1)
    assert 1 / c == pytest.approx(ik, rel=1e-9)
    assert np.all(np.diff(iL) > 0)


def test_cv_reversible_matches_randles_sevcik():
    v = 0.1
    cv = ec.simulate_cv(scan_rate=v, **CV)
    Epc, ipc = cv.cathodic_peak()
    ip = ec.randles_sevcik(1, CV["area"], CV["conc"], CV["D"], v)
    assert abs(ipc) == pytest.approx(ip, rel=1.5e-2)
    # peak position: E1/2 - 1.109 RT/F = -28.5 mV for n = 1
    assert Epc == pytest.approx(-1.109 * ec.thermal_voltage(), abs=2e-3)


def test_cv_reversible_peak_separation_and_midpoint():
    cv = ec.simulate_cv(scan_rate=0.1, **CV)
    dEp = cv.peak_separation()
    assert dEp == pytest.approx(ec.REVERSIBLE_PEAK_SEPARATION, abs=2.5e-3)
    mid = 0.5 * (cv.anodic_peak()[0] + cv.cathodic_peak()[0])
    assert mid == pytest.approx(CV["E0"], abs=2e-3)


def test_cv_peak_current_scales_with_sqrt_scan_rate():
    i1 = abs(ec.simulate_cv(scan_rate=0.05, **CV).cathodic_peak()[1])
    i4 = abs(ec.simulate_cv(scan_rate=0.2, **CV).cathodic_peak()[1])
    assert i4 / i1 == pytest.approx(2.0, rel=5e-3)


def test_slow_kinetics_widen_peak_separation():
    fast = ec.simulate_cv(scan_rate=0.1, k0=1.0, **CV)
    slow = ec.simulate_cv(scan_rate=0.1, k0=1e-5, **CV)
    rev = ec.simulate_cv(scan_rate=0.1, **CV)
    assert fast.peak_separation() == pytest.approx(rev.peak_separation(), abs=1e-3)
    assert slow.peak_separation() > rev.peak_separation() + 0.05
    assert abs(slow.cathodic_peak()[1]) < abs(rev.cathodic_peak()[1])


def test_cv_input_validation():
    with pytest.raises(ValueError):
        ec.simulate_cv(E_start=-0.3, E_switch=0.3, scan_rate=0.1)
