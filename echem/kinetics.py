"""Electrode kinetics: Butler-Volmer, Tafel analysis and rotating-disk mass transport.

Sign convention (IUPAC): anodic (oxidation) currents are positive and
overpotential ``eta = E - E_eq``.
"""

from __future__ import annotations

import numpy as np

from .thermo import F, R, T25

__all__ = [
    "butler_volmer", "tafel_slope", "charge_transfer_resistance", "fit_tafel",
    "levich_current", "koutecky_levich",
]


def butler_volmer(eta, i0: float, alpha_a: float = 0.5, alpha_c: float | None = None,
                  n: int = 1, T: float = T25):
    """Current density ``i0 [exp(alpha_a n f eta) - exp(-alpha_c n f eta)]`` with ``f = F/RT``.

    ``alpha_c`` defaults to ``1 - alpha_a``.
    """
    if alpha_c is None:
        alpha_c = 1.0 - alpha_a
    f = n * F / (R * T)
    eta = np.asarray(eta, dtype=float)
    return i0 * (np.exp(alpha_a * f * eta) - np.exp(-alpha_c * f * eta))


def tafel_slope(alpha: float, n: int = 1, T: float = T25) -> float:
    """Tafel slope in V per decade, ``ln(10) R T / (alpha n F)`` (118 mV/dec for alpha = 0.5)."""
    return np.log(10) * R * T / (alpha * n * F)


def charge_transfer_resistance(i0: float, n: int = 1, T: float = T25) -> float:
    """Small-signal resistance ``R T / (n F i0)`` (ohm * area when ``i0`` is a current density)."""
    return R * T / (n * F * i0)


def fit_tafel(eta, i, branch: str = "anodic", min_eta: float = 0.1, n: int = 1, T: float = T25):
    """Fit ``log10|i|`` against ``eta`` on one Tafel branch.

    Returns ``dict(i0, alpha, slope)``, using only points with ``|eta| >= min_eta``
    (far enough from equilibrium that the back reaction is negligible).
    """
    eta = np.asarray(eta, dtype=float)
    i = np.asarray(i, dtype=float)
    sel = eta >= min_eta if branch == "anodic" else eta <= -min_eta
    if sel.sum() < 2:
        raise ValueError("not enough points on the requested Tafel branch")
    m, c = np.polyfit(eta[sel], np.log10(np.abs(i[sel])), 1)
    slope = 1.0 / abs(m)
    alpha = np.log(10) * R * T / (slope * n * F)
    return {"i0": 10.0**c, "alpha": float(alpha), "slope": float(slope)}


def levich_current(n: int, area: float, D: float, omega: float, nu: float, conc: float) -> float:
    """Levich limiting current at a rotating disk (SI units).

    ``i_L = 0.620 n F A D^(2/3) omega^(1/2) nu^(-1/6) C`` with ``omega`` in rad/s,
    ``nu`` the kinematic viscosity (m^2/s) and ``C`` in mol/m^3.
    """
    return 0.620 * n * F * area * D ** (2 / 3) * np.sqrt(omega) * nu ** (-1 / 6) * conc


def koutecky_levich(i_kinetic: float, i_limit):
    """Measured current from kinetic and mass-transport limits, ``1/i = 1/i_k + 1/i_L``."""
    return 1.0 / (1.0 / i_kinetic + 1.0 / np.asarray(i_limit, dtype=float))
