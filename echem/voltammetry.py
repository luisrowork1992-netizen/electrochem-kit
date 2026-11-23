"""Cyclic voltammetry of ``O + n e- <=> R`` at a planar electrode (finite differences).

Semi-infinite linear diffusion, equal diffusion coefficients, only ``O`` present
at the start. Each time step is solved implicitly (backward Euler), so it is
unconditionally stable. The electrode boundary is either Nernstian (reversible)
or Butler-Volmer with a standard rate constant ``k0`` (quasi-reversible).

Units are SI: concentration mol/m^3 (1 mM = 1 mol/m^3), D m^2/s, area m^2, k0 m/s.
Reduction currents are negative (IUPAC convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .thermo import F, R, T25

__all__ = ["CVResult", "simulate_cv", "randles_sevcik", "REVERSIBLE_PEAK_SEPARATION"]

#: Peak separation of a reversible couple, 2.218 RT/nF (57.0 mV at 25 C for n = 1)
REVERSIBLE_PEAK_SEPARATION = 2.218 * R * T25 / F


@dataclass
class CVResult:
    E: np.ndarray        # applied potential (V)
    i: np.ndarray        # current (A)
    t: np.ndarray        # time (s)
    forward: np.ndarray  # boolean mask for the forward (cathodic) sweep

    def cathodic_peak(self):
        k = np.argmin(np.where(self.forward, self.i, np.inf))
        return float(self.E[k]), float(self.i[k])

    def anodic_peak(self):
        k = np.argmax(np.where(~self.forward, self.i, -np.inf))
        return float(self.E[k]), float(self.i[k])

    def peak_separation(self) -> float:
        return self.anodic_peak()[0] - self.cathodic_peak()[0]


def randles_sevcik(n: int, area: float, conc: float, D: float, scan_rate: float, T: float = T25) -> float:
    """Reversible peak current magnitude, ``0.4463 n F A C sqrt(n F v D / (R T))``."""
    return 0.4463 * n * F * area * conc * np.sqrt(n * F * scan_rate * D / (R * T))


def simulate_cv(E_start: float, E_switch: float, scan_rate: float, E0: float = 0.0,
                n: int = 1, conc: float = 1.0, D: float = 1e-9, area: float = 1e-6,
                k0: float | None = None, alpha: float = 0.5, T: float = T25,
                dE: float = 2.5e-4, n_x: int | None = None) -> CVResult:
    """Simulate one cycle: sweep from ``E_start`` down to ``E_switch`` and back.

    ``k0=None`` gives a reversible (Nernstian) electrode; otherwise Butler-Volmer
    kinetics with standard rate constant ``k0`` and transfer coefficient ``alpha``.
    """
    if E_switch >= E_start:
        raise ValueError("this simulator sweeps cathodically first: need E_switch < E_start")
    f = n * F / (R * T)
    E_fwd = np.arange(E_start, E_switch, -dE)
    E = np.concatenate([E_fwd, E_switch + np.arange(0, E_start - E_switch + dE / 2, dE)])
    forward = np.arange(E.size) < E_fwd.size
    dt = dE / scan_rate
    t_total = dt * E.size
    L = 6.0 * np.sqrt(D * t_total)
    if n_x is None:
        n_x = 800
    dx = L / (n_x - 1)
    lam = D * dt / dx**2

    c = np.full(n_x, conc)  # concentration of O
    i_out = np.empty(E.size)
    # banded matrix rows: (upper, diag, lower); interior nodes 1..n_x-2, bulk fixed at the far end
    ab = np.zeros((3, n_x))
    ab[0, 2:] = -lam
    ab[1, 1:-1] = 1 + 2 * lam
    ab[2, :-2] = -lam
    ab[1, -1] = 1.0
    ab[2, -2] = 0.0  # far boundary row: c = conc
    for k, Ek in enumerate(E):
        rhs = c.copy()
        rhs[-1] = conc
        if k0 is None:
            theta = np.exp(f * (Ek - E0))
            ab[1, 0], ab[0, 1] = 1.0, 0.0
            rhs[0] = conc * theta / (1.0 + theta)
        else:
            kf = k0 * np.exp(-alpha * f * (Ek - E0))       # reduction
            kb = k0 * np.exp((1 - alpha) * f * (Ek - E0))  # oxidation
            # D (c1 - c0)/dx = kf c0 - kb (conc - c0)
            ab[1, 0] = D / dx + kf + kb
            ab[0, 1] = -D / dx
            rhs[0] = kb * conc
        c = solve_banded((1, 1), ab, rhs)
        flux = D * (-3 * c[0] + 4 * c[1] - c[2]) / (2 * dx)  # mol/(m^2 s) of O arriving
        i_out[k] = -n * F * area * flux
    return CVResult(E, i_out, np.arange(E.size) * dt, forward)
