"""Equilibrium electrochemistry: Nernst equation, cell thermodynamics and Faraday's laws."""

from __future__ import annotations

import numpy as np

__all__ = [
    "F", "R", "T25", "STANDARD_POTENTIALS", "thermal_voltage", "nernst",
    "cell_potential", "delta_g", "equilibrium_constant", "charge_passed",
    "mass_deposited", "moles_from_charge",
]

F = 96485.33212        # C / mol
R = 8.314462618        # J / (mol K)
T25 = 298.15           # K

#: Standard reduction potentials at 25 C, V vs SHE: name -> (E0, electrons)
STANDARD_POTENTIALS = {
    "Li+/Li": (-3.04, 1),
    "Zn2+/Zn": (-0.76, 2),
    "Fe2+/Fe": (-0.44, 2),
    "Ni2+/Ni": (-0.26, 2),
    "Pb2+/Pb": (-0.13, 2),
    "H+/H2": (0.00, 2),
    "Cu2+/Cu": (0.34, 2),
    "Fe3+/Fe2+": (0.77, 1),
    "Ag+/Ag": (0.80, 1),
    "O2/H2O": (1.23, 4),
    "Cl2/Cl-": (1.36, 2),
}


def thermal_voltage(T: float = T25) -> float:
    """``R T / F`` in volts (25.69 mV at 25 C)."""
    return R * T / F


def nernst(E0: float, n: int, Q, T: float = T25):
    """Electrode or cell potential ``E = E0 - (RT/nF) ln Q``.

    For a single couple ``Ox + n e- -> Red``, ``Q = a_Red / a_Ox``.
    """
    return E0 - thermal_voltage(T) / n * np.log(Q)


def cell_potential(cathode: str, anode: str) -> float:
    """Standard cell potential ``E0_cathode - E0_anode`` from the table."""
    return STANDARD_POTENTIALS[cathode][0] - STANDARD_POTENTIALS[anode][0]


def delta_g(n: int, E: float) -> float:
    """Gibbs energy of the cell reaction, ``-n F E`` (J/mol)."""
    return -n * F * E


def equilibrium_constant(n: int, E0: float, T: float = T25) -> float:
    """``K = exp(n F E0 / (R T))``."""
    return float(np.exp(n * E0 / thermal_voltage(T)))


def charge_passed(current: float, time_s: float) -> float:
    return current * time_s


def moles_from_charge(charge: float, n: int) -> float:
    return charge / (n * F)


def mass_deposited(current: float, time_s: float, molar_mass: float, n: int, efficiency: float = 1.0) -> float:
    """Faraday's law: mass (in the units of ``molar_mass``, e.g. g) plated by a constant current."""
    return efficiency * moles_from_charge(charge_passed(current, time_s), n) * molar_mass
