# electrochem-kit

![tests](https://github.com/luisrowork1992-netizen/electrochem-kit/actions/workflows/tests.yml/badge.svg)

`echem` is a Python toolkit for **electrochemistry**. It covers equilibrium (Nernst, cell potentials, Faraday's laws), electrode kinetics (Butler–Volmer, Tafel, rotating disk) and a finite-difference **cyclic voltammetry simulator** for reversible and quasi-reversible couples.

## Features

- Table of standard reduction potentials
- Nernst equation, standard cell potential, ΔG = −nFE and K = exp(nFE/RT)
- Faraday's law: charge, moles and mass plated, with a current efficiency
- Butler–Volmer current density with independent anodic and cathodic transfer coefficients
- Tafel slope, Tafel fit on either branch (returns i₀ and α) and charge-transfer resistance
- Levich limiting current and the Koutecky–Levich combination for rotating-disk electrodes
- `simulate_cv`: implicit finite-difference simulation of semi-infinite linear diffusion
  - Nernstian or Butler–Volmer (k₀, α) boundary
  - peak finding and peak separation
- Randles–Sevcik peak current

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/electrochem-kit.git
cd electrochem-kit
pip install -e .
```

## Quick start

```python
import echem as ec

print(ec.cell_potential("Cu2+/Cu", "Zn2+/Zn"))     # 1.10 V (Daniell cell)
print(ec.nernst(0.34, 2, Q=1/0.01))               # Cu electrode in 0.01 M Cu2+
print(ec.mass_deposited(2.0, 3600, 63.546, 2))    # 2.371 g of Cu in 1 h at 2 A

cv = ec.simulate_cv(E_start=0.3, E_switch=-0.3, scan_rate=0.1,
                    conc=1.0, D=1e-9, area=1e-6)  # SI units: 1 mM, 1 mm^2
print(cv.cathodic_peak(), cv.peak_separation())   # ip ~ -2.686 uA, dEp ~ 57.7 mV
print(ec.randles_sevcik(1, 1e-6, 1.0, 1e-9, 0.1)) # 2.686e-6 A
```

## Notebook

[`notebooks/electrochemistry.ipynb`](notebooks/electrochemistry.ipynb) covers five topics:

- galvanic cells from standard potentials
- a Nernst concentration cell
- a Tafel analysis that recovers α and i₀ from noisy synthetic data
- a Koutecky–Levich analysis
- simulated cyclic voltammograms:
  - the scan-rate dependence, compared with Randles–Sevcik
  - how slow electron transfer (k₀) widens the peak separation

## Tests

```bash
pip install pytest
pytest
```

The tests check the following:

- the thermal voltage and the Nernst slope of 59.16 mV per decade
- the Daniell cell (E, ΔG, log K)
- Faraday plating
- Butler–Volmer symmetry and its linear limit (R_ct)
- Tafel fits recover α and i₀ on both branches
- the Koutecky–Levich intercept
- the simulated CV peak current matches Randles–Sevcik within 1.5 %
- the cathodic peak sits at E½ − 1.109 RT/F
- ΔEp ≈ 2.218 RT/F with the midpoint at E⁰
- i_p ∝ √v
- fast k₀ reproduces the reversible case, and slow k₀ widens ΔEp and lowers i_p

## Scope and limitations

- Activities are taken as concentrations. The standard potentials are rounded textbook values at 25 °C.
- The CV simulator assumes equal diffusion coefficients for O and R, planar semi-infinite diffusion, and no uncompensated resistance or double-layer charging.

## References

- A. J. Bard and L. R. Faulkner, *Electrochemical Methods*, 2nd ed., Wiley (2001).
- R. S. Nicholson and I. Shain, *Anal. Chem.* 36, 706 (1964).
- D. Britz and J. Strutwolf, *Digital Simulation in Electrochemistry*, 4th ed., Springer (2016).

## License

MIT
