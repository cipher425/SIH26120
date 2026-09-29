"""Fluid & steam property correlations."""
import math
import numpy as np

from twin import config as C


def oil_viscosity_cp(T_c, visc_factor=1.0):
    """Dead-oil viscosity (cP) from an Arrhenius/Andrade fit, scaled by a per-well
    asphaltene factor. Works on scalars and numpy arrays."""
    TK = np.asarray(T_c, dtype=float) + 273.15
    mu = visc_factor * np.exp(C.VISC_A + C.VISC_B / TK)
    return float(mu) if mu.ndim == 0 else mu


def mixture_viscosity_cp(mu_o, wc):
    """Effective viscosity of an oil/water stream seen by the rod string.
    Oil-continuous emulsions thicken (Einstein-type); above ~60 % water cut the
    emulsion inverts to water-continuous and viscosity collapses."""
    s = 1.0 / (1.0 + np.exp(-(np.asarray(wc) - 0.6) / 0.04))
    oil_cont = mu_o * (1.0 + 2.5 * np.asarray(wc))
    water_cont = np.maximum(2.0, 0.05 * mu_o)
    out = (1 - s) * oil_cont + s * water_cont
    return float(out) if np.ndim(out) == 0 else out


def t_sat_c(p_mpa):
    """Saturation temperature of steam (degC), valid ~0.3-10 MPa."""
    return 100.0 * (p_mpa / 0.101325) ** 0.25


def latent_heat_kj(T_c):
    """Latent heat of vaporisation (kJ/kg), Watson correlation."""
    TK = T_c + 273.15
    return 2257.0 * max(0.0, (647.1 - TK) / (647.1 - 373.15)) ** 0.38


def steam_enthalpy_above(T_ref, p_mpa, quality):
    """Heat (kJ/kg) carried by wet steam above reference temperature."""
    Ts = t_sat_c(p_mpa)
    return C.CP_WATER * (Ts - T_ref) + quality * latent_heat_kj(Ts)


def pvt_table(visc_factor=1.0):
    temps = np.arange(35, 261, 5)
    return [{"T_c": float(t), "mu_cp": round(oil_viscosity_cp(t, visc_factor), 2)} for t in temps]


def mix_density(wc):
    return wc * C.RHO_WATER + (1 - wc) * C.RHO_OIL


def fast_oil_visc(T_c, vf):
    return vf * math.exp(C.VISC_A + C.VISC_B / (T_c + 273.15))
