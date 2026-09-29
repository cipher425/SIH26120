"""Wellbore heat transmission (Ramey, 1962) for fluid flowing up the tubing."""
import math
import numpy as np

from twin import config as C
from twin.physics import fluid


def ramey_profile(T_in, q_liq_bbl, pump_depth_m, n=20):
    """Fluid temperature profile from pump (bottom) to surface.
    Returns (depths_m, T_fluid_c). Relaxation distance A grows with mass rate."""
    A = max(40.0, 4.0 * max(q_liq_bbl, 1.0))
    z = np.linspace(0.0, pump_depth_m, n)
    Te = C.T_SURF + C.GEO_GRAD * z
    Te_pump = C.T_SURF + C.GEO_GRAD * pump_depth_m
    dz = pump_depth_m - z
    e = np.exp(-dz / A)
    Tf = Te + (T_in - Te_pump) * e + C.GEO_GRAD * A * (1.0 - e)
    return z, Tf


def tubing_state(T_in, q_liq_bbl, wc, pump_depth_m, vf):
    """Average tubing-fluid viscosity seen by rods, wellhead and pump temperatures."""
    z, Tf = ramey_profile(T_in, q_liq_bbl, pump_depth_m)
    mu_o = fluid.oil_viscosity_cp(Tf, vf)
    mu_mix = fluid.mixture_viscosity_cp(mu_o, wc)
    return {
        "T_wellhead": float(Tf[0]),
        "T_pump": float(Tf[-1]),
        "T_tubing_avg": float(Tf.mean()),
        "mu_tubing_cp": float(mu_mix.mean()),
        "mu_pump_cp": float(mu_mix[-1]),
        "profile_T": Tf,
        "profile_z": z,
    }


def earth_temp(z):
    return C.T_SURF + C.GEO_GRAD * z
