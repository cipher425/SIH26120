"""Sucker-rod pump (SRP) performance model - vectorised over control grids.

Key physics for heavy crude:
* Viscous (Couette) drag on the rod string, amplified by couplings:
      F_visc = 2*pi*mu*v*L / ln(R_t/r_r) * K_coupling
* Rod floating: on the downstroke the rods can only fall as fast as buoyed weight
  overcomes drag. Rod Float Index RFI = (F_visc_down + inertia) / W_rod_buoyed.
  RFI > 1 -> carrier-bar separation, compressive/impact loading, pump unseating.
* VFD "slow-down / fast-up" profile: down_frac = share of cycle time on the downstroke.
* Modified Goodman diagram for rod-stress loading.
"""
import math
import numpy as np

from twin import config as C
from twin.physics import fluid

_DRAG_PER_M = 2 * math.pi / math.log((C.TUBING_ID / 2) / (C.ROD_DIAM / 2)) * C.COUPLING_DRAG_FACTOR
_AP = math.pi * C.PLUNGER_DIAM ** 2 / 4


def srp_eval(spm, stroke, down_frac, q_in_bbl, wc, mu_tubing_cp, mu_pump_cp,
             pump_depth_m, thp_mpa=C.THP_MPA):
    spm = np.asarray(spm, dtype=float)
    stroke = np.asarray(stroke, dtype=float)
    d = np.asarray(down_frac, dtype=float)
    L = pump_depth_m
    rho_f = fluid.mix_density(wc)
    m_rod = C.ROD_MASS_PER_M * L
    W_rb = m_rod * C.G * (1.0 - rho_f / C.RHO_STEEL)

    # theoretical displacement
    disp_bbl = _AP * stroke * spm * 1440.0 / C.BBL

    # fluid load on plunger (net lift ~ pump depth minus small submergence)
    H = L - 40.0
    Fo = rho_f * C.G * H * _AP + thp_mpa * 1e6 * _AP
    stretch = Fo * L / (C.E_STEEL * C.ROD_AREA)
    sp_frac = np.clip((stroke - stretch) / stroke, 0.3, 1.0)

    mu_t = mu_tubing_cp / 1000.0            # Pa.s
    c_visc = _DRAG_PER_M * mu_t * L + 2000.0 * mu_t   # rods + plunger, N per (m/s)
    v_up = math.pi * stroke * spm / (120.0 * (1.0 - d))
    v_dn = math.pi * stroke * spm / (120.0 * d)
    F_up = c_visc * v_up
    F_dn = c_visc * v_dn
    w_up = math.pi * spm / (60.0 * (1.0 - d))
    w_dn = math.pi * spm / (60.0 * d)
    I_up = 0.7 * m_rod * stroke / 2.0 * w_up ** 2
    I_dn = 0.7 * m_rod * stroke / 2.0 * w_dn ** 2

    PPRL = W_rb + Fo + F_up + I_up
    MPRL_raw = W_rb - F_dn - I_dn
    RFI = (F_dn + I_dn) / W_rb
    impact = np.clip(RFI - 1.0, 0.0, None)
    MPRL = np.maximum(MPRL_raw, 0.0)
    PPRL = PPRL + impact * 0.35 * W_rb       # impact on pickup after separation

    # pump filling - viscous standing-valve restriction and inflow limit
    fill_visc = 1.0 / (1.0 + 0.0015 * np.sqrt(mu_pump_cp) * spm)
    cap_eff = disp_bbl * sp_frac * fill_visc * 0.97
    q_liq = np.minimum(q_in_bbl, cap_eff)
    fill_inflow = np.clip(q_in_bbl / np.maximum(cap_eff, 1e-6), 0.0, 1.0)
    fillage = fill_inflow * fill_visc
    pump_eff = q_liq / np.maximum(disp_bbl, 1e-6)

    # energy
    E_hyd = rho_f * C.G * H * q_liq * C.BBL                      # J/day
    v_up_mean = stroke * spm / (60.0 * (1.0 - d))
    v_dn_mean = stroke * spm / (60.0 * d)
    W_fric = c_visc * (v_up_mean + v_dn_mean) * stroke * 1.23    # J/stroke
    E_fric = W_fric * spm * 1440.0
    E_pound = (1.0 - fill_inflow) * Fo * stroke * 0.15 * spm * 1440.0
    kwh = (E_hyd + E_fric + E_pound) / C.SURFACE_EFF / 3.6e6 + C.IDLE_KW * 24.0 * (spm > 0)

    # modified Goodman
    s_max = PPRL / C.ROD_AREA
    s_min = MPRL / C.ROD_AREA
    s_allow = (C.ROD_TENSILE / 4.0 + 0.5625 * s_min) * C.SERVICE_FACTOR
    stress_ratio = (s_max - s_min) / np.maximum(s_allow - s_min, 1.0)

    return {
        "disp_bbl": disp_bbl, "q_liq": q_liq, "fillage": fillage, "fill_inflow": fill_inflow,
        "pump_eff": pump_eff, "PPRL": PPRL, "MPRL": MPRL, "MPRL_raw": MPRL_raw, "RFI": RFI,
        "impact": impact, "kwh": kwh, "stress_ratio": stress_ratio, "W_rb": W_rb + 0 * spm,
        "Fo": Fo + 0 * spm, "c_visc": c_visc + 0 * spm, "stretch": stretch + 0 * spm,
    }


def failure_hazards(RFI, stress_ratio, impact):
    """Daily hazard rates (events/day) for rod parting and pump unseating."""
    RFI = np.asarray(RFI)
    lam_rod = (1.0 / 1100.0) * np.exp(3.0 * np.clip(stress_ratio - 0.6, 0, None)) \
        * (1.0 + 5.0 * impact + 1.0 * (RFI > 0.9))
    lam_unseat = (1.0 / 2500.0) * (1.0 + 20.0 * impact + 1.5 * (RFI > 0.9))
    return lam_rod, lam_unseat
