"""Cyclic Steam Stimulation reservoir model.

Heating : Marx-Langenheim heated-volume with overburden/underburden losses.
Cooling : Boberg-Lantz average heated-zone temperature
          T = Tb + (Ts - Tb) * [vr*vz*(1 - delta) - delta]
          with closed-form approximations for the conduction factors vr, vz and delta
          the fraction of heat removed by produced fluids (convective loss).
Inflow  : Boberg-Lantz composite radial model (hot zone r<rh, cold zone rh<r<re),
          with relative-permeability decline as the drainage area depletes.
"""
from dataclasses import dataclass, field, asdict
import math

from scipy.special import erfcx

from twin import config as C
from twin.physics import fluid


@dataclass
class WellProps:
    well_id: str
    lat: float
    lon: float
    depth_m: float          # mid-perforation depth
    pump_depth_m: float
    h_m: float              # net pay
    k_md: float
    phi: float
    so_init: float
    swc: float
    sor: float
    p_res_mpa: float
    re_m: float
    rw_m: float
    visc_factor_lab: float  # lab PVT asphaltene factor (what the twin knows)
    stroke_m: float
    wc_base: float

    def to_dict(self):
        return asdict(self)


@dataclass
class TruthParams:
    """Hidden 'real-world' deviations from the nominal model. Only the data generator
    sees these; the digital twin has to learn them from history."""
    k_mult: float = 1.0
    heat_eff: float = 0.88          # wellbore loss + steam conformance
    conformance_decay: float = 0.03  # per cycle loss of steam conformance
    visc_factor: float = 1.0
    channel_p_mpa: float = 6.5      # above this injection pressure steam channels
    channel_loss: float = 0.15      # heat-efficiency loss per MPa above channel_p


@dataclass
class WellState:
    """Multi-cycle memory of the near-wellbore region."""
    cycle_no: int = 0
    cum_oil_m3: float = 0.0
    residual_dT: float = 0.0        # leftover heating from earlier cycles
    p_res_mpa: float = 0.0
    vh_max: float = 0.0

    def copy(self):
        return WellState(**asdict(self))


def marx_langenheim_efficiency(t_days, h_m):
    """Fraction of injected heat retained in the pay zone after t days of injection."""
    tD = 4.0 * C.ALPHA_ROCK * max(t_days, 1e-3) / (h_m ** 2)
    G = erfcx(math.sqrt(tD)) + 2.0 * math.sqrt(tD / math.pi) - 1.0
    return max(0.05, min(1.0, G / tD))


class CSSReservoir:
    """Reservoir response for a single CSS cycle."""

    def __init__(self, well: WellProps, css: dict, state: WellState, heat_eff: float,
                 k_mult: float, visc_factor: float, truth: TruthParams = None):
        self.w = well
        self.css = css
        self.vf = visc_factor
        self.k_mult = k_mult
        self.state = state
        self.p_inj = css["inj_pressure_mpa"]
        self.Ts = fluid.t_sat_c(self.p_inj)
        self.T_bg = C.T_RES + state.residual_dT
        # effective heat efficiency (truth adds conformance decay & high-pressure channeling)
        eff = heat_eff
        if truth is not None:
            eff *= max(0.65, 1.0 - truth.conformance_decay * max(0, state.cycle_no - 1))
            eff *= (1.0 - truth.channel_loss * max(0.0, self.p_inj - truth.channel_p_mpa))
        self.heat_eff = max(0.3, eff)
        steam_kg = css["steam_t"] * 1000.0
        h_per_kg = fluid.steam_enthalpy_above(self.T_bg, self.p_inj, css["quality"])
        self.Q_inj = steam_kg * h_per_kg                 # kJ
        self.inj_days = max(1, math.ceil(css["steam_t"] / C.STEAM_GEN_RATE))
        E_ml = marx_langenheim_efficiency(self.inj_days, well.h_m)
        self.Q_res = self.Q_inj * self.heat_eff
        dT = max(10.0, self.Ts - self.T_bg)
        self.Vh = E_ml * self.Q_res / (C.M_ROCK * dT)    # m3 heated rock
        self.rh = max(well.rw_m * 3, math.sqrt(self.Vh / (math.pi * well.h_m)))
        self.rh = min(self.rh, 0.8 * well.re_m)
        self.heat0 = self.Vh * C.M_ROCK * dT
        self.delta = 0.0
        # condensed water stored near wellbore, returns during production
        self.W_rem = 0.6 * css["steam_t"]                # m3
        self.dp_steam = 0.4 * max(0.0, self.p_inj - state.p_res_mpa)
        # drainage-volume oil saturation (depletes cycle by cycle)
        v_drain = math.pi * 60.0 ** 2 * well.h_m * well.phi
        self.so = max(well.sor + 0.03, well.so_init - state.cum_oil_m3 / v_drain)
        s_norm = (self.so - well.sor) / (1.0 - well.sor - well.swc)
        self.kro = max(0.03, min(1.0, s_norm ** 1.5))

    # ------------------------------------------------------------------
    def avg_temperature(self, t_days):
        """Boberg-Lantz average heated-zone temperature t days after injection ended."""
        tdr = C.ALPHA_ROCK * t_days / self.rh ** 2
        tdz = 4.0 * C.ALPHA_ROCK * t_days / self.w.h_m ** 2
        vr = 1.0 / math.sqrt(1.0 + 5.0 * tdr)
        vz = 1.0 / math.sqrt(1.0 + 5.0 * tdz)
        f = vr * vz * (1.0 - self.delta) - self.delta
        return self.T_bg + (self.Ts - self.T_bg) * max(0.0, f)

    def inflow_capacity(self, T_avg, t_days, pwf_mpa=0.35):
        """Oil & water inflow capacity (bbl/d) at pumped-off bottomhole pressure."""
        w = self.w
        mu_h = fluid.fast_oil_visc(T_avg, self.vf)
        mu_c = fluid.fast_oil_visc(self.T_bg, self.vf)
        # asphaltene skin is thermally cleaned inside the heated zone (skin scales with mu ratio)
        ln_e = math.log(w.re_m / w.rw_m) + C.SKIN_COLD
        ratio = ln_e / ((mu_h / mu_c) * (math.log(self.rh / w.rw_m) + C.SKIN_COLD)
                        + math.log(w.re_m / self.rh))
        h_ft = w.h_m * 3.281
        j_cold = 0.00708 * w.k_md * self.k_mult * h_ft / (mu_c * C.OIL_FVF * ln_e)  # bbl/d/psi
        pe = self.state.p_res_mpa + self.dp_steam * math.exp(-t_days / 30.0)
        dp_psi = max(0.0, (pe - pwf_mpa)) * C.PSI_PER_MPA
        q_o = j_cold * self.kro * ratio * dp_psi
        q_w_cond = self.W_rem * 0.02 / C.BBL
        q_w_form = q_o * w.wc_base / (1 - w.wc_base)
        return q_o, q_w_cond, q_w_form, mu_h

    def produce(self, oil_bbl, water_cond_bbl, water_form_bbl, T_avg):
        """Remove produced heat & condensate from the heated zone."""
        m_o = oil_bbl * C.BBL * C.RHO_OIL
        m_w = (water_cond_bbl + water_form_bbl) * C.BBL * C.RHO_WATER
        heat_out = (m_o * C.CP_OIL + m_w * C.CP_WATER) * max(0.0, T_avg - self.T_bg)
        self.delta = min(0.95, self.delta + heat_out / max(self.heat0, 1.0))
        self.W_rem = max(0.0, self.W_rem - water_cond_bbl * C.BBL)

    def close_cycle(self, T_end, oil_bbl_cycle):
        s = self.state
        s.cum_oil_m3 += oil_bbl_cycle * C.BBL
        s.residual_dT = 0.45 * max(0.0, T_end - C.T_RES)
        s.vh_max = max(s.vh_max, self.Vh)
        s.p_res_mpa = max(1.5, s.p_res_mpa - 0.00002 * oil_bbl_cycle)
