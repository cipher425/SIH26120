"""Coupled well-to-surface simulator for one CSS cycle, stepped one day at a time.

reservoir (Boberg-Lantz)  ->  wellbore (Ramey)  ->  rod pump (SRP physics)
      ^                                                   |
      +-------- produced heat / condensate removal -------+

The same engine is used (a) with hidden TruthParams + random failures + sensor
noise to *generate* field history, and (b) with calibrated parameters by the
digital twin for prediction, what-if analysis and optimisation.
"""
import math
from datetime import date, timedelta

import numpy as np

from twin import config as C
from twin.physics import srp, wellbore
from twin.physics.reservoir import CSSReservoir, WellProps, WellState, TruthParams


def classify_state(r, early_hot):
    if r["phase_failure"] == "pump_unseat":
        return "pump_unseated"
    if r["RFI"] > 1.0:
        return "rod_float"
    if early_hot:
        return "gas_interference"
    if r["fill_inflow"] < 0.8:
        return "fluid_pound"
    return "normal"


class CycleSim:
    def __init__(self, well: WellProps, css: dict, state: WellState, start: date,
                 heat_eff, k_mult, visc_factor, truth: TruthParams = None,
                 rng=None, stochastic=False, oil_correction=1.0):
        self.w = well
        self.css = dict(css)
        self.state = state
        self.state.cycle_no += 1
        self.cycle_no = self.state.cycle_no
        self.res = CSSReservoir(well, css, state, heat_eff, k_mult, visc_factor, truth)
        self.vf = visc_factor
        self.start = start
        self.rng = rng or np.random.default_rng(0)
        self.stochastic = stochastic
        self.oil_corr = oil_correction
        self.day = 0
        self.inj_days = self.res.inj_days
        self.soak_days = int(css["soak_days"])
        self.prod_day = 0
        self.down_left = 0
        self.q_prev = 80.0
        self.oil_hist = []
        self.cum = {"oil": 0.0, "water": 0.0, "kwh": 0.0, "rod_fail": 0, "unseat": 0,
                    "exp_rod": 0.0, "exp_unseat": 0.0, "float_days": 0, "down_days": 0}
        self.done = False
        self.T_last = self.res.Ts

    @property
    def phase(self):
        if self.day < self.inj_days:
            return "inject"
        if self.day < self.inj_days + self.soak_days:
            return "soak"
        return "produce"

    def observe(self):
        """Current reservoir-side state before today's pump controls are applied."""
        t_since = max(0, self.day - self.inj_days)
        T_avg = self.res.avg_temperature(t_since)
        q_o, q_wc, q_wf, mu_h = self.res.inflow_capacity(T_avg, t_since)
        q_o *= self.oil_corr
        q_in = q_o + q_wc + q_wf
        wc = (q_wc + q_wf) / max(q_in, 1e-6)
        T_in = self.res.T_bg + 0.9 * (T_avg - self.res.T_bg)
        tub = wellbore.tubing_state(T_in, min(self.q_prev, q_in), wc, self.w.pump_depth_m, self.vf)
        return {"T_avg": T_avg, "T_in": T_in, "q_o_cap": q_o, "q_wc_cap": q_wc, "q_wf_cap": q_wf,
                "q_in": q_in, "wc": wc, "mu_res_cp": mu_h, "t_since": t_since,
                "prod_day": self.prod_day, **{k: v for k, v in tub.items() if not k.startswith("profile")},
                "profile_T": tub["profile_T"], "profile_z": tub["profile_z"]}

    def step(self, controller):
        today = self.start + timedelta(days=self.day)
        ph = self.phase
        base = {"date": today.isoformat(), "well_id": self.w.well_id, "cycle": self.cycle_no,
                "day_in_cycle": self.day, "phase": ph}
        if ph in ("inject", "soak"):
            T_now = self.res.Ts if ph == "inject" else self.res.avg_temperature(self.day - self.inj_days)
            rec = {**base, "steam_tpd": C.STEAM_GEN_RATE if ph == "inject" else 0.0,
                   "inj_pressure_mpa": self.css["inj_pressure_mpa"] if ph == "inject" else 0.0,
                   "T_res": T_now, "oil_bbl": 0.0, "water_bbl": 0.0, "liquid_bbl": 0.0, "kwh": 0.0,
                   "spm": 0.0, "stroke_m": 0.0, "down_frac": 0.0, "card_class": "none"}
            self.day += 1
            return rec

        obs = self.observe()
        self.T_last = obs["T_avg"]
        if self.down_left > 0:
            self.down_left -= 1
            self.cum["down_days"] += 1
            rec = {**base, "phase": "down", "T_res": obs["T_avg"], "oil_bbl": 0.0, "water_bbl": 0.0,
                   "liquid_bbl": 0.0, "kwh": 0.0, "spm": 0.0, "stroke_m": 0.0, "down_frac": 0.0,
                   "card_class": "none", "steam_tpd": 0.0, "inj_pressure_mpa": 0.0}
            self.day += 1
            self.prod_day += 1
            self._check_end(0.0)
            return rec

        ctl = controller.decide(self, obs)
        if ctl.get("down"):
            self.down_left = int(ctl["down"]) - 1
            self.cum["down_days"] += 1
            rec = {**base, "phase": "down", "T_res": obs["T_avg"], "oil_bbl": 0.0, "water_bbl": 0.0,
                   "liquid_bbl": 0.0, "kwh": 0.0, "spm": 0.0, "stroke_m": 0.0, "down_frac": 0.0,
                   "card_class": "none", "steam_tpd": 0.0, "inj_pressure_mpa": 0.0,
                   "failure": ctl.get("failure", "")}
            self.day += 1
            self.prod_day += 1
            return rec

        p = srp.srp_eval(ctl["spm"], ctl["stroke_m"], ctl["down_frac"], obs["q_in"], obs["wc"],
                         obs["mu_tubing_cp"], obs["mu_pump_cp"], self.w.pump_depth_m)
        p = {k: float(v) for k, v in p.items()}
        q_liq = p["q_liq"]
        share = q_liq / max(obs["q_in"], 1e-6)
        oil = obs["q_o_cap"] * share
        w_c = obs["q_wc_cap"] * share
        w_f = obs["q_wf_cap"] * share
        self.res.produce(oil / self.oil_corr, w_c, w_f, obs["T_avg"])
        self.q_prev = q_liq
        lam_rod, lam_un = srp.failure_hazards(p["RFI"], p["stress_ratio"], p["impact"])
        lam_rod, lam_un = float(lam_rod), float(lam_un)
        self.cum["exp_rod"] += lam_rod
        self.cum["exp_unseat"] += lam_un
        failure = ""
        if self.stochastic:
            u = self.rng.random()
            if u < lam_rod:
                failure = "rod_part"
            elif u < lam_rod + lam_un:
                failure = "pump_unseat"
        early_hot = self.prod_day < 6 and obs["T_in"] > 175 and self.w.well_id[-1] in "13579"
        state_cls = classify_state({**p, "phase_failure": failure}, early_hot)
        if p["RFI"] > 1.0:
            self.cum["float_days"] += 1
        self.cum["oil"] += oil
        self.cum["water"] += w_c + w_f
        self.cum["kwh"] += p["kwh"]
        rec = {**base, "T_res": obs["T_avg"], "T_pump": obs["T_pump"], "T_wellhead": obs["T_wellhead"],
               "mu_res_cp": obs["mu_res_cp"], "mu_tubing_cp": obs["mu_tubing_cp"],
               "q_in_bbl": obs["q_in"], "oil_bbl": oil, "water_bbl": w_c + w_f, "liquid_bbl": q_liq,
               "wc": obs["wc"], "spm": ctl["spm"], "stroke_m": ctl["stroke_m"], "down_frac": ctl["down_frac"],
               "fillage": p["fillage"], "fill_inflow": p["fill_inflow"], "pump_eff": p["pump_eff"],
               "RFI": p["RFI"], "PPRL_kN": p["PPRL"] / 1000, "MPRL_kN": p["MPRL"] / 1000,
               "stress_ratio": p["stress_ratio"], "kwh": p["kwh"],
               "kwh_per_bbl_oil": p["kwh"] / max(oil, 0.1), "W_rb_kN": p["W_rb"] / 1000,
               "Fo_kN": p["Fo"] / 1000, "c_visc": p["c_visc"], "lam_rod": lam_rod, "lam_unseat": lam_un,
               "card_class": state_cls, "failure": failure, "steam_tpd": 0.0, "inj_pressure_mpa": 0.0,
               "prod_day": self.prod_day}
        if failure:
            self.cum["rod_fail" if failure == "rod_part" else "unseat"] += 1
            self.down_left = (C.DOWNTIME_DAYS_ROD if failure == "rod_part" else C.DOWNTIME_DAYS_UNSEAT)
            controller.on_failure(failure)
        self.day += 1
        self.prod_day += 1
        self._check_end(oil)
        return rec

    def _check_end(self, oil):
        self.oil_hist.append(oil)
        cut = self.css["cutoff_bopd"]
        forced = self.css.get("force_prod_days")
        if forced is not None:
            self.done = self.prod_day >= forced
        elif self.prod_day >= C.MAX_PROD_DAYS:
            self.done = True
        elif self.prod_day > 25 and np.mean(self.oil_hist[-7:]) < cut and self.down_left == 0:
            self.done = True

    def close(self):
        self.res.close_cycle(self.T_last, self.cum["oil"])

    def summary(self):
        steam = self.css["steam_t"]
        oil = self.cum["oil"]
        days = self.day
        exp_fail_cost = (self.cum["exp_rod"] * C.ROD_FAILURE_COST + self.cum["exp_unseat"] * C.PUMP_UNSEAT_COST)
        revenue = oil * C.OIL_PRICE
        cost = steam * C.STEAM_COST + self.cum["kwh"] * C.POWER_COST + exp_fail_cost
        return {"cycle": self.cycle_no, **{k: self.css[k] for k in
                ("steam_t", "quality", "inj_pressure_mpa", "soak_days", "cutoff_bopd")},
                "inj_days": self.inj_days, "prod_days": self.prod_day, "cycle_days": days,
                "cum_oil_bbl": oil, "cum_water_bbl": self.cum["water"],
                "csor": steam / max(oil * C.BBL, 1e-6), "kwh": self.cum["kwh"],
                "kwh_per_bbl": self.cum["kwh"] / max(oil, 1e-6),
                "rod_failures": self.cum["rod_fail"], "pump_unseats": self.cum["unseat"],
                "exp_failures": self.cum["exp_rod"] + self.cum["exp_unseat"],
                "float_days": self.cum["float_days"], "down_days": self.cum["down_days"],
                "peak_T": self.res.Ts, "heated_radius_m": self.res.rh,
                "net_value_inr": revenue - cost, "value_per_day": (revenue - cost) / max(days, 1)}


def run_cycle(sim: CycleSim, controller, keep_profiles=False):
    rows = []
    while not sim.done:
        rows.append(sim.step(controller))
    sim.close()
    return rows, sim.summary()
