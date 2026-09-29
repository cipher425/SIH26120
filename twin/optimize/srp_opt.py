"""Real-time SRP optimiser.

Exhaustive vectorised search over SPM x VFD down-stroke fraction (x stroke length at
cycle start). Objective = oil revenue - power cost - expected failure cost, with hard
penalties on rod floating (RFI) and rod over-stress (Goodman). Failure probability can
come from the learned reliability model (M4) or the physics hazard fallback.
"""
import numpy as np

from twin import config as C
from twin.physics import srp

SPM_GRID = np.round(np.arange(C.SPM_MIN, C.SPM_MAX + 1e-9, 0.25), 2)


class SRPOptimizer:
    def __init__(self, well, risk_model=None):
        self.w = well
        self.risk_model = risk_model

    def evaluate(self, obs, strokes=None, spm_grid=SPM_GRID, down_grid=C.DOWN_FRAC_OPTIONS):
        strokes = strokes or [self.w.stroke_m]
        S, D, V = np.meshgrid(np.asarray(strokes), np.asarray(down_grid), spm_grid, indexing="ij")
        p = srp.srp_eval(V, S, D, obs["q_in"], obs["wc"], obs["mu_tubing_cp"],
                         obs["mu_pump_cp"], self.w.pump_depth_m)
        lam_rod, lam_un = srp.failure_hazards(p["RFI"], p["stress_ratio"], p["impact"])
        oil = p["q_liq"] * (1 - obs["wc"])
        if self.risk_model is not None:
            risk = self.risk_model.predict_grid(p, obs, V, D)
            lam_rod = lam_rod * 0.5 + 0.5 * risk / 14.0
        value = (oil * C.OIL_PRICE - p["kwh"] * C.POWER_COST
                 - lam_rod * (C.ROD_FAILURE_COST + C.DOWNTIME_DAYS_ROD * oil * C.OIL_PRICE)
                 - lam_un * (C.PUMP_UNSEAT_COST + C.DOWNTIME_DAYS_UNSEAT * oil * C.OIL_PRICE))
        penalty = (np.clip(p["RFI"] - 0.9, 0, None) * 5e4
                   + np.clip(p["stress_ratio"] - 0.95, 0, None) * 1e5
                   + np.clip(0.75 - p["fillage"], 0, None) * 2e4)
        score = value - penalty
        return {"S": S, "D": D, "V": V, "p": p, "oil": oil, "value": value, "score": score,
                "lam_rod": lam_rod, "lam_un": lam_un}

    def optimize(self, obs, strokes=None):
        ev = self.evaluate(obs, strokes)
        i = np.unravel_index(np.argmax(ev["score"]), ev["score"].shape)
        p = ev["p"]
        return {"spm": float(ev["V"][i]), "stroke_m": float(ev["S"][i]), "down_frac": float(ev["D"][i]),
                "oil_bbl": float(ev["oil"][i]), "liquid_bbl": float(p["q_liq"][i]),
                "kwh": float(p["kwh"][i]), "RFI": float(p["RFI"][i]), "fillage": float(p["fillage"][i]),
                "stress_ratio": float(p["stress_ratio"][i]), "value_inr": float(ev["value"][i]),
                "lam_rod": float(ev["lam_rod"][i]), "lam_unseat": float(ev["lam_un"][i])}

    def heatmap(self, obs, stroke):
        ev = self.evaluate(obs, [stroke])
        return {"spm": SPM_GRID.tolist(), "down_frac": list(C.DOWN_FRAC_OPTIONS),
                "value": np.round(ev["value"][0] / 1000, 2).tolist(),
                "RFI": np.round(ev["p"]["RFI"][0], 3).tolist(),
                "oil": np.round(ev["oil"][0], 1).tolist(),
                "kwh_per_bbl": np.round(ev["p"]["kwh"][0] / np.maximum(ev["oil"][0], 0.1), 2).tolist(),
                "fillage": np.round(ev["p"]["fillage"][0], 3).tolist()}


def explain(rec, current, obs):
    """Plain-language rationale for a recommendation."""
    msgs = []
    if rec["spm"] < current.get("spm", rec["spm"]) - 0.2:
        msgs.append(f"Reduce SPM {current['spm']:.2f} -> {rec['spm']:.2f}: inflow ({obs['q_in']:.0f} bbl/d) "
                    f"is below pump capacity; slower pumping cuts pound & energy.")
    elif rec["spm"] > current.get("spm", rec["spm"]) + 0.2:
        msgs.append(f"Increase SPM {current['spm']:.2f} -> {rec['spm']:.2f}: hot zone inflow "
                    f"({obs['q_in']:.0f} bbl/d) exceeds pump capacity - oil is being left behind.")
    if rec["down_frac"] > 0.5:
        msgs.append(f"Set VFD down-stroke to {rec['down_frac']*100:.0f}% of cycle: tubing viscosity "
                    f"{obs['mu_tubing_cp']:.0f} cP - slower down-stroke keeps Rod-Float-Index at {rec['RFI']:.2f} (<1).")
    if current.get("RFI", 0) > 1.0:
        msgs.append(f"Rod floating detected (RFI {current['RFI']:.2f}); impact loading raises rod-failure risk.")
    if not msgs:
        msgs.append("Current settings are near optimal for present reservoir temperature and inflow.")
    return msgs
