"""History matching: fit each well's permeability multiplier and steam heat efficiency
so the coupled physics reproduces observed cycle oil, replaying the recorded SRP
settings and downtime day by day."""
import json
from datetime import date

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

from twin import config as C
from twin.physics.reservoir import WellProps, WellState
from twin.sim.cycle import CycleSim, run_cycle
from twin.sim.controllers import ReplayController

TRAIN_FRAC = 0.7


def load_field():
    wells = pd.read_csv(C.DATA_DIR / "wells.csv")
    cyc = pd.read_csv(C.DATA_DIR / "css_cycles.csv")
    daily = pd.read_csv(C.DATA_DIR / "daily_operations.csv")
    return wells, cyc, daily


def well_from_row(r) -> WellProps:
    d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
    return WellProps(**{k: d[k] for k in WellProps.__dataclass_fields__})


def css_from_row(r, force=True):
    css = {"steam_t": float(r["steam_t"]), "quality": float(r["quality"]),
           "inj_pressure_mpa": float(r["inj_pressure_mpa"]), "soak_days": int(r["soak_days"]),
           "cutoff_bopd": float(r["cutoff_bopd"])}
    if force:
        css["force_prod_days"] = int(r["prod_days"])
    return css


def replay_records(daily_cycle):
    p = daily_cycle[daily_cycle["prod_day"] >= 0]
    rec = {}
    for pd_, ph, spm, s, d in zip(p["prod_day"], p["phase"], p["spm"], p["stroke_m"], p["down_frac"]):
        rec[int(pd_)] = {"down": ph == "down", "spm": spm, "stroke_m": s, "down_frac": d}
    return rec


def replay_history(well, cyc_w, daily_w, k_mult, heat_eff, oil_corr=None, collect=False):
    """Simulate every recorded cycle sequentially. Returns list of simulated cycle oil
    (and optionally daily rows) and the final WellState."""
    state = WellState(p_res_mpa=well.p_res_mpa)
    sims, rows_all = [], []
    groups = dict(tuple(daily_w.groupby("cycle")))
    for _, r in cyc_w.iterrows():
        css = css_from_row(r)
        dc = groups.get(int(r["cycle"]))
        if dc is None:
            break
        corr = 1.0 if oil_corr is None else oil_corr.get(int(r["cycle"]), 1.0)
        sim = CycleSim(well, css, state, date.fromisoformat(r["start_date"]), heat_eff, k_mult,
                       well.visc_factor_lab, oil_correction=corr)
        rows, s = run_cycle(sim, ReplayController(replay_records(dc)))
        sims.append(s["cum_oil_bbl"])
        if collect:
            rows_all.extend(rows)
    return sims, state, rows_all


def calibrate_well(well, cyc_w, daily_w):
    n = len(cyc_w)
    obs = cyc_w["cum_oil_bbl"].to_numpy()
    complete = cyc_w["complete"].to_numpy()
    n_train = max(2, int(round(n * TRAIN_FRAC)))
    mask = np.zeros(n, bool)
    mask[:n_train] = True
    mask &= complete

    def loss(x):
        km, he = float(np.exp(x[0])), float(np.clip(x[1], 0.4, 1.0))
        sim, _, _ = replay_history(well, cyc_w, daily_w, km, he)
        sim = np.asarray(sim)
        return float(np.mean((np.log(sim[mask] + 1) - np.log(obs[mask] + 1)) ** 2))

    grid = [(np.log(km), he) for km in (0.6, 0.8, 1.0, 1.25, 1.55) for he in (0.6, 0.72, 0.84, 0.96)]
    scores = [loss(g) for g in grid]
    x0 = np.array(grid[int(np.argmin(scores))])
    simplex = np.array([x0, x0 + [0.12, 0.0], x0 + [0.0, 0.06]])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxfev": 45, "xatol": 0.005, "fatol": 1e-5, "initial_simplex": simplex})
    km, he = float(np.exp(res.x[0])), float(np.clip(res.x[1], 0.4, 1.0))
    sim, state, _ = replay_history(well, cyc_w, daily_w, km, he)
    nominal, _, _ = replay_history(well, cyc_w, daily_w, 1.0, 0.85)
    return {"well_id": well.well_id, "k_mult": km, "heat_eff": he, "loss": float(res.fun),
            "n_train_cycles": int(n_train), "sim_oil": [float(v) for v in sim],
            "nominal_oil": [float(v) for v in nominal],
            "final_state": state.__dict__}


def calibrate_field(n_jobs=-1):
    wells, cyc, daily = load_field()
    jobs = []
    for _, wr in wells.iterrows():
        w = well_from_row(wr)
        jobs.append(delayed(calibrate_well)(w, cyc[cyc.well_id == w.well_id].reset_index(drop=True),
                                           daily[daily.well_id == w.well_id]))
    out = Parallel(n_jobs=n_jobs)(jobs)
    calib = {o["well_id"]: o for o in out}
    C.ART_DIR.mkdir(exist_ok=True)
    with open(C.ART_DIR / "calibration.json", "w") as f:
        json.dump(calib, f)
    return calib


if __name__ == "__main__":
    import time
    t = time.time()
    cal = calibrate_field()
    truth = json.load(open(C.DATA_DIR / "truth_params.json"))
    for k, v in cal.items():
        print(k, f"k_mult={v['k_mult']:.2f} (true {truth[k]['k_mult']:.2f})  heat_eff={v['heat_eff']:.2f} "
                 f"(true {truth[k]['heat_eff']:.2f})  loss={v['loss']:.4f}")
    print("elapsed", round(time.time() - t, 1))
