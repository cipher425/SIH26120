"""Synthetic Baghewala-like field history generator.

Produces the same data families the problem statement lists as available:
production history, CSS cycle records, steam injection parameters, VFD/SRP data,
rod-failure & pump-unseating history, well completion/reservoir data, PVT.

Hidden 'truth' parameters (conformance decay, steam channeling, true viscosity,
permeability multipliers) make the field behave differently from the nominal physics
so the digital twin must calibrate + learn - as it would on the real field.
"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from twin import config as C
from twin.physics import fluid, cards
from twin.physics.reservoir import WellProps, TruthParams, WellState
from twin.sim.cycle import CycleSim, run_cycle
from twin.sim.controllers import HistoricalController

OBS_COLS = ["date", "well_id", "cycle", "day_in_cycle", "prod_day", "phase", "steam_tpd",
            "inj_pressure_mpa", "oil_bbl", "water_bbl", "liquid_bbl", "wc", "T_wellhead", "T_pump",
            "spm", "stroke_m", "down_frac", "fillage", "PPRL_kN", "MPRL_kN", "kwh", "thp_mpa",
            "chp_mpa", "card_class", "failure"]
TRUTH_COLS = ["date", "well_id", "T_res", "mu_res_cp", "mu_tubing_cp", "q_in_bbl", "RFI",
              "stress_ratio", "lam_rod", "lam_unseat", "fill_inflow", "anomaly"]


def make_wells(rng):
    wells, truths = [], {}
    for i in range(C.N_WELLS):
        wid = f"BGW-{i + 1:02d}"
        depth = rng.uniform(1000, 1150)
        w = WellProps(
            well_id=wid, lat=27.95 + rng.normal(0, 0.012), lon=72.05 + rng.normal(0, 0.015),
            depth_m=round(depth, 1), pump_depth_m=round(depth - 60, 1), h_m=round(rng.uniform(10, 20), 1),
            k_md=round(float(np.exp(rng.normal(np.log(650), 0.25))), 0), phi=round(rng.uniform(0.25, 0.30), 3),
            so_init=round(rng.uniform(0.70, 0.78), 3), swc=0.2, sor=0.25,
            p_res_mpa=round(rng.uniform(3.2, 3.8), 2), re_m=100.0, rw_m=0.09,
            visc_factor_lab=round(rng.uniform(0.9, 1.5), 3),
            stroke_m=float(rng.choice(C.STROKE_OPTIONS, p=[0.25, 0.55, 0.20])),
            wc_base=round(rng.uniform(0.10, 0.25), 3))
        t = TruthParams(k_mult=float(np.exp(rng.normal(0, 0.2))), heat_eff=float(rng.uniform(0.78, 0.93)),
                        conformance_decay=float(rng.uniform(0.01, 0.05)),
                        visc_factor=float(w.visc_factor_lab * (1 + rng.normal(0, 0.08))),
                        channel_p_mpa=float(rng.uniform(6.0, 7.0)), channel_loss=0.15)
        wells.append(w)
        truths[wid] = t
    return wells, truths


def historical_css(rng):
    return {"steam_t": float(np.clip(round(rng.normal(1800, 350) / 50) * 50, 1000, 3000)),
            "quality": round(float(rng.uniform(0.62, 0.78)), 2),
            "inj_pressure_mpa": round(float(np.clip(rng.normal(5.6, 0.8), 4.0, 7.8)), 2),
            "soak_days": int(rng.integers(4, 13)),
            "cutoff_bopd": round(float(rng.uniform(8, 14)), 1)}


def inject_anomalies(df, rng, rate=0.025):
    df["anomaly"] = ""
    prod_idx = np.flatnonzero((df["phase"] == "produce").to_numpy())
    n_ep = int(len(prod_idx) * rate / 2.5)
    kinds = ["load_cell_drift", "tubing_leak", "motor_fault", "sensor_freeze"]
    for _ in range(n_ep):
        s = int(rng.choice(prod_idx))
        L = int(rng.integers(1, 5))
        idx = [j for j in range(s, s + L) if j in set(prod_idx)]
        if not idx:
            continue
        k = kinds[rng.integers(0, 4)]
        ix = df.index[idx]
        df.loc[ix, "anomaly"] = k
        if k == "load_cell_drift":
            df.loc[ix, ["PPRL_kN", "MPRL_kN"]] *= 1.22
        elif k == "tubing_leak":
            df.loc[ix, ["oil_bbl", "water_bbl", "liquid_bbl"]] *= 0.5
            df.loc[ix, "thp_mpa"] -= 0.25
        elif k == "motor_fault":
            df.loc[ix, "kwh"] *= 1.5
        else:
            df.loc[ix, "T_wellhead"] = df.loc[ix[0] - 1, "T_wellhead"] if ix[0] - 1 in df.index else 40.0
            df.loc[ix, "T_pump"] = df.loc[ix, "T_pump"] + 25
    return df


def add_sensor_noise(df, rng):
    prod = df["phase"] == "produce"
    n = prod.sum()
    for col, s in (("oil_bbl", 0.04), ("water_bbl", 0.05), ("PPRL_kN", 0.015), ("MPRL_kN", 0.03), ("kwh", 0.03)):
        df.loc[prod, col] = df.loc[prod, col] * (1 + rng.normal(0, s, n))
    for col in ("T_wellhead", "T_pump"):
        df.loc[prod, col] = df.loc[prod, col] + rng.normal(0, 1.2, n)
    df.loc[prod, "liquid_bbl"] = df.loc[prod, "oil_bbl"] + df.loc[prod, "water_bbl"]
    df.loc[prod, "wc"] = df.loc[prod, "water_bbl"] / df.loc[prod, "liquid_bbl"].clip(lower=0.1)
    df.loc[prod, "thp_mpa"] = C.THP_MPA + rng.normal(0, 0.03, n)
    df.loc[prod, "chp_mpa"] = 0.25 + rng.normal(0, 0.03, n)
    df.loc[prod, "fillage"] = (df.loc[prod, "fillage"] + rng.normal(0, 0.02, n)).clip(0, 1)
    return df


def simulate_well(w, truth, rng, end):
    state = WellState(p_res_mpa=w.p_res_mpa)
    start = date.fromisoformat(C.FIELD_START) + timedelta(days=int(rng.integers(0, 540)))
    ctl = HistoricalController(w.stroke_m, spm_start=float(rng.integers(6, 9)), rng=rng)
    rows, cyc = [], []
    while start < end:
        css = historical_css(rng)
        sim = CycleSim(w, css, state, start, truth.heat_eff, truth.k_mult, truth.visc_factor,
                       truth=truth, rng=rng, stochastic=True)
        r, s = run_cycle(sim, ctl)
        r = [x for x in r if date.fromisoformat(x["date"]) <= end]
        rows.extend(r)
        s.update({"well_id": w.well_id, "start_date": start.isoformat(),
                  "end_date": (start + timedelta(days=s["cycle_days"] - 1)).isoformat(),
                  "complete": date.fromisoformat(r[-1]["date"]) < end if r else False})
        cyc.append(s)
        start = start + timedelta(days=s["cycle_days"] + int(rng.integers(5, 25)))
    return rows, cyc


def make_card_dataset(rng, n_per_class=1100):
    X, y = [], []
    L = 980.0
    for ci, cls in enumerate(C.CARD_CLASSES):
        k = 0
        while k < n_per_class:
            stroke = float(rng.choice(C.STROKE_OPTIONS))
            spm = rng.uniform(2, 8.5)
            d = rng.uniform(0.5, 0.65)
            wc = rng.uniform(0.1, 0.8)
            rho = fluid.mix_density(wc)
            W_rb = C.ROD_MASS_PER_M * L * C.G * (1 - rho / C.RHO_STEEL) * rng.uniform(0.9, 1.1)
            Fo = rho * C.G * (L - 40) * np.pi * C.PLUNGER_DIAM ** 2 / 4 * rng.uniform(0.85, 1.15)
            mu = float(np.exp(rng.uniform(np.log(5), np.log(3000)))) / 1000
            c_visc = 16.0 * mu * L
            v_dn = np.pi * stroke * spm / (120 * d)
            if cls != "rod_float" and c_visc * v_dn > 0.8 * W_rb:
                continue
            card = cards.synth_card(cls, stroke, spm, d, W_rb, Fo, c_visc, fillage=1.0, rng=rng,
                                    noise=rng.uniform(0.005, 0.025))
            X.append(cards.normalise_card(card["position"], card["load"], stroke, W_rb, Fo))
            y.append(ci)
            k += 1
    return np.stack(X), np.array(y)


def main(seed=42):
    rng = np.random.default_rng(seed)
    C.DATA_DIR.mkdir(exist_ok=True)
    end = date.fromisoformat(C.FIELD_END)
    wells, truths = make_wells(rng)
    all_rows, all_cyc = [], []
    for w in wells:
        r, c = simulate_well(w, truths[w.well_id], rng, end)
        all_rows.extend(r)
        all_cyc.extend(c)
    df = pd.DataFrame(all_rows)
    df["thp_mpa"] = np.where(df["phase"] == "produce", C.THP_MPA, 0.0)
    df["chp_mpa"] = 0.0
    df["failure"] = df["failure"].fillna("")
    df = df.reset_index(drop=True)
    df = add_sensor_noise(df, rng)
    df = inject_anomalies(df, rng)
    df["prod_day"] = df["prod_day"].fillna(-1).astype(int)

    obs = df[[c for c in OBS_COLS if c in df.columns]].copy()
    truth = df[[c for c in TRUTH_COLS if c in df.columns]].copy()
    num = obs.select_dtypes("number").columns
    obs[num] = obs[num].round(4)
    obs.to_csv(C.DATA_DIR / "daily_operations.csv", index=False)
    truth.to_parquet(C.DATA_DIR / "truth_daily.parquet", index=False)

    cyc = pd.DataFrame(all_cyc)
    cols = ["well_id", "cycle", "start_date", "end_date", "complete", "steam_t", "quality",
            "inj_pressure_mpa", "inj_days", "soak_days", "cutoff_bopd", "prod_days", "cycle_days",
            "cum_oil_bbl", "cum_water_bbl", "csor", "kwh", "kwh_per_bbl", "rod_failures",
            "pump_unseats", "float_days", "down_days"]
    cyc[cols].round(3).to_csv(C.DATA_DIR / "css_cycles.csv", index=False)

    pd.DataFrame([w.to_dict() for w in wells]).to_csv(C.DATA_DIR / "wells.csv", index=False)
    fails = obs[obs["failure"] != ""].merge(truth[["date", "well_id", "RFI", "stress_ratio"]],
                                            on=["date", "well_id"])
    fails["probable_cause"] = np.where(fails["RFI"] > 1.0, "rod floating / impact loading",
                                       np.where(fails["stress_ratio"] > 0.9, "rod fatigue (Goodman overload)",
                                                "wear / random"))
    fails["downtime_days"] = np.where(fails["failure"] == "rod_part", C.DOWNTIME_DAYS_ROD,
                                      C.DOWNTIME_DAYS_UNSEAT)
    fails[["date", "well_id", "cycle", "failure", "probable_cause", "downtime_days", "spm",
           "stroke_m"]].to_csv(C.DATA_DIR / "failures.csv", index=False)
    pvt = []
    for w in wells:
        for r in fluid.pvt_table(truths[w.well_id].visc_factor):
            pvt.append({"well_id": w.well_id, **r})
    pd.DataFrame(pvt).to_csv(C.DATA_DIR / "pvt.csv", index=False)
    with open(C.DATA_DIR / "truth_params.json", "w") as f:
        json.dump({k: v.__dict__ for k, v in truths.items()}, f, indent=1)

    X, y = make_card_dataset(rng)
    np.savez_compressed(C.DATA_DIR / "dyno_cards.npz", X=X, y=y, classes=np.array(C.CARD_CLASSES))

    print(f"wells={len(wells)} daily_rows={len(obs)} cycles={len(cyc)} failures={len(fails)} "
          f"cards={len(y)}")
    print(cyc.groupby("well_id")[["cum_oil_bbl", "csor"]].mean().describe().round(2))
    print(fails["failure"].value_counts())


if __name__ == "__main__":
    main()
