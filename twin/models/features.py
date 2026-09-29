"""Physics-informed feature engineering.

The twin converts raw SCADA measurements into physical quantities that are not
directly measured - tubing viscosity (Ramey + PVT), Rod-Float-Index, Goodman stress
ratio, expected loads/energy - and residuals between measurement and physics.
These features let the ML models generalise to operating settings never seen in
history (e.g. asymmetric VFD strokes)."""
import numpy as np
import pandas as pd

from twin.physics import srp, wellbore

M4_FEATURES = ["RFI_est", "stress_est", "MPRL_norm", "PPRL_kN", "spm", "down_frac", "fillage",
               "log_mu_tub", "rfi_7d", "float_7d", "float_30d", "impact_30d", "haz_phys", "haz_phys_14d",
               "days_since_failure"]
M5_RAW = ["kwh_ratio", "pprl_ratio", "mprl_resid", "liquid_ratio", "twh_resid", "tpump_jump"]
M5_FEATURES = [f"z_{c}" for c in M5_RAW] + ["twh_frozen"]


def twin_features(daily: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    d = daily[daily["phase"] == "produce"].copy()
    wmap = wells.set_index("well_id")
    mu_t, mu_p, twh = np.empty(len(d)), np.empty(len(d)), np.empty(len(d))
    vf = d["well_id"].map(wmap["visc_factor_lab"]).to_numpy()
    depth = d["well_id"].map(wmap["pump_depth_m"]).to_numpy()
    for i, (tp, q, wc) in enumerate(zip(d["T_pump"].to_numpy(), d["liquid_bbl"].to_numpy(), d["wc"].to_numpy())):
        ts = wellbore.tubing_state(tp, q, wc, depth[i], vf[i])
        mu_t[i], mu_p[i], twh[i] = ts["mu_tubing_cp"], ts["mu_pump_cp"], ts["T_wellhead"]
    p = srp.srp_eval(d["spm"].to_numpy(), d["stroke_m"].to_numpy(), d["down_frac"].to_numpy(),
                     d["liquid_bbl"].to_numpy(), d["wc"].to_numpy(), mu_t, mu_p, depth)
    d["mu_tub_est"] = mu_t
    d["log_mu_tub"] = np.log(mu_t)
    d["RFI_est"] = p["RFI"]
    d["stress_est"] = p["stress_ratio"]
    d["W_rb_kN"] = p["W_rb"] / 1000
    d["MPRL_norm"] = d["MPRL_kN"] / d["W_rb_kN"]
    d["kwh_ratio"] = d["kwh"] / np.maximum(p["kwh"], 1)
    d["pprl_ratio"] = d["PPRL_kN"] / (p["PPRL"] / 1000)
    d["mprl_resid"] = (d["MPRL_kN"] - p["MPRL"] / 1000) / d["W_rb_kN"]
    cap = p["disp_bbl"] * np.clip((d["stroke_m"].to_numpy() - p["stretch"]) / d["stroke_m"].to_numpy(), 0.3, 1)
    d["liquid_ratio"] = d["liquid_bbl"] / np.maximum(cap * d["fillage"].clip(lower=0.05), 1)
    d["twh_resid"] = d["T_wellhead"] - twh

    lam_r, lam_u = srp.failure_hazards(p["RFI"], p["stress_ratio"], p["impact"])
    d["haz_phys"] = lam_r + lam_u
    d["impact"] = p["impact"]

    d = d.sort_values(["well_id", "date"])
    g = d.groupby("well_id")
    d["rfi_7d"] = g["RFI_est"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["float_7d"] = g["RFI_est"].transform(lambda s: (s > 1).astype(float).rolling(7, min_periods=1).sum())
    d["float_30d"] = g["RFI_est"].transform(lambda s: (s > 1).astype(float).rolling(30, min_periods=1).sum())
    d["impact_30d"] = g["impact"].transform(lambda s: s.rolling(30, min_periods=1).sum())
    d["haz_phys_14d"] = g["haz_phys"].transform(lambda s: s.rolling(14, min_periods=1).sum())
    d["tpump_jump"] = g["T_pump"].diff().fillna(0)
    d["twh_frozen"] = (g["T_wellhead"].diff().abs() < 1e-6).astype(float)
    # robust z-score of each physics residual against the well's own trailing 21-day behaviour
    for c in M5_RAW:
        med = g[c].transform(lambda s: s.shift(1).rolling(21, min_periods=5).median())
        mad = g[c].transform(lambda s: (s.shift(1) - s.shift(1).rolling(21, min_periods=5).median())
                             .abs().rolling(21, min_periods=5).median())
        d[f"z_{c}"] = ((d[c] - med) / (1.4826 * mad + 1e-3)).fillna(0).clip(-30, 30)
    # days since last failure
    dates = pd.to_datetime(daily["date"])
    fail = daily.assign(_d=dates)[daily["failure"].fillna("") != ""][["well_id", "_d"]]
    d["_d"] = pd.to_datetime(d["date"])
    dsf = np.full(len(d), 365.0)
    for wid, grp in d.groupby("well_id"):
        fd = np.sort(fail.loc[fail.well_id == wid, "_d"].to_numpy())
        if len(fd):
            idx = np.searchsorted(fd, grp["_d"].to_numpy(), side="right") - 1
            last = np.where(idx >= 0, fd[np.clip(idx, 0, None)], np.datetime64("NaT"))
            delta = (grp["_d"].to_numpy() - last) / np.timedelta64(1, "D")
            dsf[d.index.get_indexer(grp.index)] = np.where(np.isnan(delta), 365, np.minimum(delta, 365))
    d["days_since_failure"] = dsf
    return d.drop(columns=["_d"])


def failure_labels(d: pd.DataFrame, daily: pd.DataFrame, horizon=14):
    """1 if a rod part / pump unseat happens within `horizon` days after this day."""
    fail = daily[daily["failure"].fillna("") != ""][["well_id", "date"]].copy()
    fail["date"] = pd.to_datetime(fail["date"])
    y = np.zeros(len(d), dtype=int)
    dd = pd.to_datetime(d["date"]).to_numpy()
    for wid, idx in d.groupby("well_id").indices.items():
        fd = np.sort(fail.loc[fail.well_id == wid, "date"].to_numpy())
        if not len(fd):
            continue
        t = dd[idx]
        j = np.searchsorted(fd, t, side="right")
        nxt = np.where(j < len(fd), fd[np.clip(j, 0, len(fd) - 1)], np.datetime64("NaT"))
        gap = (nxt - t) / np.timedelta64(1, "D")
        y[idx] = ((gap > 0) & (gap <= horizon)).astype(int)
    return y
