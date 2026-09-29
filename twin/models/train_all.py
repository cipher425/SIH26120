"""Train & evaluate all AI models of the digital twin and save artifacts + metrics.

M1  Hybrid reservoir model : calibrated Boberg-Lantz physics x ML residual (GBM)
M2  Production forecaster  : 2-layer LSTM, 21-day window -> 14-day oil forecast
M3  Dyno-card diagnosis    : 1-D CNN on (position, load) card, 7 pump conditions
M4  Reliability model      : GBM on physics features -> P(rod part / unseat in 14 d)
M5  Anomaly detector       : Isolation Forest on physics residuals (measured vs twin)
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from twin import config as C
from twin.core.calibration import load_field, TRAIN_FRAC
from twin.models import deep
from twin.models.features import M4_FEATURES, M5_FEATURES, failure_labels, twin_features

M1_FEATURES = ["cycle", "steam_t", "quality", "inj_pressure_mpa", "soak_days", "cutoff_bopd",
               "log_sim_oil", "k_mult", "heat_eff", "visc_factor_lab", "h_m"]


def mape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs(a - b) / np.maximum(np.abs(a), 1e-6)) * 100)


# ---------------------------------------------------------------- M1
def m1_dataset(cyc, wells, calib):
    rows = []
    for wid, g in cyc.groupby("well_id"):
        cal = calib[wid]
        n_train = cal["n_train_cycles"]
        w = wells.set_index("well_id").loc[wid]
        for i, (_, r) in enumerate(g.sort_values("cycle").iterrows()):
            if i >= len(cal["sim_oil"]) or not r["complete"]:
                continue
            rows.append({**r.to_dict(), "sim_oil": cal["sim_oil"][i], "nominal_oil": cal["nominal_oil"][i],
                         "log_sim_oil": np.log(cal["sim_oil"][i] + 1), "k_mult": cal["k_mult"],
                         "heat_eff": cal["heat_eff"], "visc_factor_lab": w["visc_factor_lab"],
                         "h_m": w["h_m"], "split": "train" if i < n_train else "test"})
    return pd.DataFrame(rows)


def train_m1(cyc, wells, calib):
    df = m1_dataset(cyc, wells, calib)
    df["target"] = np.log((df["cum_oil_bbl"] + 1) / (df["sim_oil"] + 1))
    tr, te = df[df.split == "train"], df[df.split == "test"]
    model = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.04,
                                      subsample=0.8, min_samples_leaf=5, random_state=0)
    model.fit(tr[M1_FEATURES], tr["target"])
    hyb = te["sim_oil"] * np.exp(model.predict(te[M1_FEATURES]))
    metrics = {"n_train": int(len(tr)), "n_test": int(len(te)),
               "mape_nominal_physics": mape(te["cum_oil_bbl"], te["nominal_oil"]),
               "mape_calibrated_physics": mape(te["cum_oil_bbl"], te["sim_oil"]),
               "mape_hybrid": mape(te["cum_oil_bbl"], hyb)}
    imp = dict(zip(M1_FEATURES, np.round(model.feature_importances_, 4).tolist()))
    metrics["feature_importance"] = imp
    metrics["scatter"] = [{"well": w, "cycle": int(c), "actual": round(a, 1), "physics": round(p, 1),
                           "hybrid": round(h, 1)} for w, c, a, p, h in
                          zip(te["well_id"], te["cycle"], te["cum_oil_bbl"], te["sim_oil"], hyb)]
    # refit on all cycles for deployment
    model.fit(df[M1_FEATURES], df["target"])
    joblib.dump(model, C.ART_DIR / "m1_residual.joblib")
    return metrics


# ---------------------------------------------------------------- M2
def m2_windows(daily, wells_list):
    X, Y, meta = [], [], []
    d = daily[(daily.phase == "produce") & daily.well_id.isin(wells_list)]
    for (wid, cyc), g in d.groupby(["well_id", "cycle"]):
        g = g.sort_values("prod_day")
        arr = g[deep.M2_FEATS].to_numpy(dtype=float)
        if len(arr) < deep.WIN + deep.HOR:
            continue
        xt = deep.m2_transform(arr)
        for s in range(0, len(arr) - deep.WIN - deep.HOR + 1, 2):
            x = xt[s:s + deep.WIN]
            y = np.log1p(arr[s + deep.WIN:s + deep.WIN + deep.HOR, 0]) - x[-1, 0]
            X.append(x)
            Y.append(y.astype(np.float32))
            meta.append((wid, cyc, s))
    return np.stack(X), np.stack(Y), meta


def train_m2(daily):
    wl = sorted(daily.well_id.unique())
    tr_w, te_w = wl[:16], wl[16:]
    Xtr, Ytr, _ = m2_windows(daily, tr_w)
    Xte, Yte, meta = m2_windows(daily, te_w)
    model = deep.LSTMForecaster()
    print(f" M2 hybrid LSTM: {len(Xtr)} train windows, {len(Xte)} test windows")
    # LSTM learns the correction on top of the decline-curve physics
    Rtr = Ytr - deep.decline_baseline(Xtr)
    deep.train_loop(model, Xtr, Rtr, epochs=30, lr=2e-3, loss_fn=nn.SmoothL1Loss(beta=0.05))
    P = deep.forecast(model, Xte)
    last = Xte[:, -1, 0][:, None]
    oil_true = np.expm1(Yte + last)
    oil_lstm = np.expm1(P + last)
    oil_dec = np.expm1(deep.decline_baseline(Xte) + last)
    oil_persist = np.expm1(np.repeat(last, deep.HOR, 1))
    m = {"test_wells": te_w, "n_test_windows": int(len(Xte)),
         "mae_lstm": float(np.mean(np.abs(oil_true - oil_lstm))),
         "mae_decline_curve": float(np.mean(np.abs(oil_true - oil_dec))),
         "mae_persistence": float(np.mean(np.abs(oil_true - oil_persist))),
         "mape_lstm": mape(oil_true.ravel(), oil_lstm.ravel()),
         "mape_decline_curve": mape(oil_true.ravel(), oil_dec.ravel()),
         "mae_by_horizon_lstm": np.round(np.mean(np.abs(oil_true - oil_lstm), 0), 3).tolist(),
         "mae_by_horizon_decline": np.round(np.mean(np.abs(oil_true - oil_dec), 0), 3).tolist()}
    torch.save(model.state_dict(), C.ART_DIR / "m2_lstm.pt")
    return m


# ---------------------------------------------------------------- M3
def train_m3():
    z = np.load(C.DATA_DIR / "dyno_cards.npz")
    X, y = z["X"], z["y"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_te = int(0.2 * len(y))
    te, tr = idx[:n_te], idx[n_te:]
    model = deep.CardCNN(len(C.CARD_CLASSES))
    print(f" M3 CNN: {len(tr)} train cards, {len(te)} test cards")
    deep.train_loop(model, X[tr], y[tr].astype(np.int64), epochs=15, lr=3e-3,
                    loss_fn=nn.CrossEntropyLoss())
    with torch.no_grad():
        pred = model(torch.as_tensor(X[te])).argmax(1).numpy()
    m = {"accuracy": float(accuracy_score(y[te], pred)),
         "macro_f1": float(f1_score(y[te], pred, average="macro")),
         "classes": C.CARD_CLASSES,
         "confusion_matrix": confusion_matrix(y[te], pred).tolist()}
    torch.save(model.state_dict(), C.ART_DIR / "m3_cnn.pt")
    return m


# ---------------------------------------------------------------- M4
def train_m4(feat, daily):
    y = failure_labels(feat, daily)
    feat = feat.assign(label=y)
    cut = "2024-07-01"
    tr, te = feat[feat.date < cut], feat[feat.date >= cut]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=0)
    model.fit(tr[M4_FEATURES], tr["label"])
    p = model.predict_proba(te[M4_FEATURES])[:, 1]
    k = max(1, int(0.1 * len(te)))
    top = np.argsort(-p)[:k]
    base = te["label"].mean()
    m = {"train_period": f"< {cut}", "test_period": f">= {cut}", "positive_rate": float(base),
         "roc_auc": float(roc_auc_score(te["label"], p)),
         "pr_auc": float(average_precision_score(te["label"], p)),
         "lift_top10pct": float(te["label"].to_numpy()[top].mean() / max(base, 1e-9))}
    # theoretical ceiling: an oracle that knows the TRUE hidden daily hazard of the next 14 days
    truth = pd.read_parquet(C.DATA_DIR / "truth_daily.parquet")[["date", "well_id", "lam_rod", "lam_unseat"]]
    tt = te[["date", "well_id"]].merge(truth, on=["date", "well_id"], how="left")
    tt["lam"] = (tt["lam_rod"] + tt["lam_unseat"]).fillna(0)
    tt = tt.sort_values(["well_id", "date"])
    tt["fwd"] = tt.groupby("well_id")["lam"].transform(lambda s: s[::-1].rolling(14, min_periods=1).sum()[::-1])
    oracle = tt.set_index(["date", "well_id"]).loc[list(zip(te["date"], te["well_id"])), "fwd"].to_numpy()
    m["oracle_roc_auc"] = float(roc_auc_score(te["label"], oracle))
    m["pct_of_achievable"] = float((m["roc_auc"] - 0.5) / max(m["oracle_roc_auc"] - 0.5, 1e-6) * 100)
    pi = permutation_importance(model, te[M4_FEATURES].iloc[:6000], te["label"].iloc[:6000],
                                scoring="roc_auc", n_repeats=3, random_state=0)
    m["feature_importance"] = dict(zip(M4_FEATURES, np.round(pi.importances_mean, 4).tolist()))
    model.fit(feat[M4_FEATURES], feat["label"])
    joblib.dump(model, C.ART_DIR / "m4_reliability.joblib")
    return m


# ---------------------------------------------------------------- M5
def train_m5(feat):
    truth = pd.read_parquet(C.DATA_DIR / "truth_daily.parquet")[["date", "well_id", "anomaly"]]
    f = feat.merge(truth, on=["date", "well_id"], how="left")
    f["is_anom"] = (f["anomaly"].fillna("") != "").astype(int)
    f = f.reset_index(drop=True)
    X = f[M5_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-30, 30)
    # (a) unsupervised: Isolation Forest on physics residual z-scores (finds unknown faults)
    iso = IsolationForest(n_estimators=400, contamination=0.03, random_state=0)
    iso.fit(X)
    pred_iso = (iso.predict(X) == -1).astype(int)
    # (b) supervised: classifier that names the fault type (trained on labelled events)
    labels = f["anomaly"].fillna("").replace("", "none")
    cut = "2024-07-01"
    trm, tem = (f.date < cut).to_numpy(), (f.date >= cut).to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced",
                                         random_state=0)
    clf.fit(X[trm], labels[trm])
    lp = clf.predict(X[tem])
    yt = labels[tem].to_numpy()
    is_a_true, is_a_pred = (yt != "none").astype(int), (lp != "none").astype(int)
    m = {"iso_precision": float(precision_score(f["is_anom"], pred_iso)),
         "iso_recall": float(recall_score(f["is_anom"], pred_iso)),
         "iso_f1": float(f1_score(f["is_anom"], pred_iso)),
         "precision": float(precision_score(is_a_true, is_a_pred)),
         "recall": float(recall_score(is_a_true, is_a_pred)),
         "f1": float(f1_score(is_a_true, is_a_pred)),
         "type_accuracy_on_detected": float(np.mean(lp[(yt != "none") & (lp != "none")] ==
                                                    yt[(yt != "none") & (lp != "none")])),
         "injected_anomaly_rate": float(f["is_anom"].mean())}
    m["recall_by_type"] = {k: float(np.mean(lp[yt == k] != "none")) for k in sorted(set(yt) - {"none"})}
    clf.fit(X, labels)
    joblib.dump({"iso": iso, "clf": clf}, C.ART_DIR / "m5_anomaly.joblib")
    return m


def main():
    t0 = time.time()
    C.ART_DIR.mkdir(exist_ok=True)
    wells, cyc, daily = load_field()
    calib = json.load(open(C.ART_DIR / "calibration.json"))
    metrics = {}
    print("M1 hybrid reservoir model ...")
    metrics["M1"] = train_m1(cyc, wells, calib)
    print("M2 LSTM forecaster ...")
    metrics["M2"] = train_m2(daily)
    print("M3 CNN card classifier ...")
    metrics["M3"] = train_m3()
    print("Physics features ...")
    feat = twin_features(daily, wells)
    feat.to_parquet(C.ART_DIR / "features_daily.parquet", index=False)
    print("M4 reliability ...")
    metrics["M4"] = train_m4(feat, daily)
    print("M5 anomaly ...")
    metrics["M5"] = train_m5(feat)
    metrics["trained_at"] = time.strftime("%Y-%m-%d %H:%M")
    with open(C.ART_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    for k in ("M1", "M2", "M3", "M4", "M5"):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in metrics[k].items()
                  if not isinstance(b, (list, dict))})
    print("elapsed", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
