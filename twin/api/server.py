"""FastAPI backend for the ThermoLift Twin dashboard.

    .venv\\Scripts\\python -m uvicorn twin.api.server:app --port 8000
"""
import json
import threading
from datetime import date, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from twin import config as C
from twin.core.twin import FieldTwin, LiveSession
from twin.optimize.srp_opt import SRPOptimizer, explain
from twin.physics import fluid
from twin.sim.controllers import HistoricalController
from twin.sim.cycle import CycleSim

app = FastAPI(title="ThermoLift Twin API", version="1.0")


@app.middleware("http")
async def no_cache(request, call_next):
    """Always serve the latest dashboard files (avoids stale HTML/JS after updates)."""
    resp = await call_next(request)
    if not request.url.path.startswith("/vendor/"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
TW = FieldTwin()
LOCK = threading.Lock()
LIVE: dict = {}
DEMO_WELL = "BGW-09"


def _j(o):
    """Make numpy / pandas objects JSON-safe."""
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _j(o.tolist())
    return o


def _check(wid):
    if wid not in TW._props:
        raise HTTPException(404, f"unknown well {wid}")


# ----------------------------------------------------------------- overview
@app.get("/api/overview")
def overview():
    d = TW.daily
    last_date = d.date.max()
    d30 = d[d.date > (pd.to_datetime(last_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")]
    d365 = d[d.date > (pd.to_datetime(last_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")]
    oil30 = d30.oil_bbl.sum() / 30
    steam365 = d365.steam_tpd.sum()
    oil365 = d365.oil_bbl.sum()
    last = d[d.date == last_date]
    f = TW.feat
    f90 = f[f.date > (pd.to_datetime(last_date) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")]
    risk = TW.risk_table()
    kpi = {"as_of": last_date, "field_oil_bopd": oil30, "wells": len(TW.wells),
           "wells_producing": int((last.phase == "produce").sum()),
           "wells_steaming": int(last.phase.isin(["inject", "soak"]).sum()),
           "csor_12m": steam365 / max(oil365 * C.BBL, 1), "kwh_per_bbl_30d": d30.kwh.sum() / max(d30.oil_bbl.sum(), 1),
           "failures_12m": int((d365.failure.fillna("") != "").sum()),
           "rod_float_pct_90d": float((f90.RFI_est > 1).mean() * 100),
           "fluid_pound_pct_90d": float((f90.card_class == "fluid_pound").mean() * 100),
           "high_risk_wells": int(sum(r["risk_14d"] > 0.15 for r in risk))}
    d = d.assign(month=d.date.str[:7])
    m = d.groupby("month").agg(oil=("oil_bbl", "sum"), steam=("steam_tpd", "sum"), kwh=("kwh", "sum"),
                               days=("date", "nunique")).reset_index()
    m["bopd"] = m.oil / m.days
    fails = d[d.failure.fillna("") != ""].groupby("month").size()
    m["failures"] = m.month.map(fails).fillna(0)
    wells = []
    for _, w in TW.wells.iterrows():
        r = last[last.well_id == w.well_id]
        rk = next((x for x in risk if x["well_id"] == w.well_id), {})
        cy = TW.cyc[TW.cyc.well_id == w.well_id]
        wells.append({"well_id": w.well_id, "lat": w.lat, "lon": w.lon,
                      "phase": r.phase.iloc[0] if len(r) else "idle",
                      "oil_bopd": float(r.oil_bbl.iloc[0]) if len(r) else 0.0,
                      "spm": float(r.spm.iloc[0]) if len(r) else 0.0,
                      "cycle": int(cy.cycle.max()), "csor_avg": float(cy.csor.mean()),
                      "risk_14d": rk.get("risk_14d", 0), "RFI": rk.get("RFI", 0),
                      "failures_total": int(cy.rod_failures.sum() + cy.pump_unseats.sum())})
    abl = json.load(open(C.ART_DIR / "field_ablation.json"))
    return _j({"kpi": kpi, "monthly": m.to_dict("records"), "wells": wells, "ablation": abl})


# ----------------------------------------------------------------- well detail
@app.get("/api/well/{wid}")
def well(wid: str):
    _check(wid)
    cyc = TW.cyc[TW.cyc.well_id == wid]
    cal = TW.calib[wid]
    props = TW.props(wid).to_dict()
    pvt = fluid.pvt_table(props["visc_factor_lab"])
    cyc_out = cyc.to_dict("records")
    for i, c in enumerate(cyc_out):
        c["twin_oil"] = cal["sim_oil"][i] if i < len(cal["sim_oil"]) else None
    return _j({"props": props, "calibration": {k: cal[k] for k in ("k_mult", "heat_eff", "loss", "n_train_cycles")},
               "cycles": cyc_out, "pvt": pvt, "historical_css": TW.historical_css(wid)})


@app.get("/api/well/{wid}/daily")
def well_daily(wid: str, cycle: int = None):
    _check(wid)
    d = TW.daily[TW.daily.well_id == wid]
    cycle = cycle or int(d.cycle.max())
    g = d[d.cycle == cycle]
    f = TW.feat[(TW.feat.well_id == wid) & (TW.feat.cycle == cycle)][["date", "RFI_est", "mu_tub_est", "stress_est"]]
    g = g.merge(f, on="date", how="left")
    return _j({"cycle": cycle, "rows": g.fillna(0).to_dict("records")})


@app.get("/api/well/{wid}/forecast")
def well_forecast(wid: str):
    _check(wid)
    return _j(TW.forecast_oil(wid))


# ----------------------------------------------------------------- CSS optimisation
@app.get("/api/css/{wid}")
def css(wid: str):
    _check(wid)
    if wid not in TW.css_cache:
        with LOCK:
            TW.css_cache[wid] = TW.scenarios(wid, n_trials=30)
    return _j(TW.css_cache[wid])


@app.post("/api/css/{wid}/optimize")
def css_opt(wid: str, trials: int = 40):
    _check(wid)
    with LOCK:
        TW.css_cache[wid] = TW.scenarios(wid, n_trials=min(max(trials, 10), 150))
    return _j(TW.css_cache[wid])


class WhatIf(BaseModel):
    steam_t: float
    quality: float
    inj_pressure_mpa: float
    soak_days: int
    cutoff_bopd: float
    srp_mode: str = "ai"


@app.post("/api/whatif/{wid}")
def whatif(wid: str, q: WhatIf):
    _check(wid)
    css_ = q.model_dump()
    mode = css_.pop("srp_mode")
    rows, s = TW.simulate(wid, css_, srp_mode=mode)
    curve = [{"d": r["day_in_cycle"], "phase": r["phase"], "oil": r["oil_bbl"], "T": r["T_res"], "spm": r["spm"],
              "rfi": r.get("RFI", 0), "kwh": r["kwh"]} for r in rows]
    return _j({"summary": s, "curve": curve})


# ----------------------------------------------------------------- SRP optimiser view
@app.get("/api/srp/{wid}")
def srp_view(wid: str, prod_day: int = 60, stroke: float = None):
    _check(wid)
    w = TW.props(wid)
    cal = TW.calib[wid]
    css_ = TW.historical_css(wid)
    sim = CycleSim(w, css_, TW.state_now(wid), date.fromisoformat(C.FIELD_END) + timedelta(days=1),
                   cal["heat_eff"], cal["k_mult"], w.visc_factor_lab)
    ctl = HistoricalController(w.stroke_m, spm_start=7.0)
    last = None
    while not sim.done and sim.prod_day < prod_day:
        last = sim.step(ctl)
    obs = sim.observe()
    opt = SRPOptimizer(w, risk_model=TW.reliability)
    stroke = stroke or w.stroke_m
    best = opt.optimize(obs, strokes=[stroke])
    best_any = opt.optimize(obs, strokes=C.STROKE_OPTIONS)
    cur = {"spm": last["spm"] if last else 7.0, "RFI": last.get("RFI", 0) if last else 0}
    # 45-day model-predictive plan: optimal settings day by day as the reservoir cools
    from twin.sim.controllers import AIController
    import copy
    sim2 = copy.deepcopy(sim)
    ai = AIController(w, fixed_stroke=stroke)
    plan = []
    for _ in range(45):
        if sim2.done:
            break
        r = sim2.step(ai)
        if r["phase"] == "produce":
            plan.append({"prod_day": r["prod_day"], "spm": r["spm"], "down_frac": r["down_frac"],
                         "T": r["T_res"], "mu": r["mu_tubing_cp"], "rfi": r["RFI"], "oil": r["oil_bbl"]})
    obs_out = {k: v for k, v in obs.items() if not k.startswith("profile")}
    return _j({"obs": obs_out, "current": cur, "best": best, "best_any_stroke": best_any,
               "heatmap": opt.heatmap(obs, stroke), "reasons": explain(best, cur, obs), "plan": plan,
               "profile": {"z": obs["profile_z"], "T": obs["profile_T"]}})


# ----------------------------------------------------------------- live closed loop
@app.post("/api/live/{wid}/reset")
def live_reset(wid: str, plan: str = "historical", seed: int = 0):
    _check(wid)
    css_ = None
    if plan == "optimized" and wid in TW.css_cache:
        css_ = TW.css_cache[wid]["scenarios"]["D_joint_ai"]["css"]
    with LOCK:
        LIVE[wid] = LiveSession(TW, wid, css=css_, seed=seed)
    s = LIVE[wid]
    return _j({"well_id": wid, "css": s.css, "inj_days": s.sim.inj_days, "soak_days": s.sim.soak_days})


@app.get("/api/live/{wid}/tick")
def live_tick(wid: str, n: int = 1, mode: str = "manual"):
    _check(wid)
    with LOCK:
        if wid not in LIVE:
            LIVE[wid] = LiveSession(TW, wid)
        s = LIVE[wid]
        s.mode = "ai" if mode == "ai" else "manual"
        out = None
        last_prod = None
        for _ in range(max(1, min(n, 60))):
            out = s.tick()
            if out.get("done"):
                break
            if "card" in out:
                last_prod = out
        if last_prod is not None and "card" not in out:
            out = {**last_prod, "record": out.get("record", last_prod["record"]), "cum": out.get("cum")}
        hist = [{k: r.get(k) for k in ("day_in_cycle", "phase", "oil_bbl", "T_res", "spm", "down_frac", "RFI",
                                       "kwh", "fillage", "failure", "mu_tubing_cp")} for r in s.log[-200:]]
        out["history"] = hist
        out["css"] = s.css
    return _j(out)


# ----------------------------------------------------------------- field / reliability / models
@app.get("/api/steam_schedule")
def steam_schedule(generators: int = 2):
    return _j({"generators": generators, "rows": TW.steam_schedule(generators)})


@app.get("/api/reliability")
def reliability():
    fails = pd.read_csv(C.DATA_DIR / "failures.csv").sort_values("date")
    by_cause = fails.groupby("probable_cause").size().to_dict()
    by_well = fails.groupby("well_id").size().to_dict()
    return _j({"risk": TW.risk_table(), "failures": fails.tail(40).iloc[::-1].to_dict("records"),
               "by_cause": by_cause, "by_well": by_well,
               "importance": TW.metrics["M4"]["feature_importance"]})


@app.get("/api/anomalies")
def anomalies():
    return _j({"rows": TW.anomalies()})


@app.get("/api/models")
def models():
    return _j(TW.metrics)


@app.get("/api/meta")
def meta():
    return {"wells": list(TW._props), "demo_well": DEMO_WELL, "classes": C.CARD_CLASSES,
            "as_of": TW.daily.date.max(),
            "economics": {"oil_price": C.OIL_PRICE, "steam_cost": C.STEAM_COST, "power_cost": C.POWER_COST,
                          "rod_failure_cost": C.ROD_FAILURE_COST}}


app.mount("/", StaticFiles(directory=str(C.DASH_DIR), html=True), name="dashboard")
