"""FieldTwin - runtime digital twin used by the API, optimisers and live demo."""
import copy
import json
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import torch

from twin import config as C
from twin.core.calibration import load_field, well_from_row
from twin.models import deep
from twin.models.features import M4_FEATURES
from twin.models.train_all import M1_FEATURES
from twin.physics import cards, srp, wellbore
from twin.physics.reservoir import WellState, TruthParams
from twin.sim.cycle import CycleSim, run_cycle
from twin.sim.controllers import AIController, HistoricalController


class ReliabilityModel:
    """Wraps M4 so the SRP optimiser can score a whole control grid."""

    def __init__(self, model):
        self.m = model

    def predict_grid(self, p, obs, V, D=0.5):
        V = np.atleast_1d(V)
        shape = V.shape
        n = V.size
        f = {"RFI_est": p["RFI"].ravel(), "stress_est": p["stress_ratio"].ravel(),
             "MPRL_norm": (p["MPRL"] / p["W_rb"]).ravel(), "PPRL_kN": (p["PPRL"] / 1000).ravel(),
             "spm": V.ravel(), "down_frac": np.broadcast_to(D, shape).ravel(), "fillage": p["fillage"].ravel(),
             "log_mu_tub": np.full(n, np.log(obs["mu_tubing_cp"])), "rfi_7d": p["RFI"].ravel(),
             "float_7d": 7.0 * (p["RFI"].ravel() > 1), "float_30d": 30.0 * (p["RFI"].ravel() > 1),
             "impact_30d": 30.0 * p["impact"].ravel()}
        lr, lu = srp.failure_hazards(p["RFI"], p["stress_ratio"], p["impact"])
        f["haz_phys"] = (lr + lu).ravel()
        f["haz_phys_14d"] = 14 * f["haz_phys"]
        f["days_since_failure"] = np.full(n, obs.get("days_since_failure", 365.0))
        X = pd.DataFrame({k: f[k] for k in M4_FEATURES})
        return self.m.predict_proba(X)[:, 1].reshape(shape)


class FieldTwin:
    def __init__(self):
        self.wells, self.cyc, self.daily = load_field()
        self.calib = json.load(open(C.ART_DIR / "calibration.json"))
        self.metrics = json.load(open(C.ART_DIR / "metrics.json"))
        self.truth = {k: TruthParams(**v) for k, v in json.load(open(C.DATA_DIR / "truth_params.json")).items()}
        self.m1 = joblib.load(C.ART_DIR / "m1_residual.joblib")
        self.m4 = joblib.load(C.ART_DIR / "m4_reliability.joblib")
        m5 = joblib.load(C.ART_DIR / "m5_anomaly.joblib")
        self.m5_iso, self.m5_clf = m5["iso"], m5["clf"]
        self.m2 = deep.LSTMForecaster()
        self.m2.load_state_dict(torch.load(C.ART_DIR / "m2_lstm.pt"))
        self.m2.eval()
        self.m3 = deep.CardCNN(len(C.CARD_CLASSES))
        self.m3.load_state_dict(torch.load(C.ART_DIR / "m3_cnn.pt"))
        self.m3.eval()
        self.reliability = ReliabilityModel(self.m4)
        self._props = {r.well_id: well_from_row(r) for _, r in self.wells.iterrows()}
        fp = C.ART_DIR / "features_daily.parquet"
        self.feat = pd.read_parquet(fp) if fp.exists() else None
        op = C.ART_DIR / "css_optimisation.json"
        self.css_cache = json.load(open(op)) if op.exists() else {}

    # ------------------------------------------------------------------ basics
    def props(self, wid):
        return self._props[wid]

    def state_now(self, wid):
        return WellState(**self.calib[wid]["final_state"])

    def historical_css(self, wid):
        g = self.cyc[self.cyc.well_id == wid].tail(4)
        return {"steam_t": float(round(g.steam_t.mean() / 50) * 50), "quality": round(float(g.quality.mean()), 3),
                "inj_pressure_mpa": round(float(g.inj_pressure_mpa.mean()), 2),
                "soak_days": int(round(g.soak_days.mean())), "cutoff_bopd": round(float(g.cutoff_bopd.mean()), 1)}

    def m1_ratio(self, wid, css, cycle_no, sim_oil):
        cal = self.calib[wid]
        w = self.props(wid)
        x = pd.DataFrame([{"cycle": cycle_no, **{k: css[k] for k in ("steam_t", "quality", "inj_pressure_mpa",
                                                                      "soak_days", "cutoff_bopd")},
                           "log_sim_oil": np.log(sim_oil + 1), "k_mult": cal["k_mult"], "heat_eff": cal["heat_eff"],
                           "visc_factor_lab": w.visc_factor_lab, "h_m": w.h_m}])[M1_FEATURES]
        return float(np.clip(np.exp(self.m1.predict(x)[0]), 0.4, 2.0))

    def controller(self, wid, srp_mode):
        w = self.props(wid)
        if srp_mode == "ai":
            return AIController(w)
        return HistoricalController(w.stroke_m, spm_start=7.0)

    def simulate(self, wid, css, srp_mode="ai", state=None, hybrid=True, start=None):
        """Predict one full CSS cycle with the calibrated hybrid twin."""
        w = self.props(wid)
        cal = self.calib[wid]
        st = (state or self.state_now(wid))
        start = start or date.fromisoformat(C.FIELD_END) + timedelta(days=1)
        corr = 1.0
        if hybrid:
            sim0 = CycleSim(w, css, st.copy(), start, cal["heat_eff"], cal["k_mult"], w.visc_factor_lab)
            _, s0 = run_cycle(sim0, self.controller(wid, srp_mode))
            corr = self.m1_ratio(wid, css, sim0.cycle_no, s0["cum_oil_bbl"])
        sim = CycleSim(w, css, st.copy(), start, cal["heat_eff"], cal["k_mult"], w.visc_factor_lab,
                       oil_correction=corr)
        rows, s = run_cycle(sim, self.controller(wid, srp_mode))
        s["m1_correction"] = corr
        return rows, s

    # ------------------------------------------------------------------ scenarios
    def scenarios(self, wid, n_trials=40):
        from twin.optimize import css_opt
        base = self.historical_css(wid)
        opt_hist = css_opt.optimize(self, wid, "historical", n_trials, baseline_css=base)
        # joint search is warm-started with both today's plan and the CSS-only optimum
        opt_ai = css_opt.optimize(self, wid, "ai", n_trials, baseline_css=base,
                                  extra_starts=[opt_hist["best_css"]])
        runs = {"A_historical": (base, "historical"), "B_ai_srp_only": (base, "ai"),
                "C_ai_css_only": (opt_hist["best_css"], "historical"), "D_joint_ai": (opt_ai["best_css"], "ai")}
        out = {}
        for k, (css, mode) in runs.items():
            rows, s = self.simulate(wid, css, mode)
            prod = [r for r in rows if r["phase"] == "produce"]
            out[k] = {"css": css, "srp_mode": mode, "summary": _clean(s),
                      "curve": [{"d": r["day_in_cycle"], "oil": round(r["oil_bbl"], 2), "T": round(r["T_res"], 1),
                                 "spm": r["spm"], "rfi": round(r.get("RFI", 0), 3), "kwh": round(r["kwh"], 1),
                                 "margin": round(r["oil_bbl"] * C.OIL_PRICE - r["kwh"] * C.POWER_COST
                                                 - r.get("lam_rod", 0) * C.ROD_FAILURE_COST
                                                 - r.get("lam_unseat", 0) * C.PUMP_UNSEAT_COST, 0)}
                                for r in rows if r["phase"] in ("produce", "down")] if k in ("A_historical", "D_joint_ai") else None,
                      "avg_spm": float(np.mean([r["spm"] for r in prod])) if prod else 0}
        return {"well_id": wid, "scenarios": out, "trials": opt_ai["trials"],
                "trials_hist": opt_hist["trials"]}

    # ------------------------------------------------------------------ steam scheduler
    def steam_schedule(self, generators=2):
        last = self.daily[self.daily.date == self.daily.date.max()]
        rows = []
        for _, wr in self.wells.iterrows():
            wid = wr.well_id
            r = last[last.well_id == wid]
            cur_margin, phase, oil = 0.0, "idle", 0.0
            if len(r):
                r = r.iloc[0]
                phase = r.phase
                oil = float(r.oil_bbl)
                cur_margin = oil * C.OIL_PRICE - float(r.kwh) * C.POWER_COST
            sc = self.css_cache.get(wid)
            if sc:
                nxt = sc["scenarios"]["D_joint_ai"]["summary"]
                css = sc["scenarios"]["D_joint_ai"]["css"]
            else:
                css = self.historical_css(wid)
                _, nxt = self.simulate(wid, css, "ai")
            gain = nxt["value_per_day"] - cur_margin
            rows.append({"well_id": wid, "phase": phase, "current_oil_bopd": round(oil, 1),
                         "current_margin_inr_d": round(cur_margin), "next_cycle_value_inr_d": round(nxt["value_per_day"]),
                         "gain_inr_d": round(gain), "steam_t": css["steam_t"],
                         "inj_days": int(np.ceil(css["steam_t"] / C.STEAM_GEN_RATE)),
                         "eligible": phase in ("produce", "idle", "down")})
        rows.sort(key=lambda x: -x["gain_inr_d"])
        gen_free = [0] * generators
        start = date.fromisoformat(C.FIELD_END) + timedelta(days=1)
        for r in rows:
            if not r["eligible"]:
                r["action"] = "steaming now"
                continue
            if r["gain_inr_d"] <= 0:
                r["action"] = "continue producing"
                continue
            g = int(np.argmin(gen_free))
            r["generator"] = f"SG-{g + 1}"
            r["start"] = (start + timedelta(days=gen_free[g])).isoformat()
            gen_free[g] += r["inj_days"] + 1
            r["end"] = (start + timedelta(days=gen_free[g] - 1)).isoformat()
            r["action"] = "re-steam"
        return rows

    # ------------------------------------------------------------------ inference helpers
    def classify_card(self, card_norm):
        with torch.no_grad():
            logits = self.m3(torch.as_tensor(card_norm[None]))
            prob = torch.softmax(logits, 1).numpy()[0]
        i = int(prob.argmax())
        return C.CARD_CLASSES[i], {c: round(float(p), 4) for c, p in zip(C.CARD_CLASSES, prob)}

    def forecast_oil(self, wid, cycle=None):
        d = self.daily[(self.daily.well_id == wid) & (self.daily.phase == "produce")]
        cycle = cycle or int(d.cycle.max())
        g = d[d.cycle == cycle].sort_values("prod_day")
        if len(g) < deep.WIN:
            g = d[d.cycle == cycle - 1].sort_values("prod_day")
        arr = g[deep.M2_FEATS].to_numpy(float)
        x = deep.m2_transform(arr[-deep.WIN:])[None]
        f = deep.forecast(self.m2, x)[0]
        base = deep.decline_baseline(x)[0]
        last = x[0, -1, 0]
        return {"history": [{"d": int(a), "oil": round(float(b), 2)} for a, b in zip(g.prod_day, g.oil_bbl)],
                "forecast": np.round(np.expm1(f + last), 2).tolist(),
                "decline": np.round(np.expm1(base + last), 2).tolist(),
                "start_day": int(g.prod_day.iloc[-1]) + 1}

    def risk_table(self):
        if self.feat is None:
            return []
        f = self.feat.sort_values("date").groupby("well_id").tail(1)
        p = self.m4.predict_proba(f[M4_FEATURES])[:, 1]
        out = []
        for (_, r), pr in zip(f.iterrows(), p):
            out.append({"well_id": r.well_id, "date": r.date, "risk_14d": round(float(pr), 3),
                        "RFI": round(float(r.RFI_est), 2), "stress": round(float(r.stress_est), 2),
                        "float_30d": int(r.float_30d), "mu_tubing_cp": round(float(r.mu_tub_est), 0),
                        "spm": float(r.spm), "days_since_failure": int(r.days_since_failure)})
        return sorted(out, key=lambda x: -x["risk_14d"])

    def anomalies(self, limit=60):
        if self.feat is None:
            return []
        from twin.models.features import M5_FEATURES
        f = self.feat[self.feat.date >= "2025-07-01"]
        X = f[M5_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-30, 30)
        lab = self.m5_clf.predict(X)
        score = -self.m5_iso.score_samples(X)
        f = f.assign(label=lab, score=score)
        a = f[f.label != "none"].sort_values("date", ascending=False).head(limit)
        return [{"date": r.date, "well_id": r.well_id, "type": r.label, "score": round(float(r.score), 3),
                 "kwh_ratio": round(float(r.kwh_ratio), 2), "pprl_ratio": round(float(r.pprl_ratio), 2),
                 "liquid_ratio": round(float(r.liquid_ratio), 2)} for _, r in a.iterrows()]


def _clean(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


# ====================================================================== live session
class LiveSession:
    """Emulates the real well (hidden truth physics + stochastic failures) while the twin
    watches: diagnoses the dyno card (M3), scores risk (M4), forecasts rod float, and -
    when AI mode is on - drives SPM / VFD profile through the SRP optimiser."""

    def __init__(self, twin: FieldTwin, wid, css=None, seed=0):
        self.twin = twin
        self.wid = wid
        w = twin.props(wid)
        self.w = w
        truth = twin.truth[wid]
        self.css = css or twin.historical_css(wid)
        self.rng = np.random.default_rng(seed)
        st = twin.state_now(wid)
        self.sim = CycleSim(w, self.css, st, date.fromisoformat(C.FIELD_END) + timedelta(days=1),
                            truth.heat_eff, truth.k_mult, truth.visc_factor, truth=truth, rng=self.rng,
                            stochastic=True)
        self.hist_ctl = HistoricalController(w.stroke_m, spm_start=7.0, rng=self.rng)
        self.ai_ctl = AIController(w, fixed_stroke=w.stroke_m, estimation_noise=0.05, rng=self.rng,
                                   risk_model=None)
        self.mode = "manual"
        self.log = []
        self.cum = {"manual": {"oil": 0, "kwh": 0, "days": 0, "fail": 0, "float": 0},
                    "ai": {"oil": 0, "kwh": 0, "days": 0, "fail": 0, "float": 0}}
        self.days_since_failure = 365

    def _ctl(self):
        outer = self

        class Switch:
            def decide(self, sim, obs):
                c = outer.ai_ctl if outer.mode == "ai" else outer.hist_ctl
                out = c.decide(sim, obs)
                if outer.mode == "ai":
                    outer.hist_ctl.spm = out["spm"]
                return out

            def on_failure(self, kind):
                outer.hist_ctl.on_failure(kind)
        return Switch()

    def lookahead_float(self, spm, stroke, d, horizon=60):
        """Days until rod floating starts if current settings are kept."""
        sim = copy.deepcopy(self.sim)
        sim.stochastic = False
        from twin.sim.controllers import FixedController
        ctl = FixedController(spm, stroke, d)
        for k in range(horizon):
            if sim.done:
                return None
            r = sim.step(ctl)
            if r.get("RFI", 0) > 1.0:
                return k
        return None

    def tick(self):
        if self.sim.done:
            return {"done": True}
        rec = self.sim.step(self._ctl())
        out = {"done": False, "mode": self.mode, "record": _clean({k: v for k, v in rec.items()})}
        if rec["phase"] == "produce":
            obs = self.sim.observe()
            base = "fluid_pound" if rec["fill_inflow"] < 0.8 else "normal"
            if rec["card_class"] == "gas_interference":
                base = "gas_interference"
            if rec["RFI"] > 1.0:
                base = "rod_float"
            if rec["failure"] == "pump_unseat":
                base = "pump_unseated"
            card = cards.synth_card(base, rec["stroke_m"], rec["spm"], rec["down_frac"], rec["W_rb_kN"] * 1000,
                                    rec["Fo_kN"] * 1000, rec["c_visc"], fillage=rec["fill_inflow"],
                                    rng=self.rng, severity=0.0 if base == "normal" else None)
            norm = cards.normalise_card(card["position"], card["load"], rec["stroke_m"],
                                        rec["W_rb_kN"] * 1000, rec["Fo_kN"] * 1000)
            diag, probs = self.twin.classify_card(norm)
            opt = self.ai_ctl.opt
            best = opt.optimize({**obs, "days_since_failure": self.days_since_failure}, strokes=[rec["stroke_m"]])
            from twin.optimize.srp_opt import explain
            reasons = explain(best, {"spm": rec["spm"], "RFI": rec["RFI"]}, obs)
            p = srp.srp_eval(rec["spm"], rec["stroke_m"], rec["down_frac"], obs["q_in"], obs["wc"],
                             obs["mu_tubing_cp"], obs["mu_pump_cp"], self.w.pump_depth_m)
            risk = float(self.twin.reliability.predict_grid(
                {k: np.atleast_1d(v) for k, v in p.items()}, {**obs, "days_since_failure": self.days_since_failure},
                rec["spm"], rec["down_frac"])[0])
            float_in = self.lookahead_float(rec["spm"], rec["stroke_m"], rec["down_frac"]) if rec["RFI"] <= 1 else 0
            out.update({
                "card": {"x": np.round(card["position"], 3).tolist(), "y": np.round(card["load"] / 1000, 2).tolist(),
                         "px": np.round(card["pump_position"], 3).tolist(),
                         "py": np.round(card["pump_load"] / 1000, 2).tolist()},
                "diagnosis": diag, "diag_probs": probs, "risk_14d": round(risk, 3),
                "recommendation": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()},
                "reasons": reasons, "float_in_days": float_in,
                "profile": {"z": np.round(obs["profile_z"], 0).tolist(), "T": np.round(obs["profile_T"], 1).tolist()},
            })
            c = self.cum[self.mode]
            c["oil"] += rec["oil_bbl"]
            c["kwh"] += rec["kwh"]
            c["days"] += 1
            c["float"] += int(rec["RFI"] > 1)
            self.days_since_failure += 1
        if rec.get("failure"):
            self.cum[self.mode]["fail"] += 1
            self.days_since_failure = 0
        out["cum"] = self.cum
        self.log.append(out["record"])
        return out
