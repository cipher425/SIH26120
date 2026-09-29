"""SRP operating policies.

HistoricalController : today's practice - high SPM after steaming, fixed stroke,
                       symmetric VFD profile, manual weekly review, reactive
                       slow-down only after low fillage or a failure.
AIController         : digital-twin optimiser picks SPM + VFD down-stroke
                       fraction every day (proactive, model-predictive).
ReplayController     : replays recorded historical settings (for calibration).
"""
import numpy as np

from twin import config as C


class HistoricalController:
    def __init__(self, stroke_m, spm_start=7.0, rng=None):
        self.stroke = stroke_m
        self.spm_start = spm_start
        self.spm = spm_start
        self.rng = rng or np.random.default_rng(1)
        self.fill_log = []
        self.last_prod_day = -1

    def decide(self, sim, obs):
        if obs["prod_day"] == 0 or obs["prod_day"] < self.last_prod_day:
            self.spm = self.spm_start
            self.fill_log = []
        self.last_prod_day = obs["prod_day"]
        if obs["prod_day"] > 0 and obs["prod_day"] % 7 == 0 and self.fill_log:
            f = np.mean(self.fill_log[-7:])
            if f < 0.65:
                self.spm = max(2.0, self.spm - 1.0)
            elif f > 0.97 and self.rng.random() < 0.3:
                self.spm = min(8.0, self.spm + 0.5)
        # operator estimates fillage from pump-off controller
        from twin.physics import srp
        p = srp.srp_eval(self.spm, self.stroke, 0.5, obs["q_in"], obs["wc"],
                         obs["mu_tubing_cp"], obs["mu_pump_cp"], sim.w.pump_depth_m)
        self.fill_log.append(float(p["fill_inflow"]))
        return {"spm": self.spm, "stroke_m": self.stroke, "down_frac": 0.5}

    def on_failure(self, kind):
        self.spm = max(2.0, self.spm - 1.0)


class FixedController:
    def __init__(self, spm, stroke_m, down_frac=0.5):
        self.c = {"spm": spm, "stroke_m": stroke_m, "down_frac": down_frac}

    def decide(self, sim, obs):
        return dict(self.c)

    def on_failure(self, kind):
        pass


class ReplayController:
    """Replays recorded daily settings; records keyed by prod_day."""

    def __init__(self, records):
        self.rec = records

    def decide(self, sim, obs):
        r = self.rec.get(obs["prod_day"])
        if r is None:
            last = self.rec[max(self.rec)] if self.rec else {"spm": 4, "stroke_m": 2.18, "down_frac": 0.5}
            r = last
        if r.get("down"):
            return {"down": 1}
        return {"spm": r["spm"], "stroke_m": r["stroke_m"], "down_frac": r["down_frac"]}

    def on_failure(self, kind):
        pass


class AIController:
    """Uses the SRP optimiser each day. `estimation_noise` emulates the twin's
    state-estimation error on inflow capacity."""

    def __init__(self, well, fixed_stroke=None, estimation_noise=0.0, rng=None, risk_model=None):
        from twin.optimize.srp_opt import SRPOptimizer
        self.opt = SRPOptimizer(well, risk_model=risk_model)
        self.stroke = fixed_stroke
        self.noise = estimation_noise
        self.rng = rng or np.random.default_rng(7)
        self.last = None

    def decide(self, sim, obs):
        o = dict(obs)
        if self.noise:
            o["q_in"] = obs["q_in"] * (1 + self.rng.normal(0, self.noise))
        if self.stroke is None or obs["prod_day"] == 0:
            best = self.opt.optimize(o, strokes=C.STROKE_OPTIONS if self.stroke is None else [self.stroke])
            if obs["prod_day"] == 0 or self.stroke is None:
                self.stroke = best["stroke_m"]
        best = self.opt.optimize(o, strokes=[self.stroke])
        self.last = best
        return {"spm": best["spm"], "stroke_m": best["stroke_m"], "down_frac": best["down_frac"]}

    def on_failure(self, kind):
        pass
