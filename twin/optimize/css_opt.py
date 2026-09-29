"""CSS cycle optimiser (Bayesian optimisation, Optuna TPE).

Every trial runs the full coupled twin for one cycle - calibrated reservoir physics
x ML residual (M1), wellbore, and the SRP controller - so the steam plan is chosen
*knowing how the pump will be operated*. That is the integration the field lacks today.

Objective: net value rate (INR/day) = (oil revenue - steam - power - expected failure
cost) / cycle length. Maximising value *rate* makes the production cut-off obey the
marginal value theorem: stop when today's margin falls below the cycle's average rate.
"""
import optuna

from twin import config as C

optuna.logging.set_verbosity(optuna.logging.WARNING)

BOUNDS = {"steam_t": (1000, 3500), "quality": (0.60, 0.85), "inj_pressure_mpa": (4.0, 8.0),
          "soak_days": (2, 14), "cutoff_bopd": (5.0, 25.0)}


def suggest(trial):
    return {"steam_t": float(trial.suggest_int("steam_t", 1000, 3500, step=50)),
            "quality": round(trial.suggest_float("quality", *BOUNDS["quality"]), 3),
            "inj_pressure_mpa": round(trial.suggest_float("inj_pressure_mpa", *BOUNDS["inj_pressure_mpa"]), 2),
            "soak_days": trial.suggest_int("soak_days", *BOUNDS["soak_days"]),
            "cutoff_bopd": round(trial.suggest_float("cutoff_bopd", *BOUNDS["cutoff_bopd"]), 1)}


def optimize(twin, wid, srp_mode="ai", n_trials=50, seed=0, baseline_css=None, extra_starts=()):
    trials = []

    def objective(trial):
        css = suggest(trial)
        if css["inj_pressure_mpa"] >= C.FRAC_PRESSURE_MPA:
            return -1e9
        _, s = twin.simulate(wid, css, srp_mode=srp_mode)
        trials.append({**css, "oil": s["cum_oil_bbl"], "csor": s["csor"], "value_per_day": s["value_per_day"],
                       "cycle_days": s["cycle_days"], "kwh_per_bbl": s["kwh_per_bbl"]})
        return s["value_per_day"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    if baseline_css:
        study.enqueue_trial({k: baseline_css[k] for k in BOUNDS})
    for s in extra_starts:
        study.enqueue_trial({k: s[k] for k in BOUNDS})
    study.optimize(objective, n_trials=n_trials)
    best = trials[study.best_trial.number]
    return {"best_css": {k: best[k] for k in BOUNDS}, "best": best, "trials": trials}
