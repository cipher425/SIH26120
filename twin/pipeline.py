"""End-to-end pipeline: data -> calibration -> AI models -> CSS/SRP optimisation.

    python -m twin.pipeline            # everything
    python -m twin.pipeline --skip-data --trials 60
"""
import argparse
import json
import time

from joblib import Parallel, delayed

from twin import config as C


def _opt_well(wid, trials):
    from twin.core.twin import FieldTwin
    tw = FieldTwin()
    return wid, tw.scenarios(wid, n_trials=trials)


def optimise_all(trials=50, n_jobs=-1):
    from twin.core.calibration import load_field
    wells, _, _ = load_field()
    res = Parallel(n_jobs=n_jobs)(delayed(_opt_well)(w, trials) for w in wells.well_id)
    out = dict(res)
    with open(C.ART_DIR / "css_optimisation.json", "w") as f:
        json.dump(out, f)
    # field-level ablation summary
    keys = ["A_historical", "B_ai_srp_only", "C_ai_css_only", "D_joint_ai"]
    agg = {k: {"oil": 0.0, "steam": 0.0, "kwh": 0.0, "fail": 0.0, "value_rate": 0.0, "float_days": 0.0,
               "oil_per_day": 0.0, "steam_per_day": 0.0, "fail_per_year": 0.0, "kwh_per_day": 0.0} for k in keys}
    for wid, sc in out.items():
        for k in keys:
            s = sc["scenarios"][k]["summary"]
            a = agg[k]
            a["oil"] += s["cum_oil_bbl"]
            a["steam"] += s["steam_t"]
            a["kwh"] += s["kwh"]
            a["fail"] += s["exp_failures"]
            a["value_rate"] += s["value_per_day"]
            a["float_days"] += s["float_days"]
            a["oil_per_day"] += s["cum_oil_bbl"] / s["cycle_days"]
            a["steam_per_day"] += s["steam_t"] / s["cycle_days"]
            a["kwh_per_day"] += s["kwh"] / s["cycle_days"]
            a["fail_per_year"] += s["exp_failures"] / s["cycle_days"] * 365
    for k, a in agg.items():
        a["csor"] = a["steam_per_day"] / (a["oil_per_day"] * C.BBL)
        a["kwh_per_bbl"] = a["kwh_per_day"] / a["oil_per_day"]
    agg["truth_validation"] = validate_on_truth(out)
    with open(C.ART_DIR / "field_ablation.json", "w") as f:
        json.dump(agg, f, indent=1)
    return agg


def validate_on_truth(out):
    """Replay each scenario's plan on the hidden ground-truth field physics (which the
    twin never sees) - checks that twin-optimised decisions really pay off."""
    from datetime import date, timedelta
    from twin.core.twin import FieldTwin
    from twin.sim.cycle import CycleSim, run_cycle
    tw = FieldTwin()
    res = {}
    for k in ("A_historical", "B_ai_srp_only", "C_ai_css_only", "D_joint_ai"):
        a = {"oil_per_day": 0.0, "steam_per_day": 0.0, "kwh_per_day": 0.0, "fail_per_year": 0.0, "value_rate": 0.0}
        for wid, sc in out.items():
            s_ = sc["scenarios"][k]
            t = tw.truth[wid]
            w = tw.props(wid)
            sim = CycleSim(w, s_["css"], tw.state_now(wid), date.fromisoformat(C.FIELD_END) + timedelta(days=1),
                           t.heat_eff, t.k_mult, t.visc_factor, truth=t)
            _, s = run_cycle(sim, tw.controller(wid, s_["srp_mode"]))
            a["oil_per_day"] += s["cum_oil_bbl"] / s["cycle_days"]
            a["steam_per_day"] += s["steam_t"] / s["cycle_days"]
            a["kwh_per_day"] += s["kwh"] / s["cycle_days"]
            a["fail_per_year"] += s["exp_failures"] / s["cycle_days"] * 365
            a["value_rate"] += s["value_per_day"]
        a["csor"] = a["steam_per_day"] / (a["oil_per_day"] * C.BBL)
        a["kwh_per_bbl"] = a["kwh_per_day"] / a["oil_per_day"]
        res[k] = a
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()
    t0 = time.time()
    C.ART_DIR.mkdir(exist_ok=True)
    if not a.skip_data:
        print("[1/4] Generating synthetic field data ...")
        from twin.data import generate
        generate.main()
    if not a.skip_train:
        print("[2/4] History-matching (calibrating) the twin for every well ...")
        from twin.core import calibration
        calibration.calibrate_field()
        print("[3/4] Training AI models M1-M5 ...")
        from twin.models import train_all
        train_all.main()
    print(f"[4/4] Optimising CSS + SRP for every well ({a.trials} Bayesian trials x 2 policies) ...")
    agg = optimise_all(a.trials)
    base = agg["A_historical"]
    tv = agg.pop("truth_validation")
    for k, v in agg.items():
        print(f"  {k:15s} oil={v['oil_per_day']:6.0f} bopd  cSOR={v['csor']:.2f}  kWh/bbl={v['kwh_per_bbl']:.2f}  "
              f"failures/yr={v['fail_per_year']:.1f}  value/day={v['value_rate'] / 1e5:.2f} lakh "
              f"({(v['value_rate'] / base['value_rate'] - 1) * 100:+.1f}%)")
    print("  -- validated on hidden ground-truth physics --")
    for k, v in tv.items():
        print(f"  {k:15s} oil={v['oil_per_day']:6.0f} bopd  cSOR={v['csor']:.2f}  kWh/bbl={v['kwh_per_bbl']:.2f}  "
              f"failures/yr={v['fail_per_year']:.1f}  value/day={v['value_rate'] / 1e5:.2f} lakh "
              f"({(v['value_rate'] / tv['A_historical']['value_rate'] - 1) * 100:+.1f}%)")
    print(f"Pipeline finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
