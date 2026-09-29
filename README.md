# ThermoLift Twin — AI-enabled Well-to-Surface Digital Twin
**SIH 2026 · Problem Statement 120 · Baghewala heavy-oil field (CSS + SRP)**

An integrated digital twin that couples **reservoir (CSS) → wellbore → sucker-rod pump**, learns from field
history, and **jointly** optimises steam cycles and artificial lift. It replaces today's approach, where the two
are tuned separately from experience.

## Quick start (Windows)
```powershell
.\setup.ps1            # one time: Python venv + libraries
.\run_pipeline.ps1     # ~3 min: data -> calibration -> 5 AI models -> optimisation
.\start_dashboard.ps1  # opens http://127.0.0.1:8000
```
Demo deep links: `http://127.0.0.1:8000/#live/ff=100` (jump to cycle day 100), `#css`, `#srp`, `#models`.

## Headline results (next CSS cycle, 20 wells, validated on hidden ground-truth physics)
| Metric (per day of operation) | Today's practice | Joint AI (ours) | Change |
|---|---|---|---|
| Oil rate | 356 bopd | 490 bopd | **+38 %** |
| Steam-oil ratio (cSOR) | 4.27 | 3.63 | **−15 %** |
| Lift energy | 6.34 kWh/bbl | 4.11 kWh/bbl | **−35 %** |
| Rod / pump failures | 23.2 /yr | 9.6 /yr | **−59 %** |
| Net value | ₹14.1 L/day | ₹21.1 L/day | **+50 %** |

Ablation: AI SRP only gives +2.7 % value and AI CSS only gives +39.5 %. **Joint gives +49.7 %**, more than
optimising either system alone. That gap is the core claim of the problem statement.

## AI model performance (held-out data)
| Model | What it does | Result |
|---|---|---|
| M1 Hybrid reservoir | Boberg-Lantz physics × gradient-boosted residual | cycle-oil MAPE **7.1 %** (physics only 32 %) |
| M2 Hybrid LSTM | 14-day oil forecast = decline curve + LSTM correction | MAPE **4.8 %** (decline curve 5.6 %), unseen wells |
| M3 1-D CNN | Dyno-card diagnosis, 7 pump conditions | **99.3 %** accuracy |
| M4 Reliability | P(rod part / pump unseat in 14 days) | ROC-AUC **0.73**, at the theoretical oracle ceiling (0.71) |
| M5 Anomaly | Physics-residual fault detection + type | F1 **0.92**, fault-type accuracy **99 %** |

## How it works
```
FIELD DATA ──> DATA LAYER ──> PHYSICS TWIN (history-matched) ──> AI LAYER (M1–M5)
 SCADA/VFD      features:       Marx-Langenheim heating              │
 dyno cards     tubing μ, RFI,  Boberg-Lantz cooling + inflow        ▼
 CSS logs       Goodman,        Ramey wellbore heat loss      OPTIMISERS
 PVT, failures  residuals       Rod drag, rod float, pump fill  CSS: Bayesian (Optuna) with the
                                Calibrated per well                  SRP controller inside each trial
                                                                SRP: model-predictive SPM + VFD profile
                                                                Steam scheduler across generators
                                                                     │
      ▲──────────── new data recalibrates the twin <── DECISIONS: VFD set-points, steam plan, alerts
```
Key physics for this field:
* **Rod-Float-Index** `RFI = (viscous drag + inertia on downstroke) / buoyed rod weight`. When RFI > 1 the rods
  separate from the carrier bar, causing impact loading, rod failures and pump unseating.
* **VFD slow-down-stroke profile.** A longer downstroke lowers rod velocity in viscous crude without losing production.
* **Economic cut-off.** Maximising value *rate* ends a cycle when the daily margin drops below the cycle average
  (marginal value theorem), instead of at a fixed oil rate.

## Project layout
```
twin/physics/    fluid (PVT, steam), reservoir (CSS), wellbore (Ramey), srp (rod pump), cards (dyno cards)
twin/sim/        day-by-day coupled cycle simulator + operating policies (historical / AI / replay)
twin/data/       synthetic field-history generator (20 wells, 2019-2026, hidden "truth" parameters)
twin/core/       calibration (history matching), FieldTwin runtime, LiveSession (closed-loop demo)
twin/models/     physics features, M1–M5 training & evaluation (train_all.py)
twin/optimize/   srp_opt (real-time SRP), css_opt (Bayesian CSS)
twin/api/        FastAPI backend
dashboard/       web dashboard (vanilla JS + ECharts, works offline)
data/            generated datasets (CSV/Parquet/NPZ)
artifacts/       calibration, trained models, metrics, optimisation results
```

## About the data
Real Baghewala data is not public, so `twin/data/generate.py` builds a realistic synthetic field with the same
data families the problem statement lists: production, CSS cycles, steam, VFD/SRP, failures, completion and PVT.
It includes sensor noise, injected sensor/equipment faults and **hidden reservoir behaviour the twin must learn**:
per-cycle steam conformance decay, channelling at high injection pressure, and true permeability and viscosity.
To use real field data, replace the CSVs in `data/` with the same columns and re-run steps 2–4 of the pipeline
(`.\run_pipeline.ps1 --skip-data`).
