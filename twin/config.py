"""Field constants, equipment defaults and economics for a Baghewala-like heavy-oil field.

All values are representative public-domain magnitudes for the Jodhpur Sandstone
(17-19 deg API, 46-48 degC, low pressure). Replace them with field data when available.
"""
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
DASH_DIR = ROOT / "dashboard"

G = 9.81
BBL = 0.158987            # m3 per barrel
PSI_PER_MPA = 145.038

# ---------------- Reservoir / thermal ----------------
T_SURF = 30.0              # degC surface temperature (Rajasthan)
GEO_GRAD = 0.017           # degC per m
T_RES = 47.0               # degC initial reservoir temperature
API_GRAVITY = 18.0
RHO_OIL = 141.5 / (131.5 + API_GRAVITY) * 1000.0   # ~946 kg/m3
RHO_WATER = 1000.0
CP_OIL = 2.0               # kJ/kg/K
CP_WATER = 4.18            # kJ/kg/K
M_ROCK = 2400.0            # kJ/m3/K volumetric heat capacity of saturated sand
K_ROCK_D = 1.7 * 86.4      # thermal conductivity, kJ/day/m/K (1.7 W/m/K)
ALPHA_ROCK = K_ROCK_D / M_ROCK   # m2/day
OIL_FVF = 1.05
SKIN_COLD = 4.0            # asphaltene / fines damage skin, cleaned up by steam

# Arrhenius (Andrade) viscosity fit ln(mu_cp) = A + B/T_K, fitted to
# mu(47 C) ~ 1500 cP and mu(150 C) ~ 20 cP for asphaltenic 18 API crude.
VISC_A = -10.42
VISC_B = 5678.0

# ---------------- Rod string / pump ----------------
E_STEEL = 2.07e11
RHO_STEEL = 7850.0
ROD_DIAM = 0.01905                          # 3/4" rods
ROD_AREA = math.pi * ROD_DIAM ** 2 / 4
ROD_MASS_PER_M = 2.45                       # kg/m incl. couplings
TUBING_ID = 0.062                           # 2-7/8" tubing
PLUNGER_DIAM = 0.04445                      # 1.75" pump
ROD_TENSILE = 793e6                         # API grade D (115 ksi)
SERVICE_FACTOR = 0.9
COUPLING_DRAG_FACTOR = 3.0                  # couplings/guides multiply Couette drag
SURFACE_EFF = 0.55                          # motor + gearbox + unit efficiency
IDLE_KW = 1.0
STROKE_OPTIONS = [1.73, 2.18, 2.64]         # m (68", 86", 104")
SPM_MIN, SPM_MAX = 1.5, 9.0
DOWN_FRAC_OPTIONS = [0.50, 0.55, 0.60, 0.65]
THP_MPA = 0.6

# ---------------- CSS equipment limits ----------------
STEAM_GEN_RATE = 180.0      # t/day of steam (CWE) per generator
FRAC_PRESSURE_MPA = 9.0
MAX_PROD_DAYS = 320

# ---------------- Economics (INR) ----------------
OIL_PRICE = 5800.0          # INR/bbl
STEAM_COST = 2500.0         # INR/t steam (gas fuel + water treatment)
POWER_COST = 8.0            # INR/kWh
ROD_FAILURE_COST = 600000.0 # workover + rods
PUMP_UNSEAT_COST = 300000.0
DOWNTIME_DAYS_ROD = 5
DOWNTIME_DAYS_UNSEAT = 3

CARD_CLASSES = ["normal", "fluid_pound", "gas_interference", "rod_float",
                "pump_unseated", "tv_leak", "sv_leak"]

FIELD_START = "2019-01-01"
FIELD_END = "2026-06-30"
N_WELLS = 20
