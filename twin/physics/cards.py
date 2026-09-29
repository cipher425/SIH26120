"""Dynamometer-card synthesis (surface + computed downhole card).

The surface load is built from the physical components of a heavy-oil rod pump:
buoyed rod weight, plunger fluid load (picked up / released through rod stretch),
viscous drag (sign-opposite to rod velocity), inertia, a damped rod-string stress
wave after each load change, and pump-condition specific fluid-load shapes.
"""
import math
import numpy as np

from twin import config as C

N_PTS = 128


def _kinematics(stroke, down_frac, n=N_PTS):
    """Polished-rod position/velocity/acceleration over one cycle, VFD-shaped."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    up = t < (1.0 - down_frac)
    pos = np.where(up,
                   stroke / 2 * (1 - np.cos(np.pi * t / (1 - down_frac))),
                   stroke / 2 * (1 + np.cos(np.pi * (t - (1 - down_frac)) / down_frac)))
    return t, pos, up


def synth_card(cls, stroke, spm, down_frac, W_rb, Fo, c_visc, fillage=1.0,
               rng=None, noise=0.012, severity=None):
    """Returns dict(position, load, pump_position, pump_load) in metres / Newtons."""
    rng = rng or np.random.default_rng()
    t, pos, up = _kinematics(stroke, down_frac)
    period = 60.0 / max(spm, 0.1)
    vel = np.gradient(pos, t * period)
    acc = np.gradient(vel, t * period)
    stretch = min(0.4 * stroke, Fo * 950.0 / (C.E_STEEL * C.ROD_AREA))
    sev = rng.uniform(0.3, 0.9) if severity is None else severity

    s_top = stroke - pos                       # distance travelled down from top
    pump = np.zeros_like(pos)
    # upstroke: load picked up over stretch
    pickup = np.clip(pos / max(stretch, 1e-3), 0, 1)
    release = np.clip(1 - s_top / max(stretch, 1e-3), 0, 1)
    pump[up] = Fo * pickup[up]
    pump[~up] = Fo * release[~up]

    if cls == "fluid_pound":
        fill = min(fillage, 1 - 0.25 * sev) if fillage < 1 else 1 - 0.55 * sev
        s_fp = (1 - fill) * (stroke - stretch)
        drop = np.clip(1 - (s_top - s_fp) / (0.06 * stroke), 0, 1)
        pump[~up] = Fo * drop[~up]
    elif cls == "gas_interference":
        fill = 1 - 0.55 * sev
        s_g = (1 - fill) * (stroke - stretch) + stretch
        x = np.clip(s_top / s_g, 0, 1)
        pump[~up] = Fo * (1 - x[~up] ** 0.6)
    elif cls == "pump_unseated":
        pump[:] = Fo * 0.08 * (1 + 0.3 * np.sin(2 * np.pi * t))
    elif cls == "tv_leak":
        pump[up] = Fo * pickup[up] * (1 - 0.45 * sev * (pos[up] / stroke))
    elif cls == "sv_leak":
        slow = np.clip(1 - s_top / (stretch + 0.45 * sev * stroke), 0, 1)
        pump[~up] = Fo * slow[~up]

    m_rod = C.ROD_MASS_PER_M * 950.0
    load = W_rb + pump + c_visc * vel + 0.7 * m_rod * acc

    # damped stress wave after pickup and release
    wn = 2 * np.pi * 3.2
    t_up0 = 0.0
    t_dn0 = (1 - down_frac)
    tt = t * period
    for t0, amp in ((t_up0, 0.10 * Fo), (t_dn0 * period, -0.06 * Fo)):
        dt = tt - t0
        wave = np.where(dt > 0, amp * np.exp(-dt * 1.8) * np.sin(wn * dt), 0.0)
        load += wave

    if cls == "rod_float":
        # extra drag so downstroke drag exceeds buoyed weight
        extra = W_rb * (0.25 + 0.6 * sev)
        dn_load = load[~up] - extra * np.sin(np.pi * np.clip((t[~up] - (1 - down_frac)) / down_frac, 0, 1))
        load[~up] = dn_load
    float_mask = load < 0.03 * W_rb
    if float_mask.any():
        # carrier-bar separation: load reads ~0, then impact on next pickup
        load[float_mask] = 0.03 * W_rb * rng.uniform(0.5, 1.5, float_mask.sum())
        k = max(2, N_PTS // 30)
        load[:k] += W_rb * 0.4 * np.linspace(1, 0.2, k)

    load = load * (1 + rng.normal(0, noise, N_PTS))
    pump_pos = np.clip(pos - stretch * (pump / max(Fo, 1)), 0, None)
    return {"position": pos, "load": load, "pump_position": pump_pos, "pump_load": pump}


def card_for_state(state_cls, stroke, spm, down_frac, W_rb, Fo, c_visc, fillage, rng=None):
    return synth_card(state_cls, stroke, spm, down_frac, W_rb, Fo, c_visc, fillage=fillage,
                      rng=rng, severity=None if state_cls != "normal" else 0.0)


def normalise_card(position, load, stroke, W_rb, Fo):
    """2 x N array for the CNN: position/stroke and load/(W_rb+Fo)."""
    return np.stack([np.asarray(position) / stroke,
                     np.asarray(load) / (W_rb + Fo)]).astype(np.float32)
