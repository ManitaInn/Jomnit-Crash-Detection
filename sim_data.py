"""
Synthetic MPU-6050 scenario generator (PC only).

Why: you cannot collect real motorcycle crashes. This simulates raw IMU
streams (gravity + noise + gyro bias + impact spikes + slide vibration) for
crash and non-crash events, runs them through the REAL CrashDetector, and
captures the ML feature vector at the moment the rule gate fires.

IMPORTANT: scenario shapes/magnitudes are engineering assumptions, not
measurements. Metrics on this data are optimistic. Replace/augment with real
logs (see train_model.py --logs) as soon as you have field data.
"""
import math
import random
from crash_algo import CrashDetector, Config

DT = 0.01
BIAS = (0.6, -1.5, -0.4)

# label 1 = crash/fall, 0 = not a crash
CRASH_KINDS = ("lowside", "highside", "tumble", "tipover")
HARD_NEG_KINDS = ("maintenance", "transport", "extreme_lean")   # pass the rule gate
NORMAL_KINDS = ("idle", "cornering", "braking", "pothole", "speedbumps", "sidestand_push")
LABEL = {k: 1 for k in CRASH_KINDS}
LABEL.update({k: 0 for k in HARD_NEG_KINDS + NORMAL_KINDS})


# ------------------------------------------------------------ helpers
def _clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def smooth(x):
    x = _clip01(x)
    return x * x * (3 - 2 * x)


def ease_in(x):
    x = _clip01(x)
    return x * x


def spike(rng, t0, g, dur):
    """extra acceleration vector (fwd, side, up) of magnitude ~g at time t0"""
    v = [rng.gauss(0, 0.5), rng.gauss(0, 0.8), rng.gauss(0.3, 0.8)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return (t0, dur, tuple(g * x / n for x in v))


def lead_roll(rng):
    """riding lean before a crash: slow oscillation 0..L0 (either side)"""
    sgn = rng.choice((-1, 1))
    L0 = rng.uniform(10, 45)
    P = rng.uniform(3, 7)
    ph = rng.uniform(0, 6.28)
    return sgn, (lambda t: sgn * L0 * (0.5 + 0.5 * math.sin(2 * math.pi * t / P + ph)))


def make_noise(vib_a, vib_g, until, slide=None):
    """vibration sigma before `until`; slide=(t0,t1,sig_a,sig_g) = sliding"""
    def f(t):
        if slide and slide[0] <= t < slide[1]:
            return slide[2], slide[3]
        return (vib_a, vib_g) if t < until else (0.02, 0.0)
    return f


def accel_weight(t_imp, ride_w=0.15):
    """fraction of the real lean the accelerometer 'sees'. While riding in a
    corner centripetal force hides most of the lean (ride_w ~ 0.15). After the
    bike stops moving the accelerometer sees the full tilt."""
    def f(t):
        if t < t_imp:
            return ride_w
        return min(1.0, ride_w + (1 - ride_w) * (t - t_imp))
    return f


# ------------------------------------------------------------ scenarios
def make_scenario(kind, rng):
    s = dict(kind=kind, label=LABEL[kind], mount=rng.uniform(-15, 15),
             spikes=[], fwd=lambda t: 0.0, w=lambda t: 1.0)

    if kind == "lowside":
        sgn, lead = lead_roll(rng)
        t_fall = rng.uniform(3, 6)
        d = rng.uniform(0.3, 1.2)
        L = lead(t_fall)
        F = sgn * rng.uniform(70, 100)
        s["roll"] = lambda t: lead(t) if t < t_fall else L + (F - L) * ease_in((t - t_fall) / d)
        t_imp = t_fall + d * 0.6
        s["spikes"] = [spike(rng, t_imp, rng.uniform(5, 14), rng.uniform(0.03, 0.07))]
        for _ in range(rng.randint(0, 3)):
            s["spikes"].append(spike(rng, t_imp + rng.uniform(0.1, 1.0),
                                     rng.uniform(3, 8), rng.uniform(0.03, 0.07)))
        slide = (t_imp, t_imp + rng.uniform(1.5, 4), rng.uniform(0.2, 0.6), rng.uniform(20, 80))
        s["noise"] = make_noise(rng.uniform(0.05, 0.15), rng.uniform(1, 5), t_imp, slide)
        s["w"] = accel_weight(t_imp)
        s["t_event"], s["dur"] = t_fall, t_fall + d + 14

    elif kind == "highside":
        sgn, lead = lead_roll(rng)
        t_fall = rng.uniform(3, 6)
        L = lead(t_fall)
        d1, d2 = rng.uniform(0.15, 0.3), rng.uniform(0.3, 0.6)
        O = -sgn * rng.uniform(15, 35)
        F = sgn * rng.uniform(80, 110)

        def roll(t):
            if t < t_fall:
                return lead(t)
            if t < t_fall + d1:
                return L + (O - L) * smooth((t - t_fall) / d1)
            return O + (F - O) * ease_in((t - t_fall - d1) / d2)
        s["roll"] = roll
        t_imp = t_fall + d1 + d2 * 0.7
        s["spikes"] = [spike(rng, t_imp, rng.uniform(8, 20), rng.uniform(0.03, 0.07))]
        for _ in range(rng.randint(1, 3)):
            s["spikes"].append(spike(rng, t_imp + rng.uniform(0.1, 0.8),
                                     rng.uniform(4, 10), rng.uniform(0.03, 0.07)))
        slide = (t_imp, t_imp + rng.uniform(1.0, 3), rng.uniform(0.3, 0.7), rng.uniform(30, 90))
        s["noise"] = make_noise(rng.uniform(0.05, 0.15), rng.uniform(1, 5), t_imp, slide)
        s["w"] = accel_weight(t_imp)
        s["t_event"], s["dur"] = t_fall, t_fall + d1 + d2 + 14

    elif kind == "tumble":
        sgn, lead = lead_roll(rng)
        t_fall = rng.uniform(3, 6)
        d = rng.uniform(1.0, 2.2)
        L = lead(t_fall)
        F = sgn * rng.uniform(150, 270)
        s["roll"] = lambda t: lead(t) if t < t_fall else L + (F - L) * smooth((t - t_fall) / d)
        s["spikes"] = [spike(rng, t_fall + rng.uniform(0.1, d), rng.uniform(5, 12),
                             rng.uniform(0.03, 0.07)) for _ in range(rng.randint(2, 4))]
        slide = (t_fall, t_fall + d + rng.uniform(0.5, 2), rng.uniform(0.3, 0.8), rng.uniform(30, 100))
        s["noise"] = make_noise(rng.uniform(0.05, 0.15), rng.uniform(1, 5), t_fall, slide)
        s["w"] = accel_weight(t_fall)
        s["t_event"], s["dur"] = t_fall, t_fall + d + 14

    elif kind == "tipover":      # low-speed / parked tip-over
        sgn = rng.choice((-1, 1))
        t_fall = rng.uniform(2, 5)
        d = rng.uniform(0.6, 3.5)
        F = sgn * rng.uniform(65, 95)
        s["roll"] = lambda t: sgn * 3 + (F - sgn * 3) * ease_in((t - t_fall) / d)
        s["spikes"] = [spike(rng, t_fall + d, rng.uniform(1.5, 6), rng.uniform(0.03, 0.08))]
        s["noise"] = make_noise(0.03, 0.5, 1e9)
        s["t_event"], s["dur"] = t_fall, t_fall + d + 14

    elif kind == "maintenance":  # bike deliberately laid on its side, slowly
        sgn = rng.choice((-1, 1))
        t0 = rng.uniform(2, 4)
        d = rng.uniform(2, 12)       # 2-3.5 s overlaps slow real tip-overs
        F = sgn * rng.uniform(65, 95)
        s["roll"] = lambda t: F * smooth((t - t0) / d)
        s["spikes"] = [spike(rng, t0 + d, rng.uniform(1.2, 3), rng.uniform(0.04, 0.1))]
        for _ in range(rng.randint(0, 3)):     # tools banged on the frame
            s["spikes"].append(spike(rng, t0 + d * rng.uniform(0.6, 1.0) + rng.uniform(0, 6),
                                     rng.uniform(2, 6.5), rng.uniform(0.03, 0.1)))
        s["noise"] = make_noise(0.03, 0.5, 1e9)
        s["t_event"], s["dur"] = t0, t0 + d + 14

    elif kind == "transport":    # tilted on a trailer / truck, bumpy
        sgn = rng.choice((-1, 1))
        t0 = rng.uniform(1, 3)
        d = rng.uniform(3, 6)
        F = sgn * rng.uniform(60, 80)
        s["roll"] = lambda t: F * smooth((t - t0) / d)
        tt = t0 + d * 0.3
        while tt < t0 + d + 9:
            s["spikes"].append(spike(rng, tt, rng.uniform(2.5, 6), rng.uniform(0.03, 0.08)))
            tt += rng.uniform(0.8, 2.5)
        s["noise"] = make_noise(rng.uniform(0.15, 0.4), rng.uniform(5, 25), 1e9)
        s["t_event"], s["dur"] = t0, t0 + d + 14

    elif kind == "extreme_lean":  # very hard lean + pothole, then recover
        sgn = rng.choice((-1, 1))
        t0 = rng.uniform(3, 6)
        d = rng.uniform(1.5, 3)
        peak = sgn * rng.uniform(62, 70)
        hold = rng.uniform(0.6, 1.5)
        back = rng.uniform(1.0, 2.0)

        def roll(t):
            if t < t0 + d:
                return peak * smooth((t - t0) / d)
            if t < t0 + d + hold:
                return peak
            return peak * (1 - smooth((t - t0 - d - hold) / back)) + sgn * 8 * smooth((t - t0 - d - hold) / back)
        s["roll"] = roll
        s["spikes"] = [spike(rng, t0 + d + rng.uniform(0, hold), rng.uniform(4, 8), 0.04)]
        s["noise"] = make_noise(rng.uniform(0.05, 0.15), rng.uniform(1, 5), 1e9)
        s["w"] = lambda t: 0.5
        s["t_event"], s["dur"] = t0, t0 + d + hold + back + 4

    elif kind == "idle":
        s["roll"] = lambda t: 0.0
        s["noise"] = make_noise(0.03, 0.5, 1e9)
        s["t_event"], s["dur"] = 0, 15

    elif kind == "cornering":
        _, lead = lead_roll(rng)
        amp = rng.uniform(0.7, 1.1)
        s["roll"] = lambda t: lead(t) * amp * 1.1
        s["w"] = lambda t: 0.15
        s["noise"] = make_noise(rng.uniform(0.05, 0.2), rng.uniform(1, 6), 1e9)
        s["spikes"] = [spike(rng, rng.uniform(2, 14), rng.uniform(3, 6), 0.04)
                       for _ in range(rng.randint(0, 3))]
        s["t_event"], s["dur"] = 0, 20

    elif kind == "braking":
        b = rng.uniform(0.5, 1.0)
        t0 = rng.uniform(2, 5)
        s["roll"] = lambda t: 0.0
        s["fwd"] = lambda t: -b if t0 < t < t0 + 3 else 0.0
        s["noise"] = make_noise(0.1, 2, 1e9)
        s["t_event"], s["dur"] = 0, 12

    elif kind == "pothole":
        s["roll"] = lambda t: 2 * math.sin(t)
        s["spikes"] = [spike(rng, rng.uniform(2, 10), rng.uniform(3, 9), rng.uniform(0.03, 0.08))
                       for _ in range(rng.randint(1, 3))]
        s["noise"] = make_noise(0.1, 2, 1e9)
        s["t_event"], s["dur"] = 0, 14

    elif kind == "speedbumps":
        s["roll"] = lambda t: 1.5 * math.sin(2 * t)
        s["spikes"] = [spike(rng, 2 + 2.5 * i, rng.uniform(2, 4), rng.uniform(0.08, 0.15))
                       for i in range(5)]
        s["noise"] = make_noise(0.1, 2, 1e9)
        s["t_event"], s["dur"] = 0, 16

    elif kind == "sidestand_push":
        lean = rng.uniform(10, 30)
        s["roll"] = lambda t: lean * smooth((t - 2) / 1.0)
        s["spikes"] = [spike(rng, 4, rng.uniform(3, 5.5), 0.05)]
        s["noise"] = make_noise(0.03, 0.5, 1e9)
        s["t_event"], s["dur"] = 0, 14
    else:
        raise ValueError(kind)
    return s


# ------------------------------------------------------------ IMU sampling
def _imu(spec, t, rng, calib=False):
    mount = spec["mount"]
    if calib:
        r, rate, w, fwd, sp = 0.0, 0.0, 1.0, 0.0, ()
        sa, sg = 0.02, 0.0
    else:
        r = spec["roll"](t)
        rate = (r - spec["roll"](t - DT)) / DT
        w = spec["w"](t)
        fwd = spec["fwd"](t)
        sp = spec["spikes"]
        sa, sg = spec["noise"](t)
    ra = math.radians(r * w + mount)
    a_f, a_s, a_u = fwd, math.sin(ra), math.cos(ra)
    for (t0, dur, v) in sp:
        if t0 <= t < t0 + dur:
            a_f += v[0]; a_s += v[1]; a_u += v[2]
    ax = a_f + rng.gauss(0, sa + 0.02)
    ay = a_s + rng.gauss(0, sa + 0.02)
    az = a_u + rng.gauss(0, sa + 0.02)
    gx = rate + BIAS[0] + rng.gauss(0, 0.3 + sg)
    gy = BIAS[1] + rng.gauss(0, 0.3 + sg)
    gz = BIAS[2] + rng.gauss(0, 0.3 + sg)
    c16 = lambda v: max(-16.0, min(16.0, v))        # +/-16 g sensor range
    c1k = lambda v: max(-1000.0, min(1000.0, v))    # +/-1000 deg/s
    return c16(ax), c16(ay), c16(az), c1k(gx), c1k(gy), c1k(gz)


def run(spec, rng, classifier=None, stop_on_trigger=True):
    """Run one scenario through the real detector."""
    det = CrashDetector(Config, classifier)
    ok, _ = det.calibrate([_imu(spec, 0, rng, calib=True) for _ in range(200)])
    t = 0.0
    out = dict(trigger=False, t_trigger=None, reason=None, features=None,
               rejects=0, events=[])
    while t < spec["dur"]:
        ev = det.update(*_imu(spec, t, rng), DT)
        if ev:
            out["events"].append((round(t, 2), ev))
        if ev == "CRASH_SUSPECTED":
            out.update(trigger=True, t_trigger=t, reason=det.reason,
                       features=det.gate_features)
            if stop_on_trigger:
                break
        t += DT
    out["rejects"] = det.ml_rejects
    out["features"] = out["features"] or det.gate_features
    return out


# ------------------------------------------------------------ datasets
GATE_WEIGHTS = {"lowside": 1.0, "highside": 1.0, "tumble": 0.7, "tipover": 1.0,
                "maintenance": 1.5, "transport": 1.0, "extreme_lean": 1.5}


def build_dataset(n=150, seed=0, verbose=False):
    """Feature vectors captured when the rule gate fires.
    Returns X (list of lists), y (list), kinds (list)."""
    rng = random.Random(seed)
    X, y, kinds = [], [], []
    stats = {}
    for kind, wgt in GATE_WEIGHTS.items():
        cnt = int(n * wgt)
        fired = 0
        for _ in range(cnt):
            spec = make_scenario(kind, rng)
            r = run(spec, rng)
            if r["trigger"]:
                X.append(r["features"])
                y.append(LABEL[kind])
                kinds.append(kind)
                fired += 1
        stats[kind] = (fired, cnt)
    if verbose:
        for k, (f, c) in stats.items():
            print("  gate fired %3d/%3d  %s" % (f, c, k))
    return X, y, kinds


def evaluate_end_to_end(classifier, n=40, seed=123):
    """Run every scenario type through the full detector.
    Returns {kind: dict(n, triggers, rejects, failsafe, delays)}"""
    rng = random.Random(seed)
    res = {}
    for kind in CRASH_KINDS + HARD_NEG_KINDS + NORMAL_KINDS:
        r_ = dict(n=n, triggers=0, rejects=0, failsafe=0, delays=[])
        for _ in range(n):
            spec = make_scenario(kind, rng)
            r = run(spec, rng, classifier)
            r_["rejects"] += r["rejects"] > 0
            if r["trigger"]:
                r_["triggers"] += 1
                r_["failsafe"] += "failsafe" in (r["reason"] or "")
                r_["delays"].append(r["t_trigger"] - spec["t_event"])
        res[kind] = r_
    return res


def print_end_to_end(res, title):
    print("\n%s" % title)
    print("  %-15s %-6s %9s %9s %9s %12s" % ("scenario", "label", "triggered", "ML-vetoed", "failsafe", "median delay"))
    for kind, r in res.items():
        d = sorted(r["delays"])
        med = "%.1fs" % d[len(d) // 2] if d else "-"
        print("  %-15s %-6s %4d/%-4d %9d %9d %12s" % (
            kind, "CRASH" if LABEL[kind] else "ok", r["triggers"], r["n"],
            r["rejects"], r["failsafe"], med))
    c = [r for k, r in res.items() if LABEL[k] == 1]
    nn = [r for k, r in res.items() if LABEL[k] == 0]
    rec = sum(r["triggers"] for r in c) / sum(r["n"] for r in c)
    tot = sum(r["n"] for r in nn)
    fa = sum(r["triggers"] for r in nn) / tot
    fs = sum(r["failsafe"] for r in nn) / tot
    print("  -> crash detection rate: %.1f%%   false-trigger rate (non-crash): %.1f%%"
          "  (of which failsafe, bike down >%ds: %.1f%%)" % (100 * rec, 100 * fa, Config.FAILSAFE_S, 100 * fs))
    return rec, fa


if __name__ == "__main__":
    X, y, k = build_dataset(n=20, seed=0, verbose=True)
    print(len(X), "samples; crash fraction %.2f" % (sum(y) / len(y)))


def write_log(spec, rng, path, label_hz=50):
    """Write a CSV in the same format main.py logs (for testing train_model --logs)."""
    cal = [_imu(spec, 0, rng, calib=True) for _ in range(200)]
    with open(path, "w") as f:
        f.write("t_ms,ax,ay,az,gx,gy,gz,roll,mag\n")
        t = -2.0
        for s in cal[::2]:           # 2 s of calibration at 50 Hz
            f.write("%d,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,0,1\n" % ((t + 2) * 1000, *s))
            t += 0.02
        t = 0.0
        while t < spec["dur"]:
            s = _imu(spec, t, rng)
            f.write("%d,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,0,1\n" % ((t + 2) * 1000, *s))
            t += 1.0 / label_hz
