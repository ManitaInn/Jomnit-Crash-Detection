"""
Run on a PC:  python test_crash_algo.py
Simulates an MPU-6050 with noise + gyro bias for several scenarios and checks
the detector fires only when it should.
"""
import math, random
from crash_algo import CrashDetector, Config

random.seed(1)
HZ = 100
DT = 1.0 / HZ


class Cfg(Config):
    DEMO_MODE = True


def make_sensor(fwd, side, up):
    """returns fn(bike_vec_accel, bike_gyro_fwd) -> raw sensor tuple, given the
    mounting (same axis spec strings as Config)"""
    def idx(s):
        return 'XYZ'.index(s.lstrip('-')), (-1 if s.startswith('-') else 1)
    F, S, U = idx(fwd), idx(side), idx(up)

    def raw(a_f, a_s, a_u, g_f):
        a = [0, 0, 0]
        g = [0, 0, 0]
        a[F[0]] = a_f * F[1]; a[S[0]] = a_s * S[1]; a[U[0]] = a_u * U[1]
        g[F[0]] = g_f * F[1]
        return a, g
    return raw


def simulate(det, roll_fn, spike_fn=None, mount_off=0.0, secs=20, raw=None,
             cancel_at=None, accel_pitch_g=0.0):
    """roll_fn(t) = true roll deg. spike_fn(t) = extra g on the 'up' axis."""
    raw = raw or make_sensor('X', 'Y', 'Z')
    bias = (0.6, -1.5, -0.4)  # like your real data
    events = []
    t = 0.0
    prev = roll_fn(0)
    while t < secs:
        r = roll_fn(t)
        rate = (r - prev) / DT
        prev = r
        rr = math.radians(r + mount_off)
        a_s, a_u = math.sin(rr), math.cos(rr)
        sp = spike_fn(t) if spike_fn else 0.0
        a_u += sp
        a, g = raw(accel_pitch_g, a_s, a_u, rate)
        a = [x + random.gauss(0, 0.02) for x in a]
        g = [x + random.gauss(0, 0.3) + b for x, b in zip(g, bias)]
        ev = det.update(*a, *g, DT)
        if ev:
            events.append((round(t, 2), ev))
        if cancel_at is not None and abs(t - cancel_at) < DT / 2:
            det.cancel()
            events.append((round(t, 2), "cancelled"))
        t += DT
    return events


def fresh(mount_off=0.0, raw=None, cfg=Cfg, f='X', s='Y', u='Z'):
    class C(cfg):
        FWD, SIDE, UP = f, s, u
    det = CrashDetector(C)
    raw = raw or make_sensor(f, s, u)
    rr = math.radians(mount_off)
    samples = []
    bias = (0.6, -1.5, -0.4)
    for _ in range(200):
        a, g = raw(0, math.sin(rr), math.cos(rr), 0)
        samples.append(tuple(x + random.gauss(0, 0.02) for x in a) +
                       tuple(x + random.gauss(0, 0.3) + b for x, b in zip(g, bias)))
    ok, msg = det.calibrate(samples)
    assert ok, msg
    return det, raw


def ramp(t, t0, dur, a0, a1):
    if t <= t0: return a0
    if t >= t0 + dur: return a1
    return a0 + (a1 - a0) * (t - t0) / dur


results = []


def check(name, cond, info=""):
    results.append(cond)
    print(("PASS  " if cond else "FAIL  ") + name + "  " + str(info))


# 1. idle upright
det, raw = fresh()
ev = simulate(det, lambda t: 0, raw=raw)
check("idle upright -> no trigger", not ev, ev)

# 2. aggressive riding: +/-45 deg lean oscillation, vibration, no impact
det, raw = fresh()
ev = simulate(det, lambda t: 45 * math.sin(2 * math.pi * t / 4), raw=raw, secs=30)
check("hard cornering +/-45 deg -> no trigger", not ev, ev)

# 3. pothole: 6 g spike, upright
det, raw = fresh()
ev = simulate(det, lambda t: 0, lambda t: 5.0 if 5 < t < 5.04 else 0, raw=raw)
check("pothole 6g spike -> no trigger", not ev, ev)

# 4. side stand (15 deg for a long time)
det, raw = fresh(mount_off=0)
ev = simulate(det, lambda t: 15, raw=raw, secs=30)
check("side stand 15 deg -> no trigger", not ev, ev)

# 5. hard braking (1 g forward) - must not affect roll
det, raw = fresh()
ev = simulate(det, lambda t: 0, raw=raw, accel_pitch_g=0.9)
check("hard braking 0.9g -> no trigger", not ev, ev)

# 6. crash: 8 g impact then fall to 90 deg and lie still
det, raw = fresh()
ev = simulate(det, lambda t: ramp(t, 5.0, 0.8, 0, 90),
              lambda t: 7.0 if 5.0 < t < 5.05 else 0, raw=raw, secs=25)
names = [e[1] for e in ev]
check("crash impact+fall -> SUSPECTED then ALERT (10 s later)",
      names == ["CRASH_SUSPECTED", "ALERT"] and 5.5 < ev[0][0] < 7.0
      and abs((ev[1][0] - ev[0][0]) - 10) < 0.1, ev)
check("  reason = impact+tilt", det.reason == "impact+tilt", det.reason)

# 7. slow tip-over, no impact
det, raw = fresh()
ev = simulate(det, lambda t: ramp(t, 5.0, 1.0, 0, 85), raw=raw, secs=25)
check("tip-over no impact -> trigger after ~3 s hold",
      [e[1] for e in ev][:1] == ["CRASH_SUSPECTED"] and det.reason == "tip-over"
      and 8.0 < ev[0][0] < 9.5, ev)

# 8. false alarm cancelled during countdown, bike picked up
det, raw = fresh()
ev = simulate(det, lambda t: ramp(t, 5, 0.8, 0, 90) if t < 9 else ramp(t, 9, 1, 90, 0),
              lambda t: 7.0 if 5.0 < t < 5.05 else 0, raw=raw, secs=30, cancel_at=8.0)
check("cancel during countdown -> back to MONITORING, no ALERT",
      [e[1] for e in ev] == ["CRASH_SUSPECTED", "cancelled"]
      and det.state == det.MONITORING, ev)

# 9. mounting offset (sensor tilted 25 deg in enclosure) still works
det, raw = fresh(mount_off=25)
ev = simulate(det, lambda t: 0, raw=raw, mount_off=25)
check("25 deg mounting tilt calibrated out -> no trigger", not ev, ev)
det, raw = fresh(mount_off=25)
ev = simulate(det, lambda t: ramp(t, 5.0, 0.8, 0, -90),
              lambda t: 7.0 if 5.0 < t < 5.05 else 0, raw=raw, mount_off=25, secs=15)
check("  ...and crash to the LEFT still detected", ev and ev[0][1] == "CRASH_SUSPECTED", ev)

# 10. different axis mapping (right-handed): FWD=Z SIDE=X UP=Y
det, raw = fresh(f='Z', s='X', u='Y')
ev = simulate(det, lambda t: ramp(t, 5.0, 0.8, 0, 90),
              lambda t: 7.0 if 5.0 < t < 5.05 else 0,
              raw=make_sensor('Z', 'X', 'Y'), secs=15)
check("axis map Z/X/Y crash detected", ev and ev[0][1] == "CRASH_SUSPECTED", ev)

# 11. left-handed mapping must be rejected
try:
    class Bad(Config):
        FWD, SIDE, UP = 'X', 'Y', '-Z'
    CrashDetector(Bad)
    check("left-handed mapping rejected", False)
except ValueError:
    check("left-handed mapping rejected", True)

# 12. impact but bike stays upright (e.g. dropped / hit by object, recovers)
det, raw = fresh()
ev = simulate(det, lambda t: 0, lambda t: 9.0 if 5 < t < 5.05 else 0, raw=raw)
check("big impact but stays upright -> no trigger", not ev, ev)

print("\n%d/%d passed" % (sum(results), len(results)))
raise SystemExit(0 if all(results) else 1)
