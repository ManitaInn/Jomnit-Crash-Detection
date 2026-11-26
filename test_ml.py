"""
Tests for the ML (hybrid) layer.   python test_ml.py
Needs ml_classifier.py (run train_model.py first).
"""
import math, os, random, tempfile
import crash_algo
from crash_algo import CrashDetector, Config, FEATURES
import sim_data
from sim_data import make_scenario, run
import ml_classifier

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name + "  " + str(info))


class C(Config):
    DEMO_MODE = True


rng = random.Random(7)

# 1. feature order of the exported model matches the firmware
check("exported FEATURES == crash_algo.FEATURES", tuple(ml_classifier.FEATURES) == tuple(FEATURES))

# 2. feature extraction on a known pattern: 0->90 deg in 1.0 s, 8 g impact
spec = dict(kind="x", label=1, mount=0.0, spikes=[(5.6, 0.04, (0, 0, 7.0))],
            fwd=lambda t: 0.0, w=lambda t: 1.0, noise=lambda t: (0.02, 0.0),
            roll=lambda t: 90 * sim_data.smooth((t - 5.0) / 1.0) if t > 5 else 0.0,
            dur=12, t_event=5.0)
r = run(spec, rng)
f = dict(zip(FEATURES, r["features"]))
check("feature peak_g ~ 8 g", 6.5 < f["peak_g"] < 9.5, round(f["peak_g"], 1))
check("feature max_roll ~ 90 deg", 60 < f["max_roll"] < 100, round(f["max_roll"], 1))
check("feature fall_time short (< 1 s)", 0.1 < f["fall_time"] < 1.0, round(f["fall_time"], 2))
check("feature n_impacts == 1", f["n_impacts"] == 1.0, f["n_impacts"])

# 3. slow tilt must give fall_time = full window (never-upright rule)
spec2 = dict(spec, spikes=[], roll=lambda t: 90 * sim_data.smooth((t - 2.0) / 12.0) if t > 2 else 0.0,
             dur=30, t_event=2.0)
det = CrashDetector(C, classifier=lambda f: False)
det.calibrate([sim_data._imu(spec2, 0, rng, calib=True) for _ in range(200)])
t = 0.0
last = None
while t < 20:
    det.update(*sim_data._imu(spec2, t, rng), 0.01); t += 0.01
f = dict(zip(FEATURES, det.window_features()))
check("long slow tilt -> fall_time == WINDOW_S (6 s)", abs(f["fall_time"] - 6.0) < 1e-6, f["fall_time"])

# 4. FAILSAFE: classifier that always vetoes cannot block a tip-over forever
def run_with(clf, spec):
    return run(spec, rng, clf, stop_on_trigger=True)
tip = make_scenario("tipover", random.Random(3))
r = run_with(lambda f: False, tip)
check("classifier always 'no' -> failsafe still triggers", r["trigger"] and "failsafe" in r["reason"], (r["reason"], r["t_trigger"]))
r2 = run_with(lambda f: True, tip)
check("classifier always 'yes' -> trigger, reason has +ml", r2["trigger"] and "+ml" in r2["reason"], r2["reason"])
check("failsafe triggers later than ML-approved", r["t_trigger"] > r2["t_trigger"] + 5, (round(r["t_trigger"], 1), round(r2["t_trigger"], 1)))
r3 = run_with(None, tip)
check("no classifier -> rule-only still works", r3["trigger"] and r3["reason"] == "tip-over", r3["reason"])

# 5. veto events surface
r = run_with(lambda f: False, make_scenario("maintenance", random.Random(4)))
check("veto reports ML_REJECTED events", any(e[1] == "ML_REJECTED" for e in r["events"]), r["events"][:3])

# 6. end-to-end with the shipped model
res = sim_data.evaluate_end_to_end(ml_classifier.predict, n=40, seed=2024)
rec, fa = sim_data.print_end_to_end(res, "END-TO-END (hybrid, fresh seed):")
check("crash detection rate >= 98%", rec >= 0.98, "%.1f%%" % (100 * rec))
non = [r for k, r in res.items() if sim_data.LABEL[k] == 0]
tot = sum(r["n"] for r in non)
non_fail = sum(r["triggers"] - r["failsafe"] for r in non) / tot
check("false triggers excluding >10 s-down failsafe <= 6%", non_fail <= 0.06, "%.1f%%" % (100 * non_fail))
for k in ("idle", "cornering", "braking", "pothole", "speedbumps", "sidestand_push", "extreme_lean"):
    check("  %s -> never triggers" % k, res[k]["triggers"] == 0, res[k]["triggers"])

# 7. real-log loader: write synthetic logs to CSV, reload, train path works
import train_model
d = tempfile.mkdtemp()
for i, (kind, lab) in enumerate((("lowside", "fall"), ("tipover", "fall"), ("maintenance", "laydown"),
                                 ("transport", "transport"), ("cornering", "normal"))):
    sim_data.write_log(make_scenario(kind, rng), rng, os.path.join(d, "log_%s_%d.csv" % (lab, i + 1)))
X, y, names, warns = train_model.load_logs(d)
check("log loader: gate-firing windows labelled correctly",
      sorted(names) == ["real:fall", "real:fall", "real:laydown", "real:transport"]
      and sorted(y) == [0, 0, 1, 1], (names, y, warns))
check("log loader: normal ride produced no false window", "real:normal" not in names)

print("\n%d/%d passed" % (sum(results), len(results)))
raise SystemExit(0 if all(results) else 1)
