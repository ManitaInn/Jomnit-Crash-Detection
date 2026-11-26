"""
Threshold tuning tool (PC).  Usage:
    python replay_log.py ride_log.csv [roll_thr] [impact_g]

Feed it a CSV recorded by main.py (LOG_TO_FILE = True). The first 2 s must be
the bike upright and still (used for calibration). It replays the whole ride
through the detector and reports peak roll, peak g and every trigger, so you
can pick ROLL_THRESHOLD_DEG / IMPACT_G with real data (Oct 21 - Nov 17 task).
"""
import csv, sys
from crash_algo import CrashDetector, Config
try:
    import ml_classifier
    clf = ml_classifier.predict
except ImportError:
    clf = None

path = sys.argv[1]
if len(sys.argv) > 2:
    Config.ROLL_THRESHOLD_DEG = float(sys.argv[2])
if len(sys.argv) > 3:
    Config.IMPACT_G = float(sys.argv[3])

rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append([float(r[k]) for k in ("t_ms", "ax", "ay", "az", "gx", "gy", "gz")])

det = CrashDetector(Config, clf)
cal = [tuple(r[1:7]) for r in rows if r[0] < 2000]
ok, msg = det.calibrate(cal)
print("calibration:", msg)

max_roll = 0.0
max_g = 0.0
triggers = []
prev_t = rows[0][0]
for r in rows:
    dt = (r[0] - prev_t) / 1000.0
    prev_t = r[0]
    ev = det.update(*r[1:7], dt)
    max_roll = max(max_roll, abs(det.roll))
    max_g = max(max_g, det.mag)
    if ev == "ML_REJECTED":
        triggers.append((r[0] / 1000.0, "gate fired, ML vetoed"))
    if ev == "CRASH_SUSPECTED":
        triggers.append((r[0] / 1000.0, det.reason))
        det.cancel()                       # keep replaying the rest of the ride

print("peak |roll| = %.1f deg   peak |a| = %.2f g" % (max_roll, max_g))
print("triggers:", triggers if triggers else "none")
print("Tip: a normal ride should show peak roll well below %.0f deg "
      "and zero triggers." % Config.ROLL_THRESHOLD_DEG)
