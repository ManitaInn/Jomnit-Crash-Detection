"""
Jomnit crash detection core (NITA)  -  HYBRID: rule-based gate + ML confirm
---------------------------------------------------------------------------
Pure Python (no hardware imports) so the SAME file runs on the ESP32
(MicroPython) and on a PC for testing / training / replaying ride logs.

Pipeline:
  raw IMU -> gyro bias removal -> map to bike frame -> complementary filter
  -> roll angle -> RULE GATE (impact+tilt / tip-over)
  -> ML CONFIRM (decision tree on window features, optional)
  -> countdown state machine (MONITORING -> COUNTDOWN -> ALERT)

Units in:  accel = g,  gyro = deg/s,  dt = seconds
"""
import math
from array import array


class Config:
    # ---- Demo vs real ------------------------------------------------------
    DEMO_MODE = True            # True: 10 s countdown, False: 45 s
    COUNTDOWN_DEMO_S = 10
    COUNTDOWN_REAL_S = 45

    # ---- Sensor mounting (bike frame) -------------------------------------
    # Which sensor axis points FORWARD / to the SIDE / UP on the motorcycle.
    # Prefix with '-' to flip. The three must form a RIGHT-HANDED set,
    # i.e. SIDE = UP x FORWARD  (the code checks this for you).
    FWD = 'X'
    SIDE = 'Y'
    UP = 'Z'

    # ---- Complementary filter ---------------------------------------------
    ALPHA = 0.98                # gyro weight. time constant ~ 0.49 s @100 Hz
    ACC_TRUST_BAND = 0.25       # trust accel only if | |a| - 1g | < this

    # ---- Rule gate ---------------------------------------------------------
    ROLL_THRESHOLD_DEG = 60.0   # "bike is down" angle (street lean is <~50)
    IMPACT_G = 4.0              # |a| spike that counts as an impact
    IMPACT_WINDOW_S = 4.0       # impact must be this recent to count
    HOLD_IMPACT_S = 0.5         # tilt must persist this long (with impact)
    HOLD_NO_IMPACT_S = 3.0      # tilt must persist this long (no impact: tip-over)
    COOLDOWN_S = 10.0           # ignore triggers this long after a cancel

    # ---- ML confirm --------------------------------------------------------
    WINDOW_S = 6.0              # feature window (seconds before the gate fires)
    ML_RECHECK_S = 1.0          # if ML says "no", ask again this often
    FAILSAFE_S = 10.0           # ML can veto at most this long; then countdown starts


FEATURES = ("peak_g", "n_impacts", "max_roll", "roll_rate_max", "fall_time",
            "tilt_time", "post_motion", "post_mag_dev", "min_g", "since_peak")
BUF_N = 600                     # 6 s @ 100 Hz


# ---------------------------------------------------------------------------
def _axis(spec):
    sign = -1 if spec.startswith('-') else 1
    idx = 'XYZ'.index(spec.lstrip('-+').upper())
    return idx, sign


def _unit(spec):
    idx, sign = _axis(spec)
    v = [0, 0, 0]
    v[idx] = sign
    return v


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def wrap180(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class CrashDetector:
    MONITORING = 0
    COUNTDOWN = 1
    ALERT = 2
    NAMES = ("MONITORING", "COUNTDOWN", "ALERT")

    def __init__(self, cfg=Config, classifier=None):
        """classifier: function(list_of_features) -> True (crash) / False.
        None = rule-based only."""
        self.cfg = cfg
        self.classifier = classifier
        f, s, u = _unit(cfg.FWD), _unit(cfg.SIDE), _unit(cfg.UP)
        if _cross(f, s) != u:
            raise ValueError("FWD/SIDE/UP are not right-handed: flip the sign "
                             "of one axis (e.g. SIDE='-Y')")
        self._f, self._s, self._u = _axis(cfg.FWD), _axis(cfg.SIDE), _axis(cfg.UP)

        self.state = self.MONITORING
        self.roll = 0.0          # filtered roll, deg (0 = upright at calibration)
        self.roll_acc = 0.0      # accel-only roll, deg (debug/tuning)
        self.mag = 1.0           # |a| in g
        self.peak_g = 0.0
        self.reason = None
        self.gate_features = None   # feature vector at the last gate check
        self.ml_rejects = 0
        self.t = 0.0
        self._bias = (0.0, 0.0, 0.0)
        self._roll_off = 0.0
        self._calibrated = False
        self._last_impact = -1e9
        self._over_since = None
        self._cd_start = 0.0
        self._cool_until = -1.0
        self._next_ml = 0.0
        # ring buffer for the ML feature window
        self._b_mag = array('f', [1.0] * BUF_N)
        self._b_roll = array('f', [0.0] * BUF_N)
        self._b_gf = array('f', [0.0] * BUF_N)
        self._b_dt = array('f', [0.01] * BUF_N)
        self._head = 0
        self._n = 0

    # -- helpers -----------------------------------------------------------
    def _map(self, v):
        """sensor-frame 3-vector -> (forward, side, up)"""
        return (v[self._f[0]] * self._f[1],
                v[self._s[0]] * self._s[1],
                v[self._u[0]] * self._u[1])

    @property
    def countdown_total(self):
        c = self.cfg
        return c.COUNTDOWN_DEMO_S if c.DEMO_MODE else c.COUNTDOWN_REAL_S

    def countdown_remaining(self):
        if self.state != self.COUNTDOWN:
            return 0.0
        return max(0.0, self.countdown_total - (self.t - self._cd_start))

    # -- calibration (bike upright & still, ~2 s) ------------------------
    def calibrate(self, samples):
        """samples: list of (ax,ay,az,gx,gy,gz). Returns (ok, message)."""
        n = len(samples)
        if n < 20:
            return False, "need >= 20 samples"
        m = [sum(s[i] for s in samples) / n for i in range(6)]
        spread = max(max(abs(s[i] - m[i]) for s in samples) for i in (3, 4, 5))
        self._bias = (m[3], m[4], m[5])
        f, s, u = self._map((m[0], m[1], m[2]))
        self._roll_off = math.degrees(math.atan2(s, u))
        self.roll = 0.0
        self._calibrated = True
        g = math.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2)
        ok = spread < 5.0 and 0.9 < g < 1.1
        msg = "gravity=%.2fg gyro_bias=(%.2f,%.2f,%.2f) roll_offset=%.1fdeg%s" % (
            g, m[3], m[4], m[5], self._roll_off,
            "" if ok else "  WARNING: sensor moved or |g| off - recalibrate")
        return ok, msg

    # -- ML window features ------------------------------------------------
    def window_features(self):
        """Single backward pass over the last WINDOW_S seconds.
        Returns a list in FEATURES order. No numpy: runs on MicroPython."""
        c = self.cfg
        N = BUF_N
        cnt = self._n if self._n < N else N
        thr = c.IMPACT_G
        peak = 0.0
        peak_el = 0.0
        mn = 99.0
        mx_roll = 0.0
        mx_rate = 0.0
        n_imp = 0
        in_run = False
        tilt_t = 0.0
        post_cnt = 0
        post_gf = 0.0
        post_dev = 0.0
        el = 0.0                 # elapsed time before the newest sample
        phase = 0                # 0: in tilted run, 1: found upright, done w/ fall
        e20 = None
        e60 = None
        for k in range(cnt):
            if el > c.WINDOW_S:
                break
            i = (self._head - 1 - k) % N
            m = self._b_mag[i]
            r = self._b_roll[i]
            g = self._b_gf[i]
            ar = r if r >= 0 else -r
            ag = g if g >= 0 else -g
            if m > peak:
                peak = m
                peak_el = el
            if m < mn:
                mn = m
            if ar > mx_roll:
                mx_roll = ar
            if ag > mx_rate:
                mx_rate = ag
            if m > thr:
                if not in_run:
                    n_imp += 1
                    in_run = True
            elif m < thr * 0.6:
                in_run = False
            if ar > 45.0:
                tilt_t += self._b_dt[i]
            if el <= 1.0:
                post_cnt += 1
                post_gf += ag
                post_dev += abs(m - 1.0)
            if phase == 0:
                if ar < 20.0:
                    phase = 1
                    e20 = el
                elif ar > c.ROLL_THRESHOLD_DEG:
                    e60 = el
            el += self._b_dt[i]
        if phase == 0:
            fall = c.WINDOW_S    # never upright inside the window: slow/long tilt
        else:
            fall = e20 - (e60 if e60 is not None else 0.0)
            if fall < 0:
                fall = 0.0
        if post_cnt == 0:
            post_cnt = 1
        return [peak, float(n_imp), mx_roll, mx_rate, fall, tilt_t,
                post_gf / post_cnt, post_dev / post_cnt, mn, peak_el]

    # -- main update (call at ~100 Hz) -----------------------------------
    def update(self, ax, ay, az, gx, gy, gz, dt):
        """Returns 'CRASH_SUSPECTED', 'ALERT', 'ML_REJECTED' or None."""
        if not self._calibrated:
            raise RuntimeError("call calibrate() first")
        c = self.cfg
        if dt <= 0:
            return None
        if dt > 0.05:            # long stall: don't integrate garbage
            dt = 0.05
        self.t += dt

        # 1. bias-corrected gyro -> bike frame (only the roll rate is needed)
        b = self._bias
        gf, _, _ = self._map((gx - b[0], gy - b[1], gz - b[2]))

        # 2. accel -> bike frame, roll from gravity direction
        _, a_side, a_up = self._map((ax, ay, az))
        self.mag = math.sqrt(ax * ax + ay * ay + az * az)
        self.roll_acc = wrap180(math.degrees(math.atan2(a_side, a_up)) - self._roll_off)

        # 3. complementary filter (gyro-only while accel isn't ~1 g)
        pred = self.roll + gf * dt
        if abs(self.mag - 1.0) < c.ACC_TRUST_BAND:
            self.roll = wrap180(pred + (1.0 - c.ALPHA) * wrap180(self.roll_acc - pred))
        else:
            self.roll = wrap180(pred)

        # 3b. store in the ring buffer for ML features
        h = self._head
        self._b_mag[h] = self.mag
        self._b_roll[h] = self.roll
        self._b_gf[h] = gf
        self._b_dt[h] = dt
        self._head = (h + 1) % BUF_N
        self._n += 1

        # 4. impact + tilt bookkeeping
        if self.mag > c.IMPACT_G:
            self._last_impact = self.t
            if self.mag > self.peak_g:
                self.peak_g = self.mag
        if abs(self.roll) > c.ROLL_THRESHOLD_DEG:
            if self._over_since is None:
                self._over_since = self.t
            held = self.t - self._over_since
        else:
            self._over_since = None
            self._next_ml = 0.0
            held = 0.0

        # 5. state machine
        if self.state == self.MONITORING:
            if self.t < self._cool_until:
                return None
            recent_impact = (self.t - self._last_impact) < c.IMPACT_WINDOW_S
            if recent_impact and held >= c.HOLD_IMPACT_S:
                reason = "impact+tilt"
            elif held >= c.HOLD_NO_IMPACT_S:
                reason = "tip-over"
            else:
                return None
            # --- RULE GATE PASSED. ML confirmation (if a model is loaded) ---
            if self.classifier is not None:
                if held >= c.FAILSAFE_S:
                    reason += "+failsafe"      # ML cannot veto forever
                elif self.t >= self._next_ml:
                    self.gate_features = self.window_features()
                    if self.classifier(self.gate_features):
                        reason += "+ml"
                    else:
                        self.ml_rejects += 1
                        self._next_ml = self.t + c.ML_RECHECK_S
                        return "ML_REJECTED"
                else:
                    return None
            else:
                self.gate_features = self.window_features()
            self.reason = reason
            self.state = self.COUNTDOWN
            self._cd_start = self.t
            return "CRASH_SUSPECTED"

        if self.state == self.COUNTDOWN:
            if self.t - self._cd_start >= self.countdown_total:
                self.state = self.ALERT
                return "ALERT"
        return None

    # -- user / app actions ------------------------------------------------
    def cancel(self):
        """False alarm (button on device or 'cancel' from phone app)."""
        if self.state == self.COUNTDOWN:
            self.state = self.MONITORING
            self._over_since = None
            self._last_impact = -1e9
            self.peak_g = 0.0
            self._next_ml = 0.0
            self._cool_until = self.t + self.cfg.COOLDOWN_S
            return True
        return False

    def reset(self):
        """Back to monitoring from ALERT (after the emergency is handled)."""
        self.state = self.MONITORING
        self._over_since = None
        self._last_impact = -1e9
        self.peak_g = 0.0
        self._next_ml = 0.0
        self._cool_until = self.t + self.cfg.COOLDOWN_S
