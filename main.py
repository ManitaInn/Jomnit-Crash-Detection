# Jomnit - ESP32 (MicroPython) crash detection firmware
# Wiring: VCC>3V3  GND>GND  SCL>D5 (GPIO23)  SDA>D4 (GPIO22)
# Optional cancel button: D6 (GPIO16) <-> GND
# Upload BOTH main.py and crash_algo.py to the board.
import time
import struct
import os
from machine import Pin, SoftI2C
from crash_algo import CrashDetector, Config

try:                                   # decision tree made by train_model.py
    import ml_classifier
    classifier = ml_classifier.predict
    print("ML confirm: ON (ml_classifier.py)")
except ImportError:
    classifier = None
    print("ML confirm: OFF (rule-based only) - copy ml_classifier.py to enable")

# ---------------- settings ----------------
SDA_PIN, SCL_PIN = 22, 23
BUTTON_PIN = 16            # None to disable
# --- data collection mode (for training the ML model with REAL data) ---
# Set LOG_TO_FILE = True, pick a label, ride/do the test, then copy the
# log_<label>_<n>.csv files to the PC (Thonny / mpremote) into one folder:
#   LOG_LABEL = "normal"    ordinary riding, braking, cornering, potholes
#   LOG_LABEL = "pothole"   speed bumps, potholes, kerb hits
#   LOG_LABEL = "laydown"   bike deliberately laid on its side (maintenance)
#   LOG_LABEL = "transport" on a trailer / truck
#   LOG_LABEL = "fall"      simulated fall / tip-over (use a mount you can safely drop)
# START every recording with the bike UPRIGHT and STILL for 2 s (calibration).
LOG_TO_FILE = False
LOG_LABEL = "normal"
LOOP_US = 10000            # 100 Hz
PRINT_EVERY_MS = 500

# Desk-demo helpers: tapping a board on a table rarely reaches 4 g.
# Config.IMPACT_G = 2.5     # uncomment for desk demo
Config.DEMO_MODE = True     # False for the real 45 s countdown
# Config.FWD, Config.SIDE, Config.UP = 'X', 'Y', 'Z'   # set to match mounting

# ---------------- MPU-6050 ----------------
ADDR = 0x68
ACC_LSB = 2048.0           # +/-16 g
GYR_LSB = 32.8             # +/-1000 deg/s

i2c = SoftI2C(sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=400000)


def mpu_init():
    i2c.writeto_mem(ADDR, 0x6B, b'\x00')   # wake
    time.sleep_ms(100)
    i2c.writeto_mem(ADDR, 0x19, b'\x09')   # sample rate 1k/(1+9) = 100 Hz
    i2c.writeto_mem(ADDR, 0x1A, b'\x03')   # low-pass ~44 Hz (cuts engine vibration)
    i2c.writeto_mem(ADDR, 0x1B, b'\x10')   # gyro +/-1000 deg/s
    i2c.writeto_mem(ADDR, 0x1C, b'\x18')   # accel +/-16 g (crash impacts > 2 g!)


def read_imu():
    d = i2c.readfrom_mem(ADDR, 0x3B, 14)   # accel, temp, gyro in ONE read
    ax, ay, az, _t, gx, gy, gz = struct.unpack('>hhhhhhh', d)
    return (ax / ACC_LSB, ay / ACC_LSB, az / ACC_LSB,
            gx / GYR_LSB, gy / GYR_LSB, gz / GYR_LSB)


# ---------------- hooks for BLE / app (DARO + MARITH) ----------------
def send_to_phone(msg):
    """Replace with BLE notify. Messages: CRASH_SUSPECTED, ALERT, CANCELLED,
    plus {"roll":..,"g":..} telemetry if wanted."""
    print("BLE>", msg)


def phone_cancel_requested():
    """Return True when the app sends 'cancel' over BLE (DARO to wire up)."""
    return False


# ---------------- startup ----------------
try:
    mpu_init()
except Exception as e:
    print("MPU init failed:", e)
    raise

det = CrashDetector(Config, classifier)
print("Keep the bike upright and STILL - calibrating 2 s...")
time.sleep_ms(500)
samples = []
for _ in range(200):
    samples.append(read_imu())
    time.sleep_ms(10)
ok, msg = det.calibrate(samples)
print("Calibration:", msg)
while not ok:
    print("Retrying calibration...")
    time.sleep_ms(1000)
    samples = []
    for _ in range(200):
        samples.append(read_imu())
        time.sleep_ms(10)
    ok, msg = det.calibrate(samples)
    print("Calibration:", msg)

button = Pin(BUTTON_PIN, Pin.IN, Pin.PULL_UP) if BUTTON_PIN is not None else None
logf = None
if LOG_TO_FILE:
    existing = os.listdir()
    k = 1
    while "log_%s_%d.csv" % (LOG_LABEL, k) in existing:
        k += 1
    LOG_FILE = "log_%s_%d.csv" % (LOG_LABEL, k)
    logf = open(LOG_FILE, "w")
    logf.write("t_ms,ax,ay,az,gx,gy,gz,roll,mag\n")
    print("Logging to", LOG_FILE)
print("Monitoring. Roll threshold: %.0f deg" % Config.ROLL_THRESHOLD_DEG)

# ---------------- main loop ----------------
t0 = time.ticks_ms()
last_us = time.ticks_us()
last_print = t0
n = 0
while True:
    start = time.ticks_us()
    dt = time.ticks_diff(start, last_us) / 1e6
    last_us = start
    try:
        s = read_imu()
    except Exception as e:
        print("I2C read error:", e)
        time.sleep_ms(20)
        continue

    ev = det.update(s[0], s[1], s[2], s[3], s[4], s[5], dt)

    if ev == "ML_REJECTED":
        print("ML says NOT a crash (rule gate fired, vetoed). features:",
              [round(x, 1) for x in det.gate_features])
    elif ev == "CRASH_SUSPECTED":
        print("!!! CRASH SUSPECTED (%s, peak %.1fg) - countdown %ds" % (
            det.reason, det.peak_g, det.countdown_total))
        send_to_phone("CRASH_SUSPECTED")       # phone starts ringing
    elif ev == "ALERT":
        print("!!! NO CANCEL - SENDING EMERGENCY ALERT")
        send_to_phone("ALERT")                 # app -> Telegram bot + contacts

    if det.state == det.COUNTDOWN:
        if (button is not None and button.value() == 0) or phone_cancel_requested():
            if det.cancel():
                print("Cancelled (false alarm)")
                send_to_phone("CANCELLED")
    elif det.state == det.ALERT:
        if button is not None and button.value() == 0:
            det.reset()
            print("Reset to monitoring")

    now_ms = time.ticks_ms()
    if logf and n % 2 == 0:                      # 50 Hz log
        logf.write("%d,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%.3f\n" % (
            time.ticks_diff(now_ms, t0), s[0], s[1], s[2], s[3], s[4], s[5],
            det.roll, det.mag))
        if n % 100 == 0:
            logf.flush()
    n += 1

    if time.ticks_diff(now_ms, last_print) >= PRINT_EVERY_MS:
        last_print = now_ms
        print("roll=%6.1f deg | |a|=%.2fg | %s%s" % (
            det.roll, det.mag, det.NAMES[det.state],
            (" %.0fs left" % det.countdown_remaining())
            if det.state == det.COUNTDOWN else ""))

    spent = time.ticks_diff(time.ticks_us(), start)
    if spent < LOOP_US:
        time.sleep_us(LOOP_US - spent)
