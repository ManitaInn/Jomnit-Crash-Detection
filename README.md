# Jomnit Crash Detection

Motorcycle crash detection on an ESP32 + MPU-6050 IMU.
Hybrid approach: a rule-based gate (impact + roll angle) followed by a
decision-tree classifier that confirms or vetoes the event, then a countdown
(10 s demo / 45 s real) before an emergency alert is sent.

## How it works
1. Complementary filter fuses gyro and accelerometer into a roll angle.
2. Rule gate: impact spike (> 4 g) plus roll > 60 deg, or a long tip-over.
3. Decision tree (`ml_classifier.py`) classifies the event from 10 window features.
4. Countdown, with cancel via button or phone app; failsafe after 10 s.

## Wiring
| MPU-6050 | ESP32 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | D4 (GPIO22) |
| SCL | D5 (GPIO23) |

Optional cancel button: D6 (GPIO16) to GND.

## Run on the ESP32 (MicroPython)
Copy these three files to the board and run `main.py`:
`main.py`, `crash_algo.py`, `ml_classifier.py`.
Keep the sensor upright and still for 2 s at boot (calibration).

## Run on PC
```bash
pip install numpy scikit-learn
python test_crash_algo.py   # rule-based tests
python test_ml.py           # ML + end-to-end tests
python train_model.py       # retrain, writes ml_classifier.py
python train_model.py --logs ride_logs/   # include real logged rides
python replay_log.py ride_logs/log_normal_1.csv   # replay a ride
```

## Collecting real data
In `main.py` set `LOG_TO_FILE = True` and a `LOG_LABEL`
(normal, pothole, laydown, transport, fall). Files are saved as
`log_<label>_<n>.csv`. Start each recording upright and still for 2 s.

## Limitations
- The model is currently trained on simulated data, so reported metrics are
  optimistic. Real-world validation is future work.
- A bike laid down on purpose for more than 10 s still starts the countdown.
- Prototype only: a real product also needs GPS speed from the phone and a
  vibration-isolated mount.
