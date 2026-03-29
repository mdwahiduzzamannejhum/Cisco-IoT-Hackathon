"""Generate sample IoT sensor data for the assistive technology device."""

import csv
import random
from datetime import datetime, timedelta

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Simulate 7 days of sensor readings sampled every 30 seconds during active hours
START_DATE = datetime(2026, 3, 9, 0, 0, 0)
ACTIVE_HOURS = range(8, 20)  # 8 AM to 8 PM
INTERVAL_SECONDS = 30
NUM_DAYS = 7

ALERT_TYPES = ["obstacle_near", "obstacle_medium", "fall_detected", "low_battery", "none"]
ALERT_WEIGHTS = [0.08, 0.15, 0.02, 0.05, 0.70]

MODES = ["navigation", "indoor_assist", "standby"]
MODE_WEIGHTS = [0.50, 0.35, 0.15]


def simulate_distance(hour: int) -> float:
    """Simulate ultrasonic distance sensor reading (cm)."""
    # More obstacles during peak pedestrian hours
    base = 120.0 if 9 <= hour <= 11 or 16 <= hour <= 18 else 200.0
    noise = random.gauss(0, 30)
    value = max(10.0, min(400.0, base + noise))
    return round(value, 1)


def simulate_battery(elapsed_minutes: float, charge_events: set) -> float:
    """Simulate battery percentage at a given elapsed time.

    Finds the last charge event before elapsed_minutes and calculates
    the remaining battery based on drain since that event.
    """
    drain_rate = 0.04  # % per minute when active
    past_charges = sorted(t for t in charge_events if t <= elapsed_minutes)
    last_charge = past_charges[-1] if past_charges else 0
    minutes_since_charge = elapsed_minutes - last_charge
    level = max(5.0, 100.0 - drain_rate * minutes_since_charge)
    return round(level, 1)


def generate_records() -> list:
    """Generate all sensor records."""
    records = []
    charge_event_minutes = {120, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 4000}
    elapsed_minutes = 0

    for day in range(NUM_DAYS):
        for hour in ACTIVE_HOURS:
            # Two readings per minute (every 30 seconds)
            for half_minute in range(120):
                ts = START_DATE + timedelta(
                    days=day, hours=hour, seconds=half_minute * INTERVAL_SECONDS
                )
                distance = simulate_distance(hour)
                battery = simulate_battery(elapsed_minutes, charge_event_minutes)
                alert = random.choices(ALERT_TYPES, weights=ALERT_WEIGHTS)[0]
                mode = random.choices(MODES, weights=MODE_WEIGHTS)[0]
                # Override alert based on distance
                if distance < 30:
                    alert = "obstacle_near"
                elif distance < 80:
                    alert = "obstacle_medium"

                records.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "day_of_week": ts.strftime("%A"),
                    "hour": hour,
                    "distance_cm": distance,
                    "battery_pct": battery,
                    "alert_type": alert,
                    "mode": mode,
                })
                elapsed_minutes += INTERVAL_SECONDS / 60

    return records


def main():
    records = generate_records()
    output_path = "sensor_data.csv"
    fieldnames = ["timestamp", "day_of_week", "hour", "distance_cm", "battery_pct",
                  "alert_type", "mode"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
