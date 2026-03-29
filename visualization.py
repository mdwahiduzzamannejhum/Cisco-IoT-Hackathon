"""
Cisco IoT Hackathon — Assistive Technology for the Visually Impaired
Visualization Dashboard

Generates a multi-panel figure from IoT sensor data collected by the
wearable assistive device. The dashboard includes:

  Panel 1 – Obstacle Distance over Time (hourly average)
  Panel 2 – Alert Event Frequency by Type
  Panel 3 – Obstacle-Detection Heatmap (Hour × Day)
  Panel 4 – Device Mode Distribution
  Panel 5 – Battery Level over Time (daily sample)

Usage:
    python visualization.py [--data PATH] [--output PATH]
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sensor_data.csv")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "iot_dashboard.png")

ALERT_COLORS = {
    "obstacle_near": "#e74c3c",
    "obstacle_medium": "#f39c12",
    "fall_detected": "#8e44ad",
    "low_battery": "#3498db",
    "none": "#2ecc71",
}

MODE_COLORS = ["#2196F3", "#4CAF50", "#FF9800"]

DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load sensor CSV and parse timestamps."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["day_of_week"] = pd.Categorical(
        df["day_of_week"], categories=DAYS_ORDER, ordered=True
    )
    return df


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_distance_over_time(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Line chart: hourly mean obstacle distance for each day."""
    hourly = (
        df.groupby(["date", "hour"])["distance_cm"]
        .mean()
        .reset_index()
    )
    dates = sorted(hourly["date"].unique())
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(dates))

    for i, date in enumerate(dates):
        subset = hourly[hourly["date"] == date]
        ax.plot(
            subset["hour"],
            subset["distance_cm"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=cmap(i),
            label=str(date),
            alpha=0.85,
        )

    ax.axhline(80, color="#e74c3c", linestyle="--", linewidth=1, label="Near threshold (80 cm)")
    ax.axhline(30, color="#c0392b", linestyle=":", linewidth=1, label="Danger threshold (30 cm)")

    ax.set_title("Hourly Mean Obstacle Distance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Distance (cm)")
    ax.set_xticks(range(8, 20))
    ax.set_xlim(7.5, 19.5)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f9")


def plot_alert_frequency(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Horizontal bar chart: total alert counts by type (excluding 'none')."""
    alert_counts = (
        df[df["alert_type"] != "none"]["alert_type"]
        .value_counts()
        .sort_values(ascending=True)
    )
    colors = [ALERT_COLORS.get(a, "#95a5a6") for a in alert_counts.index]
    bars = ax.barh(alert_counts.index, alert_counts.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, alert_counts.values):
        ax.text(
            val + 5, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9
        )

    ax.set_title("Alert Event Frequency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("Alert Type")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f9")


def plot_obstacle_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Heatmap: count of obstacle alerts by hour and day of week."""
    obstacle_df = df[df["alert_type"].isin(["obstacle_near", "obstacle_medium"])]
    pivot = (
        obstacle_df.groupby(["day_of_week", "hour"])
        .size()
        .unstack(fill_value=0)
    )
    # Reindex to ensure all days present
    pivot = pivot.reindex(
        [d for d in DAYS_ORDER if d in pivot.index], axis=0
    )

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt="d",
        annot_kws={"size": 7},
        cbar_kws={"label": "Alert Count"},
    )
    ax.set_title("Obstacle Detection Heatmap\n(Hour × Day of Week)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_mode_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Pie chart: proportion of time in each device mode."""
    mode_counts = df["mode"].value_counts()
    wedge_props = {"edgecolor": "white", "linewidth": 1.5}
    ax.pie(
        mode_counts.values,
        labels=mode_counts.index,
        colors=MODE_COLORS[: len(mode_counts)],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=wedge_props,
        textprops={"fontsize": 9},
    )
    ax.set_title("Device Mode Distribution", fontsize=12, fontweight="bold")


def plot_battery_level(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Area chart: battery level sampled every 15 minutes across all days."""
    sampled = df.set_index("timestamp")["battery_pct"].resample("15min").mean().dropna()
    sampled = sampled.reset_index()

    ax.fill_between(
        sampled["timestamp"],
        sampled["battery_pct"],
        alpha=0.35,
        color="#3498db",
    )
    ax.plot(sampled["timestamp"], sampled["battery_pct"], color="#2980b9", linewidth=1.2)
    ax.axhline(20, color="#e74c3c", linestyle="--", linewidth=1, label="Low battery (20%)")

    ax.set_title("Battery Level over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Battery (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f9")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=8)


# ---------------------------------------------------------------------------
# Dashboard assembly
# ---------------------------------------------------------------------------

def build_dashboard(df: pd.DataFrame, output_path: str) -> None:
    """Compose all panels into a single dashboard figure and save."""
    sns.set_theme(style="whitegrid", font_scale=0.95)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Cisco IoT Hackathon — Assistive Technology Dashboard\n"
        "Real-time Sensor Monitoring for Visually Impaired Users",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Layout: 3 rows × 3 columns with merged cells
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    ax_distance = fig.add_subplot(gs[0, :])          # Row 0, all columns
    ax_alert    = fig.add_subplot(gs[1, :2])          # Row 1, cols 0-1
    ax_mode     = fig.add_subplot(gs[1, 2])           # Row 1, col 2
    ax_heatmap  = fig.add_subplot(gs[2, :2])          # Row 2, cols 0-1
    ax_battery  = fig.add_subplot(gs[2, 2])           # Row 2, col 2

    plot_distance_over_time(ax_distance, df)
    plot_alert_frequency(ax_alert, df)
    plot_mode_distribution(ax_mode, df)
    plot_obstacle_heatmap(ax_heatmap, df)
    plot_battery_level(ax_battery, df)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IoT Assistive Technology visualization dashboard."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to sensor_data.csv (default: data/sensor_data.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path (default: iot_dashboard.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading data from: {args.data}")
    df = load_data(args.data)
    print(f"Loaded {len(df):,} records spanning "
          f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    build_dashboard(df, args.output)


if __name__ == "__main__":
    main()
