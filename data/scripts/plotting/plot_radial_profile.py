#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def plot_radial_profiles(input_file, x_axis_value, output_file):
    # Load file
    if input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_csv(input_file)

    if output_file is None:
        output_file = f"{os.path.basename(input_file)}.png"

    # Sort by radius just in case
    df = df.sort_values("radius")

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_ylabel("Radius")
    ax1.set_xlabel(x_axis_value)
    ax1.plot(df["min"], df["radius"], label="Min", color="black", linestyle="--")
    ax1.plot(df["avg"], df["radius"], label="Avg", color="black")
    ax1.plot(df["max"], df["radius"], label="Max", color="black", linestyle=":")
    ax1.legend(loc="upper right")

    # Right y-axis for shell index
    ax2 = ax1.twinx()
    ax2.set_ylabel("Shell index", color="gray")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(df["radius"])
    ax2.set_yticklabels(df["shell_idx"])
    ax2.tick_params(axis="y", labelcolor="gray")
    ax1.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

    ax1.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)

    plt.title(f"Radial Profiles ({input_file})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot radial profiles (min, avg, max) from CSV or JSONL."
    )
    parser.add_argument("input_file", help="Path to CSV or JSONL file")
    parser.add_argument(
        "-o", "--output", help="Output PNG file name, default: <input_file_basename>.png"
    )
    parser.add_argument(
        "-x",
        "--x-axis-value",
        default="value",
        help="What is plotted on the x-axis (default: 'value')",
    )
    args = parser.parse_args()

    plot_radial_profiles(args.input_file, args.x_axis_value, args.output)


if __name__ == "__main__":
    main()
