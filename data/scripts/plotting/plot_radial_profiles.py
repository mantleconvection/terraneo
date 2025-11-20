#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from typing import List
import itertools


def plot_radial_profiles(input_file: str, output_file: str, radius_column: str, value_columns: List[str],
                         x_axis_value: str, log_scale: bool):
    # Load file
    if input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_csv(input_file)

    if output_file is None:
        output_file = f"{os.path.basename(input_file)}.png"

    # Check if column with radius exists
    if radius_column not in df.columns:
        print(f"Column '{radius_column}' not found in input file")
        print(f"Available columns: {df.columns}")
        sys.exit(1)

    # Sort by radius just in case
    df = df.sort_values(radius_column)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_ylabel(radius_column)
    ax1.set_xlabel(x_axis_value)

    line_styles = [
        '-', '--', '-.', ':',
        (0, (1, 1)), (0, (1, 2)),
        (0, (5, 1)), (0, (5, 5)),
        (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
        (0, (1, 1, 3, 1)),
        'densely dashed', 'densely dotted', 'loosely dashed'
    ]

    style_cycle = itertools.cycle(line_styles)

    if not value_columns:
        value_columns = [col for col in df.columns if col != radius_column]

    for column in value_columns:
        if log_scale:
            ax1.semilogx(df[column], df[radius_column], label=column, color="black", linestyle=next(style_cycle))
        else:
            ax1.plot(df[column], df[radius_column], label=column, color="black", linestyle=next(style_cycle))

    ax1.legend(loc="upper right")

    if False:
        # Right y-axis for shell index
        ax2 = ax1.twinx()
        ax2.set_ylabel("Shell index", color="gray")
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks(df[radius_column])
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
        description="Plot radial profiles from CSV or JSONL."
    )
    parser.add_argument("input_file", help="Path to CSV or JSONL file")
    parser.add_argument("-r", "--radius-column", default="radius", help="Name of radius column, default: 'radius'.")
    parser.add_argument("--value-columns", nargs="+", default=[],
                        help="Names of value columns to plot. If not specified, all columns (except radius) are plotted.")
    parser.add_argument("--log-scale", action="store_true", help="Use log scale for x-axis")
    parser.add_argument(
        "-o", "--output", help="Output PNG file name, default: <input_file_basename>.png"
    )
    parser.add_argument(
        "-x",
        "--x-axis-label",
        default="value",
        help="x-axis label (default: 'value')",
    )
    args = parser.parse_args()

    plot_radial_profiles(args.input_file, args.output, args.radius_column, args.value_columns, args.x_axis_label,
                         args.log_scale)


if __name__ == "__main__":
    main()
