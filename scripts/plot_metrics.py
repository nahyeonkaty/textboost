#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc("font", size=14)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        type=str,
        nargs="+",
        help="Experiment directories",
    )
    parser.add_argument(
        "-x",
        "--xaxis",
        type=str,
        default="vqa",
        help="X-axis metric",
    )
    parser.add_argument(
        "-y",
        "--yaxis",
        type=str,
        default="dino_unseen",
        help="Y-axis metric",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="X-axis label",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Y-axis label",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each experiment",
    )
    args = parser.parse_args()
    if isinstance(args.exp, list):
        for i in range(len(args.exp)):
            if args.exp[i].endswith("/"):
                args.exp[i] = args.exp[i][:-1]

    return args


def main():
    args = parse_arguments()

    plt.figure()
    if args.labels is None:
        args.labels = [None] * len(args.exp)

    for file, label in zip(args.exp, args.labels):
        if label is None:
            label = os.path.basename(os.path.dirname(file))
        print(label)
        x_values = {}
        y_values = {}

        with open(file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            i_idx = header.index("checkpoint")
            x_idx = header.index(args.xaxis)
            y_idx = header.index(args.yaxis)
            for data in list(reader):
                iteration = int(data[i_idx])
                if iteration in x_values:
                    x_values[iteration].append(float(data[x_idx]))
                    y_values[iteration].append(float(data[y_idx]))
                else:
                    x_values[iteration] = [float(data[x_idx])]
                    y_values[iteration] = [float(data[y_idx])]

        x_values = {k: np.mean(v) for k, v in x_values.items()}
        y_values = {k: np.mean(v) for k, v in y_values.items()}

        if "tb" in label and False:
            marker = "*"
            markersize = 15
        else:
            marker = "o"
            markersize = 8

        for x, y in zip(x_values.items(), y_values.items()):
            print(f"{x[0]}: {x[1]:.3f}, {y[1]:.3f}")
            # print(x, y)
        plt.plot(
            x_values.values(),
            y_values.values(),
            marker=marker,
            markersize=markersize,
            linewidth=2,
            label=label,
        )
        # Add iteration numbers
        for i, (x, y) in enumerate(zip(x_values.items(), y_values.items())):
            text = str(list(x_values.keys())[i])
            if text in ("16", "80"):
                plt.text(
                    x[1] - 0.004,
                    y[1] - 0.012,
                    text,
                    fontsize=12,
                    ha="center",
                    va="bottom",
                )

    plt.xlabel(args.xaxis if args.xlabel is None else args.xlabel)
    plt.ylabel(args.yaxis if args.ylabel is None else args.ylabel)
    # plt.yticks(np.arange(0.4, 0.61, 0.05))
    # plt.ylim(0.4, 0.6)
    # plt.xlim(0.4, 0.9)
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics.svg")
    plt.savefig("metrics.pdf")


if __name__ == "__main__":
    main()
