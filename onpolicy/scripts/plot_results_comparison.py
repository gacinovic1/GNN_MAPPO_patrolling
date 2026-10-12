
import csv
import matplotlib.pyplot as plt
import os
import argparse


def load_results(csv_path, step_duration):
    time = []
    idleness = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            time.append(float(row["time_step"]) * step_duration)
            idleness.append(float(row["avg_idleness"]) * step_duration)

    return time, idleness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dirs", nargs="+")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step-duration", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if len(args.log_dirs) > 2:
        raise ValueError("Too many arguments.")

    plt.figure(figsize=(10, 6))

    for i, log_dir in enumerate(args.log_dirs):
        if os.path.isfile(log_dir):
            csv_path = log_dir
        else:
            csv_path = os.path.join(log_dir, f"idleness_episode_{args.episode}.csv")

            if not os.path.isfile(csv_path):
                
                csv_files = sorted(f for f in os.listdir(log_dir) if f.startswith("idleness_episode_") and f.endswith(".csv"))

                if len(csv_files) == 1:
                    csv_path = os.path.join(log_dir, csv_files[0])
                else:
                    raise FileNotFoundError(f"CSV not found in {log_dir}")

        time, idleness = load_results(csv_path, args.step_duration)

        if len(args.log_dirs) == 1:
            original = "original" in log_dir.lower()
        else:
            original = i == 1

        if original:
            label = "Original GNN"
            color = "red"
        else:
            label = "MyGNN"
            color = "blue"

        plt.plot(time,idleness,color=color,linewidth=1.2,label=label)

    plt.xlabel("Time Elapsed (s)")
    plt.ylabel("Av. Node Idleness (s)")
    plt.title("Average Node Idleness During Evaluation")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if args.output:
        plot_path = args.output
        output_dir = os.path.dirname(plot_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    else:
        for i, output_dir in enumerate(args.log_dirs):
            plot_path = os.path.join(output_dir,"average_idleness_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            
    plt.show()

if __name__ == "__main__":
    main()
