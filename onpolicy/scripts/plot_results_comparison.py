
import csv
import matplotlib.pyplot as plt
import os
import ast
import argparse


def load_results(csv_path, step_duration):
    
    time = []
    idleness = []
    params = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            time.append(float(row["time_step"]) * step_duration)
            idleness.append(float(row["avg_idleness"]) * step_duration)

            if not params:
                params = {"observation_radius": row.get("observation_radius", ""), "communication_probability": row.get("communication_probability", "" ),"attrition_fixed_times": row.get("attrition_fixed_times", "")}

    return time, idleness, params


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
    
    evaluated_params = []
    attrition_steps = set()

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

        time, idleness, params = load_results(csv_path, args.step_duration)

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
            
        evaluated_params.append((label, params))

        if params["attrition_fixed_times"]:
            steps = ast.literal_eval(params["attrition_fixed_times"])
            attrition_steps.update(step * args.step_duration for step in steps)

        plt.plot(time,idleness,color=color,linewidth=1.2,label=label)

    plt.xlabel("Time Elapsed (s)")
    plt.ylabel("Av. Node Idleness (s)")
    
    radius = ", ".join(f"{label}={params['observation_radius']}" for label, params in evaluated_params if params["observation_radius"])
    communication = ", ".join(f"{label}={100*float(params['communication_probability']):.0f}%" for label, params in evaluated_params if params["communication_probability"])

    plt.title("Average Node Idleness\n" f"Observation radius: {radius} | " f"Communication chance: {communication}")
    plt.grid(True, linestyle="--", alpha=0.25)
    
    ax = plt.gca()
    for t in sorted(attrition_steps):
        if t <= max(time):
            ax.axvline(t, color="black", linestyle="--", linewidth=1)
            ax.annotate("attrition", xy=(t, 1), xycoords=("data", "axes fraction"), xytext=(3, -3), textcoords="offset points", rotation=90, ha="left", va="top", fontsize=8)
        
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
