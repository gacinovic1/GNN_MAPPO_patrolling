import argparse
import ast
import csv
import os
import matplotlib.pyplot as plt


def load_results(csv_path, step_duration):
    time, idleness, params = [], [], {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_step", "avg_idleness"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing CSV columns: {sorted(missing)}")
        for row in reader:
            time.append(float(row["time_step"]) * step_duration)
            idleness.append(float(row["avg_idleness"]) * step_duration)
            if not params:
                params = {
                    "observation_radius": row.get("observation_radius", ""),
                    "communication_probability": row.get("communication_probability", ""),
                    "communication_loss_probability": row.get("communication_loss_probability", ""),
                    "attrition_fixed_times": row.get("attrition_fixed_times", ""),
                }
    if not time:
        raise ValueError(f"No data rows found in {csv_path}")
    return time, idleness, params


def find_episode_csv(log_dir, episode):
    
    if os.path.isfile(log_dir):
        return log_dir
    
    expected = os.path.join(log_dir, f"idleness_episode_{episode}.csv")
    
    if os.path.isfile(expected):
        return expected
    
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")
    
    files = sorted(f for f in os.listdir(log_dir) if f.startswith("idleness_episode_") and f.endswith(".csv"))
    
    if len(files) == 1:
        return os.path.join(log_dir, files[0])
    
    if not files:
        raise FileNotFoundError(f"No idleness_episode_*.csv found in {log_dir}")
    
    raise FileNotFoundError(
        f"{expected} not found; multiple episode CSVs exist in {log_dir}. ""Specify --episode to select one.")


def communication_loss_percent(params):
    loss = params.get("communication_loss_probability", "")
    if loss not in ("", None):
        return 100.0 * float(loss)
    success = params.get("communication_probability", "")
    if success not in ("", None):
        # communication_probability is the probability of successful communication.
        return 100.0 * (float(success))
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare idleness curves for runs with different communication-loss probabilities.")
    parser.add_argument("log_dirs", nargs="+",help="Directories containing idleness_episode_N.csv, or paths to CSV files.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step-duration", type=float, default=1.0)
    parser.add_argument("--output", default=None,help="Optional single output path; otherwise save comparison_comm.png in every input directory.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    args = parser.parse_args()

    runs, attrition_times = [], set()
    for index, log_dir in enumerate(args.log_dirs):
        csv_path = find_episode_csv(log_dir, args.episode)
        time, idleness, params = load_results(csv_path, args.step_duration)
        loss_pct = communication_loss_percent(params)
        label = f"{loss_pct:g}% Comms" if loss_pct is not None else os.path.basename(os.path.normpath(log_dir))
        if not label:
            label = f"Run {index + 1}"
        runs.append({"path": log_dir, "time": time, "idleness": idleness,
                     "params": params, "label": label})
        raw_attrition = params.get("attrition_fixed_times", "")
        if raw_attrition:
            try:
                attrition_times.update(float(s) * args.step_duration
                                       for s in ast.literal_eval(raw_attrition))
            except (ValueError, SyntaxError, TypeError):
                print(f"Warning: cannot parse attrition_fixed_times={raw_attrition!r} in {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for run in runs:
        ax.plot(run["time"], run["idleness"], linewidth=1.2, label=run["label"])
    ax.set_xlabel("Time Elapsed (s)")
    ax.set_ylabel("Avg. Node Idleness (s)")

    if args.title:
        title = args.title
    else:
        radii = {r["params"].get("observation_radius", "") for r in runs
                 if r["params"].get("observation_radius", "") not in ("", None)}
        title = (f"Sim. Results - Obs. Range {next(iter(radii))} m (With Attrition)"
                 if len(radii) == 1 else
                 "Sim. Results - Different Communication Loss Probabilities")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)

    max_time = max(max(run["time"]) for run in runs)
    for t in sorted(attrition_times):
        if t <= max_time:
            ax.axvline(t, color="black", linestyle="--", linewidth=1)
            ax.annotate("attrition", xy=(t, 1), xycoords=("data", "axes fraction"),
                        xytext=(3, -3), textcoords="offset points", rotation=90,
                        ha="left", va="top", fontsize=8)
    ax.legend()
    fig.tight_layout()

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        output_dirs = []
        for run in runs:
            path = run["path"]
            out_dir = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
            if out_dir and out_dir not in output_dirs:
                output_dirs.append(out_dir)
        for out_dir in output_dirs:
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, "comparison_comm.png")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()