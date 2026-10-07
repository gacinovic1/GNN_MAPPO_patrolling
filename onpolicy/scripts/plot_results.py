import json
import matplotlib.pyplot as plt
import os
import argparse

def load_results(summary_path):
    with open(summary_path, "r") as f:
        data = json.load(f)
    reward_key = next(key for key in data if "average_episode_rewards" in key)
    idleness_key = next(key for key in data if "avg_idleness" in key)
    reward_data = data[reward_key]
    idleness_data = data[idleness_key]
    environment_steps = [x[1] for x in reward_data]
    average_reward = [x[2] for x in reward_data]
    average_idleness = [x[2] for x in idleness_data]
    return environment_steps, average_reward, average_idleness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    args = parser.parse_args()

    my_gnn_log = os.path.join(args.log_dir, "summary.json")
    steps_my, reward_my, idleness_my = load_results(my_gnn_log)

    plt.figure(figsize=(10, 6))
    plt.plot(steps_my, idleness_my, color="blue", marker="o", markersize=3, linewidth=1.2, label="Average Idleness")
    plt.plot(steps_my, reward_my, color="orange", marker="o", markersize=3, linewidth=1.2, label="Average Reward")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Value")
    plt.title("Average Rewards vs Average Idleness")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_dir = os.path.join(args.log_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, "rewards_vs_idleness.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
