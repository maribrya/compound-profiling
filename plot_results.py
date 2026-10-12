from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def plot_accuracy():
    summary = json.loads((ROOT / "bbbc021/classification_summary.json").read_text())
    names = ["original", "blurred", "restored"]
    values = [summary["metrics"][n]["accuracy"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(
        ["Original", "Blurred", "Restored"],
        values,
        color=["#243e5c", "#aaaaaa", "#668e9a"],
    )
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy, %")
    ax.set_title("103 profiles, whole-compound holdout")
    fig.tight_layout()
    return fig


def plot_transitions():
    row = json.loads((ROOT / "bbbc021/classification_summary.json").read_text())[
        "comparisons"
    ]["blurred_to_restored"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(
        ["Mistakes corrected", "New mistakes"],
        [row["corrected"], row["harmed"]],
        color=["#668e9a", "#b57359"],
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, 35)
    ax.set_ylabel("Treatment profiles")
    ax.set_title("Changes after restoration")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    plot_accuracy().savefig(folder / "accuracy.png", dpi=160)
    plot_transitions().savefig(folder / "transitions.png", dpi=160)
