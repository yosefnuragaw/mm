import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_pf_sorted(data, filename=None):
    """
    Plot models sorted by Score, with red line connecting them,
    but **without** showing a legend.
    """
    # DataFrame
    df = pd.DataFrame(data)
    df_sorted = df.sort_values("Score", ascending=True).reset_index(drop=True)

    # Smooth fit through log‑cost
    log_cost = np.log(df_sorted["Cost"])
    coeffs = np.polyfit(log_cost, df_sorted["Score"], deg=2)
    x_vals = np.linspace(df_sorted["Cost"].min(), df_sorted["Cost"].max(), 300)
    y_vals = coeffs[0] * np.log(x_vals)**2 + coeffs[1] * np.log(x_vals) + coeffs[2]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set(style="white", font_scale=1.2)

    ax.plot(x_vals, y_vals, color="red", linewidth=1.5, zorder=1, alpha=0.5)

    ax.scatter(
        df_sorted["Cost"],
        df_sorted["Score"],
        s=120,
        edgecolors="white",
        linewidth=1.2,
        zorder=2,
        color="black"
    )

    # Labels
    for _, row in df_sorted.iterrows():
        ax.text(
            row.Cost,
            row.Score + 0.02,
            row.Model,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_xscale("log")

    # ⬇️  add these three lines  ⬇️
    cost_ticks = df_sorted["Cost"].tolist()          # put ticks exactly at your cost values
    ax.set_xticks(cost_ticks)                        # positions on the (log) axis
    ax.set_xticklabels([f"{c:g}" for c in cost_ticks])  # plain‑number labels

    ax.set_xlim(0.1, 50)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cost Input and Output ($/1 M Tokens)", fontsize=13)
    ax.set_ylabel("Arena Score", fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.25)

    # --- legend removed ---
    # ax.legend(loc="upper right")

    plt.tight_layout()

    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    plt.show()
    return fig, ax

# Example usage
# data_thompson = [
#     {"Model": "google/gemini-2.0-flash", "Cost":0.225 , "Score": 0.5555555555555556},
#     {"Model": "google/gemini-2.5-flash", "Cost":0.45 , "Score": 0.46153846153846156},
#     {"Model": "google/gemini-2.5-pro", "Cost":2.5 , "Score": 0.4782608695652174}
# ]
data_successive = [
    {"Model": "google/gemini-2.0-flash", "Cost":0.225 , "Score": 0.0625},
    {"Model": "google/gemini-2.5-flash", "Cost":0.45 , "Score": 0.21875},
    {"Model": "google/gemini-2.5-pro", "Cost":2.5 , "Score": 0.3125}
]
plot_pf_sorted(data_successive, filename="succ_sorted_line.png")
