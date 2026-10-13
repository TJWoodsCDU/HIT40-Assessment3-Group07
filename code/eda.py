import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import config as C

COLOR = "#1B4332"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def vif_table(X):
    """VIF of each column = 1 / (1 - R^2) when that column is predicted from all the others."""
    rows = []
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        rows.append({"variable": col, "VIF": 1 / (1 - r2) if r2 < 1 else np.inf})
    return pd.DataFrame(rows)


def run(model, d, feats, response, label, out):
    model_name = model.replace("_", ".")          # "2_1" -> "2.1", used in titles
    X, y = d[list(feats)], d[response]

    # 1. summary statistics
    summary = pd.concat([X, y], axis=1).describe().T[["count", "mean", "std", "min", "50%", "max"]]
    summary.round(3).to_csv(out / f"eda_summary_{model}.csv")

    # 2. correlation of each clue with the result
    rows = []
    for f in X.columns:
        r, p = stats.pearsonr(X[f], y)
        rho, _ = stats.spearmanr(X[f], y)
        rows.append({"variable": f, "pearson_r": r, "p_value": p, "spearman_rho": rho})
    corr = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    corr.round(4).to_csv(out / f"eda_correlations_{model}.csv", index=False)

    # 3. VIF
    vif = vif_table(X)
    vif.round(2).to_csv(out / f"eda_vif_{model}.csv", index=False)

    # 4. scatterplots (2 columns x 4 rows), with a straight trend line
    fig, axes = plt.subplots(4, 2, figsize=(9, 12))
    for ax, f in zip(axes.ravel(), X.columns):
        jitter = np.random.default_rng(0).normal(0, 0.06, len(y)) if response == "goals_scored" else 0
        ax.scatter(X[f], y + jitter, s=14, alpha=0.6, color=COLOR)
        slope, intercept = np.polyfit(X[f], y, 1)
        xs = np.linspace(X[f].min(), X[f].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#D4AF37", linewidth=2)
        r = corr.loc[corr.variable == f, "pearson_r"].iloc[0]
        ax.set_title(f"{f}  (r = {r:.2f})", fontsize=10)
        ax.set_xlabel(f)
        ax.set_ylabel(label)
    fig.suptitle(f"Model {model_name}: each clue vs the result", y=0.995)
    fig.tight_layout()
    fig.savefig(out / f"fig_eda_scatter_{model}.png", dpi=150)
    plt.close(fig)

    # 5. heatmap of correlations between the clues
    cm = X.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.columns)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Model {model_name}: correlations between clues")
    fig.tight_layout()
    fig.savefig(out / f"fig_eda_heatmap_{model}.png", dpi=150)
    plt.close(fig)

    # 6. distribution of the result
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(y.min() - 0.5, y.max() + 1.5, 1)
    ax.hist(y, bins=bins, color=COLOR, edgecolor="white")
    ax.set_xlabel(label)
    ax.set_ylabel("Number of rows")
    ax.set_title(f"Model {model_name}: distribution of the result")
    fig.tight_layout()
    fig.savefig(out / f"fig_eda_response_{model}.png", dpi=150)
    plt.close(fig)

    print(f"\n--- Model {model_name} ---")
    print("Correlation with the result (strongest first):")
    print(corr.round(3).to_string(index=False))
    print("\nVIF (above 5 is worrying):")
    print(vif.round(2).to_string(index=False))


def main(mode):
    out = C.out_dir(mode)
    d21 = pd.read_csv(out / "dataset_2_1.csv")
    d22 = pd.read_csv(out / "dataset_2_2.csv")
    print(f"=== EDA ({mode} data) ===")
    run("2_1", d21, C.FEATURES_21, "goal_diff", "goal_diff", out)
    run("2_2", d22, C.FEATURES_22, "goals_scored", "goals_scored", out)
    print(f"\nSaved tables and figures to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dummy")