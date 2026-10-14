import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import config as C

COLOR, GOLD = "#1B4332", "#D4AF37"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def main(mode):
    out = C.out_dir(mode)
    m = pd.read_csv(C.data_dir(mode) / "matches.csv", parse_dates=["date"])
    # one row per team per match (the same shape as dataset_2_2)
    long = pd.concat([
        m.assign(team=m.team_a, goals=m.goals_a, host_team=(m.host_adv == 1)),
        m.assign(team=m.team_b, goals=m.goals_b, host_team=(m.host_adv == -1)),
    ], ignore_index=True)

    # ---------------------------------------------------------------- summary and baselines
    gd, g = m.goal_diff, long.goals
    mean_g, var_g = g.mean(), g.var(ddof=1)
    summary = pd.DataFrame([
        ("Matches (rows in Model 2.1)", len(m)),
        ("Goal difference: mean", gd.mean()),
        ("Goal difference: standard deviation", gd.std(ddof=1)),
        ("Goal difference: share of draws", (gd == 0).mean()),
        ("Goal difference: largest win margin", gd.abs().max()),
        ("Baseline RMSE, always predict 0 (Model 2.1)", np.sqrt((gd ** 2).mean())),
        ("Baseline RMSE, always predict the mean (Model 2.1)", gd.std(ddof=0)),
        ("Team rows (rows in Model 2.2)", len(long)),
        ("Goals per team: mean", mean_g),
        ("Goals per team: variance", var_g),
        ("Dispersion = variance / mean (1 = Poisson)", var_g / mean_g),
        ("Share of team rows with 0 goals", (g == 0).mean()),
        ("Baseline RMSE, always predict the mean (Model 2.2)", g.std(ddof=0)),
    ], columns=["measure", "value"])
    summary.round(3).to_csv(out / "response_summary.csv", index=False)

    # ---------------------------------------------------------------- group comparisons (Welch t-tests)
    rows = []

    def compare(label, a, b, name_a, name_b):
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"comparison": label, name_a: f"{a.mean():.2f} (n={len(a)})",
                     name_b: f"{b.mean():.2f} (n={len(b)})", "difference": round(a.mean() - b.mean(), 2),
                     "p_value": round(p, 3)})

    ko = long.knockout == 1
    compare("Goals per team: knockout vs group stage", g[ko], g[~ko], "knockout", "group")
    md3 = long.final_group_md == 1
    compare("Goals per team: third group match vs other matches", g[md3], g[~md3], "third group match", "other")
    host_goal_diff = (m.goal_diff * m.host_adv)[m.host_adv != 0]    # positive = the host did better
    t, p = stats.ttest_1samp(host_goal_diff, 0)
    rows.append({"comparison": "Host's goal difference when playing at home (test: is it above 0?)",
                 "knockout": "", "group": "", "difference": round(host_goal_diff.mean(), 2),
                 "p_value": round(p, 3), "note": f"n={len(host_goal_diff)} matches only"})
    t, p = stats.ttest_1samp(m.goal_diff, 0)
    rows.append({"comparison": "Is goal difference centred on 0? (Team A / Team B order check)",
                 "difference": round(m.goal_diff.mean(), 2), "p_value": round(p, 3),
                 "note": "should not be significant, otherwise the A/B order matters"})
    pd.DataFrame(rows).to_csv(out / "group_comparisons.csv", index=False)

    # ---------------------------------------------------------------- figure: goals vs Poisson
    top = 6
    observed = np.array([(g == k).sum() for k in range(top)] + [(g >= top).sum()])
    expected = np.array([stats.poisson.pmf(k, mean_g) for k in range(top)] + [stats.poisson.sf(top - 1, mean_g)]) * len(g)
    labels = [str(k) for k in range(top)] + [f"{top}+"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(labels, observed, color=COLOR, label="Observed")
    ax.plot(labels, expected, "o-", color=GOLD, linewidth=2, label=f"Poisson with mean {mean_g:.2f}")
    ax.set(xlabel="Goals scored by a team in one match", ylabel="Number of team rows",
           title="Goals per team vs a Poisson distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_goals_distribution.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- figure: goals by stage
    order = ["Group stage", "Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Third-place match", "Final"]
    by_stage = long.groupby("stage").goals.agg(["mean", "count"]).reindex([s for s in order if s in set(long.stage)])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_stage.index, by_stage["mean"], color=COLOR)
    for i, (mu, n) in enumerate(zip(by_stage["mean"], by_stage["count"])):
        ax.text(i, mu + 0.03, f"{mu:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set(ylabel="Mean goals per team", title="Goals per team by stage", ylim=(0, by_stage["mean"].max() + 0.5))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out / "fig_goals_by_stage.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print(f"=== Response EDA ({mode} data) ===")
    print(summary.round(3).to_string(index=False))
    print("\n" + pd.DataFrame(rows).fillna("").to_string(index=False))
    print("\nGoals per team: observed vs Poisson expected")
    print(pd.DataFrame({"goals": labels, "observed": observed, "poisson_expected": expected.round(1)}).to_string(index=False))
    print(f"\nSaved tables and figures to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "real")