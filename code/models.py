"""
models.py - fits the linear regression for each model, compares it with other scikit-learn
regressors, and saves tables and figures (in outputs/<mode>/).

How the evaluation works
------------------------
Each model predicts matches it has NOT seen. We use repeated 5-fold cross-validation: the matches
are split into 5 parts, the model is trained on 4 parts and predicts the 5th, and this is repeated
until every match has been predicted, then redone 10 times with different splits.
The predictions are compared with the real World Cup 2026 results (R2, RMSE, MAE).
In Model 2.2 the two rows of one match are always kept in the same part, otherwise the model could
"peek" at the other side of the match it is predicting.

Files saved for each model <m> (2_1 or 2_2):
    model_comparison_<m>.csv      one row per algorithm, sorted by RMSE (lower is better)
    fig_model_comparison_<m>.png  the same as a chart
    ols_coefficients_<m>.csv      the linear regression: coefficients, p-values, confidence intervals
    ols_summary_<m>.txt           full statsmodels summary + the most influential matches
    fig_diagnostics_<m>.png       residuals vs fitted, Q-Q plot, residual histogram
    fig_pred_vs_actual_<m>.png    predicted vs real result (linear regression and the best algorithm)
    predictions_<m>.csv           out-of-sample prediction for every row

Run:  python models.py 
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, PoissonRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config as C

warnings.filterwarnings("ignore")
COLOR, GOLD = "#1B4332", "#D4AF37"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
K_FOLDS, REPEATS, SEED = 5, 10, 42


# ------------------------------------------------------------------ cross-validation
def make_splits(groups, k=K_FOLDS, repeats=REPEATS, seed=SEED):
    """Repeated k-fold where every row with the same group value (same match) stays together."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    all_repeats = []
    for _ in range(repeats):
        order = rng.permutation(unique)
        fold_of = {g: i % k for i, g in enumerate(order)}
        fold = np.array([fold_of[g] for g in groups])
        all_repeats.append([(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)])
    return all_repeats


def evaluate(model, X, y, splits):
    """Returns metrics (mean and sd across repeats) and the average out-of-sample prediction per row."""
    preds = np.zeros((len(splits), len(y)))
    for r, folds in enumerate(splits):
        for train, test in folds:
            fitted = model.fit(X.iloc[train], y.iloc[train])
            preds[r, test] = fitted.predict(X.iloc[test])
    r2 = [r2_score(y, p) for p in preds]
    rmse = [np.sqrt(mean_squared_error(y, p)) for p in preds]
    mae = [mean_absolute_error(y, p) for p in preds]
    metrics = {"R2": np.mean(r2), "R2_sd": np.std(r2), "RMSE": np.mean(rmse), "RMSE_sd": np.std(rmse),
               "MAE": np.mean(mae)}
    return metrics, preds.mean(axis=0)


def get_models(count_data):
    models = {
        "Baseline (always the mean)": DummyRegressor(strategy="mean"),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge regression": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso regression": make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=20000)),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                               random_state=0, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.03,
                                                       max_depth=2, subsample=0.8, random_state=0),
    }
    if count_data:   # goals are counts, so a Poisson regression is a natural extra option
        models["Poisson regression"] = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=1000))
    return models


# ------------------------------------------------------------------ one model
def run(model_key, d, feats, response, label, groups, out, count_data):
    model_name = model_key.replace("_", ".")          # "2_1" -> "2.1", used in titles
    X, y = d[list(feats)], d[response]
    splits = make_splits(groups)

    # --- compare algorithms
    results, oof = [], {}
    for name, m in get_models(count_data).items():
        metrics, oof[name] = evaluate(m, X, y, splits)
        results.append({"algorithm": name, **metrics})
    comp = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    comp.round(4).to_csv(out / f"model_comparison_{model_key}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(comp.algorithm[::-1], comp.RMSE[::-1], xerr=comp.RMSE_sd[::-1], color=COLOR, capsize=3)
    ax.set_xlabel("Cross-validated RMSE (lower is better)")
    ax.set_title(f"Model {model_name}: algorithm comparison")
    fig.tight_layout()
    fig.savefig(out / f"fig_model_comparison_{model_key}.png", dpi=150)
    plt.close(fig)

    # --- the linear regression in detail (full data, for interpretation)
    Xc = sm.add_constant(X)
    if model_key == "2_2":   # two rows per match are related, so use match-clustered standard errors
        ols = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    else:
        ols = sm.OLS(y, Xc).fit()
    z = (X - X.mean()) / X.std(ddof=0)
    std_coef = sm.OLS(y, sm.add_constant(z)).fit().params          # effect of +1 standard deviation
    ci = ols.conf_int()
    coefs = pd.DataFrame({
        "variable": ols.params.index, "coefficient": ols.params.values, "std_error": ols.bse.values,
        "p_value": ols.pvalues.values, "ci_low": ci[0].values, "ci_high": ci[1].values,
        "standardised_coef": std_coef.reindex(ols.params.index).values})
    coefs.round(4).to_csv(out / f"ols_coefficients_{model_key}.csv", index=False)

    infl = ols.get_influence().cooks_distance[0]
    top = pd.DataFrame({"row": np.argsort(infl)[::-1][:5], "cooks_distance": np.sort(infl)[::-1][:5]})
    with open(out / f"ols_summary_{model_key}.txt", "w") as f:
        f.write(str(ols.summary()))
        f.write(f"\n\nMost influential rows (Cook's distance; rule of thumb: above {4 / len(y):.3f}):\n")
        f.write(top.to_string(index=False))

    # --- diagnostics for the linear regression
    fitted, resid = ols.fittedvalues, ols.resid
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(fitted, resid, s=14, alpha=0.6, color=COLOR)
    axes[0].axhline(0, color=GOLD, linewidth=2)
    axes[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
    stats.probplot(resid, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set(markerfacecolor=COLOR, markeredgecolor=COLOR, markersize=4)
    axes[1].get_lines()[1].set_color(GOLD)
    axes[1].set_title("Q-Q plot of residuals")
    axes[2].hist(resid, bins=20, color=COLOR, edgecolor="white")
    axes[2].set(xlabel="Residual", ylabel="Rows", title="Residual histogram")
    fig.suptitle(f"Model {model_name}: linear regression diagnostics")
    fig.tight_layout()
    fig.savefig(out / f"fig_diagnostics_{model_key}.png", dpi=150)
    plt.close(fig)

    # --- predicted vs actual (out-of-sample) for linear regression and the best algorithm
    best = comp[comp.algorithm != "Baseline (always the mean)"].iloc[0].algorithm
    show = ["Linear regression"] + ([best] if best != "Linear regression" else [])
    fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 4.6), squeeze=False)
    for ax, name in zip(axes[0], show):
        jitter = np.random.default_rng(1).normal(0, 0.06, len(y)) if count_data else 0
        ax.scatter(oof[name], y + jitter, s=14, alpha=0.6, color=COLOR)
        lims = [min(oof[name].min(), y.min()), max(oof[name].max(), y.max())]
        ax.plot(lims, lims, color=GOLD, linewidth=2)
        row = comp[comp.algorithm == name].iloc[0]
        ax.set(xlabel="Predicted (on matches the model had not seen)", ylabel="Actual result",
               title=f"{name}\nR2 = {row.R2:.2f}, RMSE = {row.RMSE:.2f}")
    fig.tight_layout()
    fig.savefig(out / f"fig_pred_vs_actual_{model_key}.png", dpi=150)
    plt.close(fig)

    preds = d[[c for c in d.columns if c in ("match_id", "team_a", "team_b", "team", "opponent")]].copy()
    preds["actual"] = y
    for name in show:
        preds[f"predicted_{name.lower().replace(' ', '_')}"] = oof[name]
    preds.round(3).to_csv(out / f"predictions_{model_key}.csv", index=False)

    # --- console summary
    print(f"\n=== Model {model_name} ({len(y)} rows, {len(feats)} variables) ===")
    print(comp[["algorithm", "R2", "R2_sd", "RMSE", "RMSE_sd", "MAE"]].round(3).to_string(index=False))
    print(f"\nLinear regression on all rows: R2 = {ols.rsquared:.3f}, adjusted R2 = {ols.rsquared_adj:.3f}")
    print(coefs[["variable", "coefficient", "p_value", "standardised_coef"]].round(3).to_string(index=False))
    lin = comp[comp.algorithm == "Linear regression"].iloc[0]
    top_algo = comp[comp.algorithm != "Baseline (always the mean)"].iloc[0]
    gap = lin.RMSE - top_algo.RMSE
    print(f"\nBest algorithm by RMSE: {top_algo.algorithm}. Linear regression is {gap:.3f} RMSE behind "
          f"(the repeat-to-repeat spread is about {lin.RMSE_sd:.3f}).")


def main(mode):
    out = C.out_dir(mode)
    d21 = pd.read_csv(out / "dataset_2_1.csv")
    d22 = pd.read_csv(out / "dataset_2_2.csv")
    print(f"=== Models ({mode} data) ===")
    run("2_1", d21, C.FEATURES_21, "goal_diff", "goal_diff", d21.match_id, out, count_data=False)
    run("2_2", d22, C.FEATURES_22, "goals_scored", "goals_scored", d22.match_id, out, count_data=True)
    print(f"\nSaved tables and figures to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dummy")