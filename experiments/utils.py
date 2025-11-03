import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import cdist

def compute_IGD(true_front, est_front):
    distances = cdist(true_front, est_front)  # shape: [len(true), len(estimated)]
    min_distances = np.min(distances, axis=1)  # for each true point, closest estimated
    return np.mean(min_distances)

def get_fPareto_function(pf_df):
    """Return interpolator function fPareto(c) = max score at cost c.
    
    Handles:
    - Constant scores
    - Single value
    - Extrapolation bounded by min/max
    - Log-cost safety
    """

    cost = pf_df['Cost'].values
    score = pf_df['Score'].values

    if len(cost) == 1:
        const_score = float(score[0])
        return lambda c: const_score

    sorted_idx = np.argsort(cost)
    cost = cost[sorted_idx]
    score = score[sorted_idx]

    if np.any(cost <= 0):
        raise ValueError("Cost must be positive")

    logc = np.log(cost)

    if np.all(score == score[0]):
        const_score = float(score[0])
        return lambda c: const_score

    pchip = PchipInterpolator(logc, score, extrapolate=True)

    min_score, max_score = np.min(score), np.max(score)
    return lambda c: float(np.clip(pchip(np.log(c)), min_score, max_score))

def sample_ts_theta(alpha, beta):
    X = np.random.gamma(alpha, 1.0)
    Y = np.random.gamma(beta,  1.0)
    return X / (X + Y)

def compute_pareto(df, cost_col, score_col):
    data = df.sort_values(by=[cost_col, score_col], ascending=[True, False])
    pareto = []
    max_score = -np.inf
    for _, row in data.iterrows():
        if row[score_col] > max_score:
            pareto.append(row)
            max_score = row[score_col]
    return pd.DataFrame(pareto)

def plot_frontier_panel(ax, est_data, true_data, jitter=False, seed=None, lam=None):
    est_df = est_data.copy()
    true_df = true_data.copy()

    if jitter:
        np.random.seed(seed)
        est_df['Score'] += np.random.normal(0, 0.05, size=len(est_df))

    est_pf = compute_pareto(est_df, 'Cost', 'Score')
    true_pf = compute_pareto(true_df, 'Cost', 'Score')

    true_pf_models = set(true_pf['Model'])
    est_df['Color'] = est_df['Model'].apply(lambda m: 'blue' if m in true_pf_models else 'gray')

    sns.scatterplot(
        data=est_df, x='Cost', y='Score', hue='Color',
        palette={'gray': 'gray', 'blue': 'blue'},
        s=60, alpha=0.5, ax=ax, legend=False, zorder=1
    )

    if len(est_pf) == 1:
        x = est_pf['Cost'].values[0]
        y = est_pf['Score'].values[0]
        ax.scatter(x, y, color='green', s=100, edgecolor='black',
                   linewidth=1.2, zorder=3, label='Estimated PF (single point)')
    else:
        logc = np.log(est_pf['Cost'].values)
        sc = est_pf['Score'].values
        pchip = PchipInterpolator(logc, sc)
        logc_new = np.linspace(logc.min(), logc.max(), 300)
        cost_new = np.exp(logc_new)
        score_new = pchip(logc_new)

        ax.fill_between(cost_new, lam * score_new, score_new,
                        color='green', alpha=0.15, label=f'Estimated PF Zone')
        ax.plot(cost_new, score_new, color='green', linewidth=2,
                label="Estimated PF", zorder=2)

        sns.scatterplot(x=est_pf['Cost'], y=est_pf['Score'],
                        color='green', s=100, edgecolor='black',
                        linewidth=1.2, ax=ax, legend=False, zorder=3)

    if len(true_pf) == 1:
        x = true_pf['Cost'].values[0]
        y = true_pf['Score'].values[0]
        ax.scatter(x, y, color='blue', s=80, edgecolor='black',
                   linewidth=1.2, zorder=4, label='True PF (single point)')
    else:
        logc_true = np.log(true_pf['Cost'].values)
        sc_true = true_pf['Score'].values
        pchip_true = PchipInterpolator(logc_true, sc_true)
        logc_new_true = np.linspace(logc_true.min(), logc_true.max(), 300)
        cost_new_true = np.exp(logc_new_true)
        score_new_true = pchip_true(logc_new_true)

        ax.plot(cost_new_true, score_new_true, 'b--', linewidth=2,
                label="True PF", zorder=4)

        sns.scatterplot(x=true_pf['Cost'], y=true_pf['Score'],
                        color='blue', s=80, edgecolor='black',
                        linewidth=1.2, ax=ax, legend=False, zorder=4)

    for model in true_pf_models:
        est_row = est_df[est_df['Model'] == model]
        true_row = true_df[true_df['Model'] == model]
        if not est_row.empty and not true_row.empty:
            ax.plot(
                [est_row['Cost'].values[0], true_row['Cost'].values[0]],
                [est_row['Score'].values[0], true_row['Score'].values[0]],
                color='blue', linewidth=1.0, linestyle='--', alpha=0.8
            )

    ax.set_xscale('log')
    ax.set_xlabel("Cost")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xlim(est_df['Cost'].min() * 0.9, est_df['Cost'].max() * 1.1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.3)
    ax.legend()