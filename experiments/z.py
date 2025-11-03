import utils
from utils import *
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import cdist
import itertools


L_ARMS= [50,250,500,750]
# L_ROUNDS = [1696,8069,16096,24096]
L_ROUNDS = [1696,1696,1696,1696]
SAVEFILE = "z"
for armsize,round in zip(L_ARMS,L_ROUNDS):
    np.random.seed(0)
    random.seed(42)

    # initial_seeds = [
    #     0, 30, 50, 70, 90, 110, 130, 150,190,220,
    #     4, 34, 54, 74, 94, 114, 134, 154,194,224,
    #     9, 39, 59, 79, 99, 119, 139, 159,199, 229] 

    initial_seeds = [x for x in range(900)] 

    N_ARMS  = armsize
    GLOBAL_GOOD_PROB = 0.55
    LOCAL_GOOD_PROB = 0.6

    clusters = list(range(1,11)) 
    seed_to_cluster = {}
    for idx, seed in enumerate(initial_seeds):
        seed_to_cluster[seed] = idx % 10

    # print(seed_to_cluster)

    cluster_util = {}
    for c in clusters:
        size = random.randint(1, 5)               # random length
        util = random.sample(clusters, size)      # choose different cluster IDs
        util.sort()                                # sort in ascending order
        cluster_util[c-1] = util

    # print(cluster_util)

    SEEDS = np.random.permutation(initial_seeds)
    global_good_indices = np.random.choice(N_ARMS, size=10, replace=False)
    arm_costs = np.random.uniform(0.4, 2.0, size=N_ARMS)
    arm_costs[global_good_indices] = np.random.uniform(2.5, 3.0, size=len(global_good_indices))
    norm_costs = (arm_costs - arm_costs.min()) / (arm_costs.max() - arm_costs.min())
    N_COLS = 3
    N_ROWS = 100


    N_ROUNDS = round
    N_PRIOR = 0
    K = 0.05
    # LR = 0.2
    L = 0

    # --------- Plot Setup ---------
    N_PLOT = 30  # Last 30 iterations to plot
    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(40, 50), sharey=True)
    axes = axes.flatten()

    galpha = np.ones((N_ARMS,len(cluster_util)))
    gbeta = np.ones((N_ARMS,len(cluster_util)))
    # galpha = np.ones((N_ARMS, 11, 11))
    # gbeta = np.ones((N_ARMS, 11, 11))
    calls = 0
    igd = []
    cost_regrets = []
    all_avg_regret = []
    called_arms = [set() for _ in range(len(SEEDS))]
    model_labels = [f'arm_{i}' for i in range(N_ARMS)]
    f1s = []

    for idx, seed in enumerate(SEEDS):
        np.random.seed(seed)

        # --- Initialize true probabilities ---
        true_probs = np.linspace(0.1, 0.5, N_ARMS)
        half_from_global = np.random.choice(global_good_indices, size=4, replace=False)
        true_probs[half_from_global] = GLOBAL_GOOD_PROB + arm_costs[half_from_global] * 0.1

        other_indices = np.setdiff1d(np.arange(N_ARMS), global_good_indices)
        local_other = np.random.choice(other_indices, size=10, replace=False)
        local_good_indices = np.concatenate((half_from_global, local_other))
        true_probs[local_good_indices] = LOCAL_GOOD_PROB + arm_costs[local_good_indices] * 0.1

        # --- Compute cluster Cartesian pairs ---
        # util = cluster_util[seed_to_cluster[seed]] if cluster_util[seed_to_cluster[seed]] != [] else [0]
        # unique_pairs = list(itertools.product(util, repeat=2))

        # tri_alpha = np.tril(galpha, k=0)
        # tri_beta  = np.tril(gbeta,  k=0)

        # rows, cols = np.array(unique_pairs).T
        # lalpha = tri_alpha[:, rows, cols].sum(axis=1)
        # lbeta  = tri_beta[:, rows, cols].sum(axis=1)

        # n = lalpha + lbeta
        # myu = lalpha/n

        # n_new = (myu * (1 - myu)) / K - 1
        # n_new = np.maximum(n_new, 2)

        # alpha = myu * n_new
        # beta  = (1 - myu) * n_new

        util = seed_to_cluster[seed]
    
        lalpha = np.array(galpha[:,util], dtype=float)
        lbeta  = np.array(gbeta[:,util], dtype=float)

        n = lalpha + lbeta
        myu = lalpha/n

        n_new = (myu * (1 - myu)) / K - 1
        n_new = np.maximum(n_new, 2)

        alpha = myu * n_new
        beta  = (1 - myu) * n_new

        preva, prevb = alpha.copy(), beta.copy()
        optimal_prob = true_probs.max()

        # --- Storage for arms & successes ---
        arms = np.empty(N_ROUNDS, dtype=int)
        successes = np.empty(N_ROUNDS, dtype=int)

        # ----------- TS MAIN LOOP -----------
        for t in range(N_ROUNDS):
            theta = sample_ts_theta(alpha, beta)
            arm = np.argmax(theta)
            arms[t] = arm

            success = np.random.binomial(1, true_probs[arm])
            successes[t] = success

            alpha[arm] += success
            beta[arm]  += 1 - success

        calls += N_ROUNDS
        called_arms[idx].update(arms)

        # Compute regret & cost regret in bulk
        regrets = optimal_prob - true_probs[arms]
        cost_regrets.extend(arm_costs[arms] * regrets)
        all_avg_regret.append(np.mean(regrets))

        est_probs = alpha / (alpha + beta)
        est_data = pd.DataFrame({'Cost': arm_costs, 'Score': est_probs, 'Model': model_labels})
        true_data = pd.DataFrame({'Cost': arm_costs, 'Score': true_probs, 'Model': model_labels})

        fPareto = get_fPareto_function(compute_pareto(est_data, 'Cost', 'Score'))
        est_pf = compute_pareto(est_data, 'Cost', 'Score')
        true_pf = compute_pareto(true_data, 'Cost', 'Score')
        igd.append(compute_IGD(true_pf[['Cost', 'Score']].to_numpy(),
                            est_pf[['Cost', 'Score']].to_numpy()))

        est_models = set(est_pf["Model"])
        true_models = set(true_pf["Model"])
        tp = len(est_models & true_models)
        fp = len(est_models - true_models)
        fn = len(true_models - est_models)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)

        # --- Update global priors for meta-learning ---
        # oalp = alpha - preva
        # obet = beta - prevb
        # for x in range(N_ARMS):
        #     for y in unique_pairs:
        #         galpha[x][y] += oalp[x]
        #         gbeta[x][y] += obet[x]
        oalp = alpha - preva
        obet = beta - prevb
        
        galpha[:,util] += oalp
        gbeta[:,util] += obet

        # ----------- Plot last 30 seeds using plot_frontier_panel -----------
        if idx >= len(SEEDS) - N_PLOT:
            ax_idx = idx - (len(SEEDS) - N_PLOT)
            ax = axes[ax_idx]
            plot_frontier_panel(ax, est_data, true_data, jitter=False, seed=seed, lam=L)
            ax.set_title(f"Seed {seed}")
            ax.set_xlabel("Arm Cost")
            ax.set_ylabel("Estimated Success Probability")


    for j in range(N_PLOT, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{SAVEFILE}_{armsize}")

    # --- Summary stats ---
    avg_api_calls = calls // len(SEEDS)
    avg_arms_called = np.mean([len(x) for x in called_arms])
    avg_igd = np.mean(igd)
    avg_regret = np.mean(all_avg_regret)
    avg_costregret = np.mean(cost_regrets)

    print(F"{SAVEFILE} | {armsize}")
    print(f"Number of API calls per seed: {avg_api_calls}")
    print(f"Average distinct arms called per seed: {avg_arms_called:.2f}")
    print(f"Pareto quality (mean IGD): {avg_igd:.4f}")
    print(f"Average regret per pull: {avg_regret:.4f}")
    print(f"Average cost regret per pull: {avg_costregret:.4f}")
    print(f"Average F1 Score:  {np.mean(f1s):.4f}")

