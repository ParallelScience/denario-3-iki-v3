# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_lca(data, n_classes, n_init=15, max_iter=1000, tol=1e-6):
    N, J = data.shape
    K = 5
    best_ll = -np.inf
    best_pi = None
    best_theta = None
    best_gamma = None
    valid_masks = [~np.isnan(data[:, j]) for j in range(J)]
    valid_vals = [data[valid_masks[j], j].astype(int) for j in range(J)]
    for init in range(n_init):
        np.random.seed(42 + init * 100 + n_classes)
        pi = np.random.dirichlet(np.ones(n_classes))
        theta = np.random.dirichlet(np.ones(K), size=(n_classes, J))
        ll_old = -np.inf
        for iteration in range(max_iter):
            log_gamma = np.zeros((N, n_classes))
            for c in range(n_classes):
                log_gamma[:, c] = np.log(pi[c] + 1e-15)
                for j in range(J):
                    log_gamma[valid_masks[j], c] += np.log(theta[c, j, valid_vals[j]] + 1e-15)
            max_log_gamma = np.max(log_gamma, axis=1, keepdims=True)
            gamma_unnorm = np.exp(log_gamma - max_log_gamma)
            sum_gamma = np.sum(gamma_unnorm, axis=1, keepdims=True)
            gamma = gamma_unnorm / sum_gamma
            ll = np.sum(max_log_gamma + np.log(sum_gamma))
            if ll - ll_old < tol:
                break
            ll_old = ll
            pi = np.mean(gamma, axis=0)
            for j in range(J):
                gamma_valid = gamma[valid_masks[j], :]
                for k in range(K):
                    mask = (valid_vals[j] == k)
                    theta[:, j, k] = np.sum(gamma_valid[mask, :], axis=0)
            sum_theta = np.sum(theta, axis=2, keepdims=True)
            theta = theta / (sum_theta + 1e-15)
        if ll > best_ll:
            best_ll = ll
            best_pi = pi
            best_theta = theta
            best_gamma = gamma
    P = (n_classes - 1) + n_classes * J * (K - 1)
    AIC = 2 * P - 2 * best_ll
    BIC = P * np.log(N) - 2 * best_ll
    mask = best_gamma > 1e-15
    entropy_val = -np.sum(best_gamma[mask] * np.log(best_gamma[mask]))
    entropy_norm = 1 - entropy_val / (N * np.log(n_classes)) if n_classes > 1 else 1.0
    return {'n_classes': n_classes, 'll': best_ll, 'AIC': AIC, 'BIC': BIC, 'Entropy': entropy_norm, 'pi': best_pi, 'theta': best_theta, 'gamma': best_gamma}

if __name__ == '__main__':
    raw_path = 'data/data_raw_encoded.csv'
    dummy_path = 'data/data_dummy_encoded.csv'
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_dummy = pd.read_csv(dummy_path, low_memory=False)
    ind_cols = ['Job_Security_Index_Current', 'Job_Security_Index_Future']
    data = df_raw[ind_cols].values + 2
    valid_mask = ~np.isnan(data).all(axis=1)
    data_valid = data[valid_mask]
    models = {}
    for c in range(2, 6):
        res = run_lca(data_valid, n_classes=c, n_init=15, max_iter=1000)
        models[c] = res
    best_c = min(models.keys(), key=lambda k: models[k]['BIC'])
    best_model = models[best_c]
    N_all = len(data)
    log_gamma_all = np.zeros((N_all, best_c))
    for c in range(best_c):
        log_gamma_all[:, c] = np.log(best_model['pi'][c] + 1e-15)
        for j in range(2):
            valid = ~np.isnan(data[:, j])
            vals = data[valid, j].astype(int)
            log_gamma_all[valid, c] += np.log(best_model['theta'][c, j, vals] + 1e-15)
    max_log_gamma_all = np.max(log_gamma_all, axis=1, keepdims=True)
    gamma_unnorm_all = np.exp(log_gamma_all - max_log_gamma_all)
    sum_gamma_all = np.sum(gamma_unnorm_all, axis=1, keepdims=True)
    gamma_all = gamma_unnorm_all / sum_gamma_all
    class_assignments = np.argmax(gamma_all, axis=1)
    e_curr_list = []
    e_fut_list = []
    for c in range(best_c):
        e_curr = np.sum(np.arange(-2, 3) * best_model['theta'][c, 0, :])
        e_fut = np.sum(np.arange(-2, 3) * best_model['theta'][c, 1, :])
        e_curr_list.append(e_curr)
        e_fut_list.append(e_fut)
    labels = [''] * best_c
    dist_to_zero = [abs(ec) + abs(ef) for ec, ef in zip(e_curr_list, e_fut_list)]
    neutral_idx = np.argmin(dist_to_zero)
    labels[neutral_idx] = 'Stagnant Neutral'
    sum_vals = [ec + ef for ec, ef in zip(e_curr_list, e_fut_list)]
    sum_vals_masked = [val if i != neutral_idx else -np.inf for i, val in enumerate(sum_vals)]
    opt_idx = np.argmax(sum_vals_masked)
    labels[opt_idx] = 'Resiliently Optimistic'
    sum_vals_masked_min = [val if i not in [neutral_idx, opt_idx] else np.inf for i, val in enumerate(sum_vals)]
    if best_c >= 3:
        dec_idx = np.argmin(sum_vals_masked_min)
        labels[dec_idx] = 'Anxiously Declining'
    for i in range(best_c):
        if labels[i] == '':
            ec, ef = e_curr_list[i], e_fut_list[i]
            if ec > 0 and ef < 0:
                labels[i] = 'Pessimistic Shift'
            elif ec < 0 and ef > 0:
                labels[i] = 'Optimistic Shift'
            elif ec + ef > 0:
                labels[i] = 'Moderately Optimistic'
            else:
                labels[i] = 'Moderately Pessimistic'
    for i in range(best_c):
        count = labels[:i].count(labels[i])
        if labels.count(labels[i]) > 1 and count > 0:
            labels[i] = labels[i] + ' ' + str(count + 1)
    df_raw['LCA_Class'] = class_assignments + 1
    df_raw['LCA_Class_Name'] = [labels[c] for c in class_assignments]
    df_dummy['LCA_Class'] = class_assignments + 1
    df_dummy['LCA_Class_Name'] = [labels[c] for c in class_assignments]
    df_raw.to_csv(raw_path, index=False)
    df_dummy.to_csv(dummy_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x_labels = ['Significantly\nnegative', 'Slightly\nnegative', 'No impact', 'Slightly\npositive', 'Significantly\npositive']
    x = np.arange(len(x_labels))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for c in range(best_c):
        axes[0].plot(x, best_model['theta'][c, 0, :], marker=markers[c], color=colors[c], label=labels[c], linewidth=2, markersize=8)
        axes[1].plot(x, best_model['theta'][c, 1, :], marker=markers[c], color=colors[c], label=labels[c], linewidth=2, markersize=8)
    axes[0].set_title('Current Job Security (QEA_2)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel('Conditional Probability')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_title('Expected Future Job Security (QEB_2)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    handles, plot_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc='lower center', ncol=best_c, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plot_filename = 'data/lca_profiles_1_' + str(int(time.time())) + '.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()