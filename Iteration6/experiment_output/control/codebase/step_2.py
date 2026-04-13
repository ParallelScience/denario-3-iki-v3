# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import time

def run_lca(data, n_classes, n_init=20, max_iter=1000, tol=1e-7):
    N, M = data.shape
    C_max = int(np.max(data) + 1)
    one_hots = [np.eye(C_max)[data[:, m]] for m in range(M)]
    best_ll = -np.inf
    best_model = None
    for init in range(n_init):
        pi = np.random.dirichlet(np.ones(n_classes))
        theta = np.random.dirichlet(np.ones(C_max), size=(n_classes, M))
        ll_old = -np.inf
        for iteration in range(max_iter):
            log_prob_y_given_c = np.zeros((N, n_classes))
            for m in range(M):
                log_prob_y_given_c += np.log(theta[:, m, data[:, m]].T + 1e-15)
            log_joint = np.log(pi + 1e-15) + log_prob_y_given_c
            log_marginal = logsumexp(log_joint, axis=1)
            ll = np.sum(log_marginal)
            gamma = np.exp(log_joint - log_marginal[:, np.newaxis])
            N_c = np.sum(gamma, axis=0)
            pi = N_c / N
            for m in range(M):
                counts = (one_hots[m].T @ gamma).T
                theta[:, m, :] = (counts + 1e-6) / (N_c[:, np.newaxis] + C_max * 1e-6)
            if np.abs(ll - ll_old) < tol:
                break
            ll_old = ll
        if ll > best_ll:
            best_ll = ll
            best_model = {'pi': pi, 'theta': theta, 'gamma': gamma, 'll': ll, 'n_iter': iteration}
    return best_model

if __name__ == '__main__':
    np.random.seed(42)
    data_path = os.path.join('data', 'cleaned_dataset.csv')
    df = pd.read_csv(data_path, low_memory=False)
    qea2_col = next((col for col in df.columns if col.startswith('QEA_2')), None)
    qeb2_col = next((col for col in df.columns if col.startswith('QEB_2')), None)
    if not qea2_col or not qeb2_col:
        raise ValueError('QEA_2 or QEB_2 not found in the dataset.')
    mapping = {-2.0: 0, -1.0: 1, 0.0: 2, 1.0: 3, 2.0: 4}
    y1 = df[qea2_col].map(mapping).fillna(5).astype(int).values
    y2 = df[qeb2_col].map(mapping).fillna(5).astype(int).values
    data = np.column_stack((y1, y2))
    N, M = data.shape
    C_max = 6
    bics = []
    entropies = []
    norm_entropies = []
    models = {}
    print('Running Latent Class Analysis for 2 to 6 classes...')
    for k in range(2, 7):
        model = run_lca(data, k, n_init=20, max_iter=1000)
        models[k] = model
        n_params = (k - 1) + k * M * (C_max - 1)
        bic = n_params * np.log(N) - 2 * model['ll']
        bics.append(bic)
        gamma = model['gamma']
        entropy = -np.sum(gamma * np.log(gamma + 1e-15))
        norm_entropy = 1 - entropy / (N * np.log(k))
        entropies.append(entropy)
        norm_entropies.append(norm_entropy)
        print('Class ' + str(k) + ': Log-Likelihood = ' + str(round(model['ll'], 2)) + ', BIC = ' + str(round(bic, 2)) + ', Norm Entropy = ' + str(round(norm_entropy, 4)))
    stats_df = pd.DataFrame({'Classes': range(2, 7), 'BIC': bics, 'Entropy': entropies, 'Normalized_Entropy': norm_entropies})
    stats_df.to_csv(os.path.join('data', 'lca_model_statistics.csv'), index=False)
    plt.rcParams['text.usetex'] = False
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Number of Classes')
    ax1.set_ylabel('BIC', color=color)
    ax1.plot(range(2, 7), bics, marker='o', color=color, label='BIC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Normalized Entropy', color=color)
    ax2.plot(range(2, 7), norm_entropies, marker='s', color=color, label='Normalized Entropy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    plt.title('LCA Model Selection: BIC and Normalized Entropy by Number of Classes')
    fig.tight_layout()
    plot_filename = os.path.join('data', 'lca_bic_entropy_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)
    plt.close()
    best_k = range(2, 7)[np.argmin(bics)]
    best_model = models[best_k]
    pos_prob = np.sum(best_model['theta'][:, :, 3:5], axis=(1, 2))
    sort_idx = np.argsort(-pos_prob)
    best_model['pi'] = best_model['pi'][sort_idx]
    best_model['theta'] = best_model['theta'][sort_idx]
    best_model['gamma'] = best_model['gamma'][:, sort_idx]
    cat_names = ['Significantly negative', 'Slightly negative', 'No impact', 'Slightly positive', 'Significantly positive', 'Not sure']
    var_names = ['Current Impact (QEA_2)', 'Expected Impact (QEB_2)']
    print('\nChosen class solution: ' + str(best_k) + ' classes (based on minimum BIC)')
    print('------------------------------------------------------------')
    for c in range(best_k):
        print('Class ' + str(c + 1) + ': ' + str(round(best_model['pi'][c], 4)))
    for c in range(best_k):
        print('\n--- Class ' + str(c + 1) + ' (Size: ' + str(round(best_model['pi'][c] * 100, 2)) + '%) ---')
        for m in range(2):
            print('  ' + var_names[m] + ':')
            for cat in range(6):
                prob = best_model['theta'][c, m, cat]
                print('    ' + cat_names[cat] + ': ' + str(round(prob, 4)))
    df['LCA_Class'] = np.argmax(best_model['gamma'], axis=1) + 1
    df.to_csv(os.path.join('data', 'cleaned_dataset_with_classes.csv'), index=False)
    print('\nClass membership assignments saved to data/cleaned_dataset_with_classes.csv')