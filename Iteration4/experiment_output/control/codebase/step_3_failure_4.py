# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

def fit_lca(X, n_classes, n_init=20, max_iter=2000, tol=1e-7):
    N, J = X.shape
    K = [6, 6]
    best_ll = -np.inf
    best_pi = None
    best_theta = None
    best_gamma = None
    np.random.seed(42 + n_classes)
    alpha = 1.001
    for init in range(n_init):
        pi = np.random.dirichlet(np.ones(n_classes))
        theta = []
        for j in range(J):
            theta.append(np.random.dirichlet(np.ones(K[j]), size=n_classes))
        ll_old = -np.inf
        for iteration in range(max_iter):
            gamma = np.zeros((N, n_classes))
            for c in range(n_classes):
                prob = np.ones(N) * pi[c]
                for j in range(J):
                    prob *= theta[j][c, X[:, j]]
                gamma[:, c] = prob
            prob_sum = gamma.sum(axis=1, keepdims=True)
            prob_sum[prob_sum == 0] = 1e-15
            ll = np.sum(np.log(prob_sum))
            gamma /= prob_sum
            gamma_sum = gamma.sum(axis=0)
            pi = (gamma_sum + alpha - 1) / (N + n_classes * (alpha - 1))
            for j in range(J):
                for k in range(K[j]):
                    mask_k = (X[:, j] == k)
                    sum_gamma_k = gamma[mask_k, :].sum(axis=0)
                    theta[j][:, k] = (sum_gamma_k + alpha - 1) / (gamma_sum + K[j] * (alpha - 1))
            if np.abs(ll - ll_old) < tol:
                break
            ll_old = ll
        if ll > best_ll:
            best_ll = ll
            best_pi = pi
            best_theta = theta
            best_gamma = gamma
    n_params = (n_classes - 1) + sum(n_classes * (k - 1) for k in K)
    aic = 2 * n_params - 2 * best_ll
    bic = np.log(N) * n_params - 2 * best_ll
    eps = 1e-15
    entropy_val = 1 - np.sum(-best_gamma * np.log(best_gamma + eps)) / (N * np.log(n_classes))
    return {'ll': best_ll, 'pi': best_pi, 'theta': best_theta, 'gamma': best_gamma, 'aic': aic, 'bic': bic, 'entropy': entropy_val, 'n_params': n_params, 'n_classes': n_classes}

def label_classes(theta, pi):
    scores = []
    for c in range(3):
        pos = theta[1][c, 3] + theta[1][c, 4]
        neg = theta[1][c, 0] + theta[1][c, 1]
        neu = theta[1][c, 2]
        scores.append((int(c), pos, neg, neu))
    opt_c = max(scores, key=lambda x: x[1])[0]
    rem1 = [x for x in scores if x[0] != opt_c]
    dec_c = max(rem1, key=lambda x: x[2])[0]
    neu_c = [x for x in scores if x[0] not in (opt_c, dec_c)][0][0]
    labels = {int(opt_c): 'Resiliently Optimistic', int(dec_c): 'Anxiously Declining', int(neu_c): 'Stagnant Neutral'}
    return labels

if __name__ == '__main__':
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'processed_data.csv')
    print('Loading dataset from ' + input_path + '...')
    df = pd.read_csv(input_path, low_memory=False)
    mapping = {'significantly negative': 0, 'slightly negative': 1, 'no impact': 2, 'slightly positive': 3, 'significantly positive': 4, 'not sure': 5}
    reverse_mapping = {v: k for k, v in mapping.items()}
    df['QEA_2_nom'] = df['QEA_2_raw'].astype(str).str.strip().str.lower().map(mapping)
    df['QEB_2_nom'] = df['QEB_2_raw'].astype(str).str.strip().str.lower().map(mapping)
    mask = df['QEA_2_nom'].notna() & df['QEB_2_nom'].notna()
    X = df.loc[mask, ['QEA_2_nom', 'QEB_2_nom']].values.astype(int)
    print('--- Input Data Summary ---')
    print('Total respondents: ' + str(len(df)))
    print('Respondents with valid QEA_2 and QEB_2: ' + str(len(X)))
    print('Fitting LCA models on ' + str(len(X)) + ' respondents...')
    models = {}
    for c in [2, 3, 4]:
        models[c] = fit_lca(X, n_classes=c, n_init=20, max_iter=2000)
    print('\n--- LCA Model Comparison ---')
    print('Classes    | BIC        | AIC        | Entropy    | Log-Likelihood')
    for c in [2, 3, 4]:
        m = models[c]
        bic_str = str(round(m['bic'], 2)).ljust(10)
        aic_str = str(round(m['aic'], 2)).ljust(10)
        ent_str = str(round(m['entropy'], 3)).ljust(10)
        ll_str = str(round(m['ll'], 2)).ljust(15)
        print(str(c).ljust(10) + ' | ' + bic_str + ' | ' + aic_str + ' | ' + ent_str + ' | ' + ll_str)
    best_model = models[3]
    labels = label_classes(best_model['theta'], best_model['pi'])
    print('\n--- Conditional Response Probabilities (3-Class Model) ---')
    for c in range(3):
        label = labels[int(c)]
        size_pct = str(round(best_model['pi'][c] * 100, 1)) + '%'
        print('\nClass: ' + label + ' (Size: ' + size_pct + '):')
        ind_names = ['QEA_2 (Current)', 'QEB_2 (Expected)']
        for j in range(2):
            print('  ' + ind_names[j] + ':')
            for k in range(6):
                cat_name = reverse_mapping[k].capitalize()
                prob_pct = str(round(best_model['theta'][j][c, k] * 100, 1)) + '%'
                print('    ' + cat_name + ': ' + prob_pct)
    class_preds = np.argmax(best_model['gamma'], axis=1)
    class_names = [labels[int(c)] for c in class_preds]
    df['LCA_Class'] = np.nan
    df.loc[mask, 'LCA_Class'] = class_names
    df.to_csv(input_path, index=False)
    print('\nUpdated dataset with LCA class membership saved to ' + input_path)