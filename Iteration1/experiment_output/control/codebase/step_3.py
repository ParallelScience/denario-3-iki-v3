# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from scipy.special import logsumexp
class LCA:
    def __init__(self, n_classes, n_init=10, max_iter=1000, tol=1e-6, random_state=None):
        self.n_classes = n_classes
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    def fit(self, X, K_list):
        N, J = X.shape
        self.K = K_list
        best_ll = -np.inf
        best_pi = None
        best_theta = None
        best_resp = None
        rng = np.random.default_rng(self.random_state)
        for init in range(self.n_init):
            pi = rng.dirichlet(np.ones(self.n_classes))
            theta = []
            for j in range(J):
                theta_j = rng.dirichlet(np.ones(self.K[j]), size=self.n_classes)
                theta.append(theta_j)
            ll_old = -np.inf
            for it in range(self.max_iter):
                log_pi = np.log(pi + 1e-15)
                log_theta = [np.log(theta[j] + 1e-15) for j in range(J)]
                log_resp = np.zeros((N, self.n_classes))
                for c in range(self.n_classes):
                    log_resp[:, c] = log_pi[c]
                    for j in range(J):
                        log_resp[:, c] += log_theta[j][c, X[:, j]]
                log_prob_X = logsumexp(log_resp, axis=1)
                ll = np.sum(log_prob_X)
                resp = np.exp(log_resp - log_prob_X[:, np.newaxis])
                N_c = np.sum(resp, axis=0)
                pi = N_c / N
                for j in range(J):
                    for k in range(self.K[j]):
                        mask = (X[:, j] == k)
                        if np.any(mask):
                            theta[j][:, k] = np.sum(resp[mask, :], axis=0) / (N_c + 1e-15)
                        else:
                            theta[j][:, k] = 0.0
                if np.abs(ll - ll_old) < self.tol:
                    break
                ll_old = ll
            if ll > best_ll:
                best_ll = ll
                best_pi = pi
                best_theta = theta
                best_resp = resp
        self.pi_ = best_pi
        self.theta_ = best_theta
        self.resp_ = best_resp
        self.ll_ = best_ll
        df = (self.n_classes - 1) + self.n_classes * sum(k - 1 for k in self.K)
        self.aic_ = 2 * df - 2 * self.ll_
        self.bic_ = np.log(N) * df - 2 * self.ll_
        entropy_num = -np.sum(self.resp_ * np.log(self.resp_ + 1e-15))
        self.entropy_ = 1 - entropy_num / (N * np.log(self.n_classes))
        return self
if __name__ == '__main__':
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'IKI_Cleaned_Features_Augmented.csv')
    df = pd.read_csv(data_path, low_memory=False)
    qea_col = next((col for col in df.columns if col.startswith('QEA_2')), None)
    qeb_col = next((col for col in df.columns if col.startswith('QEB_2')), None)
    if qea_col is None or qeb_col is None:
        print('Error: Could not find columns starting with QEA_2 or QEB_2.')
        sys.exit(1)
    print('Using columns:')
    print('QEA_2 -> ' + qea_col)
    print('QEB_2 -> ' + qeb_col)
    valid_mask = df[qea_col].notna() & df[qeb_col].notna()
    df_lca = df[valid_mask].copy()
    val_map = {-2.0: 0, -1.0: 1, 0.0: 2, 1.0: 3, 2.0: 4, 99.0: 5}
    inv_map = {0: '-2 (Sig. Neg)', 1: '-1 (Slight. Neg)', 2: '0 (No Impact)', 3: '1 (Slight. Pos)', 4: '2 (Sig. Pos)', 5: '99 (Not Sure)'}
    X = np.zeros((len(df_lca), 2), dtype=int)
    X[:, 0] = df_lca[qea_col].map(val_map).values
    X[:, 1] = df_lca[qeb_col].map(val_map).values
    K_list = [6, 6]
    results = []
    models = {}
    print('\nFitting LCA models for 2 to 5 classes...')
    for c in range(2, 6):
        lca = LCA(n_classes=c, n_init=20, random_state=42)
        lca.fit(X, K_list)
        models[c] = lca
        results.append({'Classes': c, 'Log-Likelihood': lca.ll_, 'AIC': lca.aic_, 'BIC': lca.bic_, 'Entropy': lca.entropy_})
    results_df = pd.DataFrame(results)
    print('\nLCA Model Comparison:')
    print(results_df.to_string(index=False))
    comp_path = os.path.join(data_dir, 'LCA_Model_Comparison.csv')
    results_df.to_csv(comp_path, index=False)
    print('\nModel comparison saved to ' + comp_path)
    best_c = results_df.loc[results_df['BIC'].idxmin(), 'Classes']
    print('\nSelected best model based on BIC: ' + str(best_c) + ' classes')
    best_model = models[best_c]
    class_scores = []
    for c in range(best_model.n_classes):
        score = 0
        for j in range(2):
            for k in range(5):
                val = k - 2
                score += best_model.theta_[j][c, k] * val
        class_scores.append(score)
    sorted_classes = np.argsort(class_scores)[::-1]
    best_model.pi_ = best_model.pi_[sorted_classes]
    for j in range(2):
        best_model.theta_[j] = best_model.theta_[j][sorted_classes]
    best_model.resp_ = best_model.resp_[:, sorted_classes]
    print('\nConditional Response Probabilities for each Latent Class:')
    for c in range(best_model.n_classes):
        print('\nClass ' + str(c+1) + ' (Size: ' + str(round(best_model.pi_[c]*100, 1)) + '%):')
        for j, col_name in enumerate(['Current Impact', 'Future Impact']):
            print('  ' + col_name + ':')
            for k in range(best_model.K[j]):
                prob = best_model.theta_[j][c, k]
                print('    ' + inv_map[k] + ': ' + str(round(prob*100, 1)) + '%')
    df_lca['LCA_Class'] = np.argmax(best_model.resp_, axis=1) + 1
    df['LCA_Class'] = np.nan
    df.loc[valid_mask, 'LCA_Class'] = df_lca['LCA_Class']
    df['LCA_Class'] = df['LCA_Class'].astype('Int64')
    print('\nSummary Table of Class Sizes:')
    class_sizes = df['LCA_Class'].value_counts(dropna=False).sort_index()
    print(class_sizes.to_string())
    print('\nCross-tabulation: LCA_Class vs Current Impact')
    xtab_qea = pd.crosstab(df['LCA_Class'], df[qea_col])
    print(xtab_qea.to_string())
    print('\nCross-tabulation: LCA_Class vs Future Impact')
    xtab_qeb = pd.crosstab(df['LCA_Class'], df[qeb_col])
    print(xtab_qeb.to_string())
    output_path = os.path.join(data_dir, 'IKI_Cleaned_Features_LCA.csv')
    df.to_csv(output_path, index=False)
    print('\nDataset with LCA class labels saved to ' + output_path)