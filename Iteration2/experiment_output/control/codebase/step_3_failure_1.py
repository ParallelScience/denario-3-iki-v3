# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

class LCA:
    def __init__(self, n_components, max_iter=1000, tol=1e-6, n_init=10, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        N, M = X.shape
        self.categories_ = []
        X_mapped = np.zeros((N, M), dtype=int) - 1
        for m in range(M):
            cats = np.unique(X[:, m][~np.isnan(X[:, m])])
            self.categories_.append(cats)
            for i, c in enumerate(cats):
                X_mapped[X[:, m] == c, m] = i
        valid_masks = []
        valid_idxs = []
        for m in range(M):
            valid = X_mapped[:, m] != -1
            valid_masks.append(valid)
            valid_idxs.append(X_mapped[valid, m])
        m_step_masks = []
        for m in range(M):
            C_m = len(self.categories_[m])
            masks = []
            for c in range(C_m):
                masks.append(X_mapped[:, m] == c)
            m_step_masks.append(masks)
        best_ll = -np.inf
        for init in range(self.n_init):
            pi = np.random.dirichlet(np.ones(self.n_components))
            theta = []
            for m in range(M):
                C_m = len(self.categories_[m])
                t = np.random.dirichlet(np.ones(C_m), size=self.n_components)
                theta.append(t)
            ll_old = -np.inf
            for it in range(self.max_iter):
                log_prob = np.zeros((N, self.n_components))
                for k in range(self.n_components):
                    log_prob[:, k] = np.log(pi[k] + 1e-15)
                    for m in range(M):
                        valid = valid_masks[m]
                        idx = valid_idxs[m]
                        log_prob[valid, k] += np.log(theta[m][k, idx] + 1e-15)
                max_log_prob = np.max(log_prob, axis=1, keepdims=True)
                prob = np.exp(log_prob - max_log_prob)
                sum_prob = np.sum(prob, axis=1, keepdims=True)
                gamma = prob / sum_prob
                ll = np.sum(max_log_prob + np.log(sum_prob))
                if ll - ll_old < self.tol:
                    break
                ll_old = ll
                N_k = np.sum(gamma, axis=0)
                pi = N_k / N
                for m in range(M):
                    C_m = len(self.categories_[m])
                    denom = np.sum(gamma[valid_masks[m], :], axis=0) + 1e-15
                    for c in range(C_m):
                        theta[m][:, c] = np.sum(gamma[m_step_masks[m][c], :], axis=0) / denom
                    theta[m] = theta[m] / np.sum(theta[m], axis=1, keepdims=True)
            if ll > best_ll:
                best_ll = ll
                self.pi_ = pi
                self.theta_ = theta
                self.gamma_ = gamma
                self.ll_ = ll
        p = (self.n_components - 1)
        for m in range(M):
            p += self.n_components * (len(self.categories_[m]) - 1)
        self.bic_ = -2 * self.ll_ + p * np.log(N)
        self.aic_ = -2 * self.ll_ + 2 * p
        gamma_safe = np.clip(self.gamma_, 1e-15, 1)
        entropy_val = -np.sum(self.gamma_ * np.log(gamma_safe))
        self.entropy_ = 1 - entropy_val / (N * np.log(self.n_components)) if self.n_components > 1 else 1.0
        return self
    def predict(self, X=None):
        return np.argmax(self.gamma_, axis=1)

def main():
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'processed_data.csv')
    df = pd.read_csv(input_path, low_memory=False)
    X1 = df[['QEA_2', 'QEB_2']].copy()
    if 'QEA_2_Not_Sure' in df.columns:
        X1.loc[df['QEA_2_Not_Sure'] == 1, 'QEA_2'] = 3.0
    if 'QEB_2_Not_Sure' in df.columns:
        X1.loc[df['QEB_2_Not_Sure'] == 1, 'QEB_2'] = 3.0
    X2 = df[['QEA_2', 'QEB_2']].copy()
    results1 = {}
    results2 = {}
    models1 = {}
    models2 = {}
    for k in range(2, 6):
        lca1 = LCA(n_components=k, n_init=15, random_state=42)
        lca1.fit(X1.values)
        results1[k] = {'bic': lca1.bic_, 'aic': lca1.aic_, 'entropy': lca1.entropy_}
        models1[k] = lca1
        lca2 = LCA(n_components=k, n_init=15, random_state=42)
        lca2.fit(X2.values)
        results2[k] = {'bic': lca2.bic_, 'aic': lca2.aic_, 'entropy': lca2.entropy_}
        models2[k] = lca2
    best_k1 = None
    best_bic1 = np.inf
    for k, res in results1.items():
        if res['entropy'] > 0.8 and res['bic'] < best_bic1:
            best_bic1 = res['bic']
            best_k1 = k
    if best_k1 is None:
        best_k1 = min(results1.keys(), key=lambda k: results1[k]['bic'])
    optimal_model = models1[best_k1]
    cat_names = {-2.0: "Sig. Negative", -1.0: "Slightly Negative", 0.0: "No Impact", 1.0: "Slightly Positive", 2.0: "Sig. Positive", 3.0: "Not Sure"}
    df['LCA_Class'] = optimal_model.predict() + 1
    df.to_csv(input_path, index=False)

if __name__ == '__main__':
    main()