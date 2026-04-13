# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

class SimpleLCA:
    def __init__(self, n_components, n_iter=500, n_init=10, random_state=42, tol=1e-5):
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_init = n_init
        self.random_state = random_state
        self.tol = tol
    def fit(self, X):
        np.random.seed(self.random_state)
        self.N, self.D = X.shape
        self.categories = [max(np.max(X[:, d]) + 1, 6) for d in range(self.D)]
        best_log_likelihood = -np.inf
        best_pi = None
        best_theta = None
        best_gamma = None
        for init in range(self.n_init):
            pi = np.random.dirichlet(np.ones(self.n_components))
            theta = []
            for k in range(self.n_components):
                theta_k = []
                for d in range(self.D):
                    theta_k.append(np.random.dirichlet(np.ones(self.categories[d])))
                theta.append(theta_k)
            log_likelihood_old = -np.inf
            for iteration in range(self.n_iter):
                log_gamma = np.zeros((self.N, self.n_components))
                for k in range(self.n_components):
                    log_gamma[:, k] = np.log(pi[k] + 1e-15)
                    for d in range(self.D):
                        log_gamma[:, k] += np.log(theta[k][d][X[:, d]] + 1e-15)
                max_log_gamma = np.max(log_gamma, axis=1, keepdims=True)
                gamma_unnorm = np.exp(log_gamma - max_log_gamma)
                gamma_sum = np.sum(gamma_unnorm, axis=1, keepdims=True)
                gamma = gamma_unnorm / gamma_sum
                log_likelihood = np.sum(max_log_gamma + np.log(gamma_sum))
                if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                    break
                log_likelihood_old = log_likelihood
                N_k = np.sum(gamma, axis=0)
                pi = N_k / self.N
                for k in range(self.n_components):
                    for d in range(self.D):
                        for c in range(self.categories[d]):
                            mask = (X[:, d] == c)
                            theta[k][d][c] = np.sum(gamma[mask, k]) / (N_k[k] + 1e-15)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_pi = pi
                best_theta = theta
                best_gamma = gamma
        self.pi = best_pi
        self.theta = best_theta
        self.gamma = best_gamma
        self.log_likelihood_ = best_log_likelihood
        n_params = (self.n_components - 1)
        for d in range(self.D):
            n_params += self.n_components * (self.categories[d] - 1)
        self.bic_ = -2 * self.log_likelihood_ + n_params * np.log(self.N)
        self.aic_ = -2 * self.log_likelihood_ + 2 * n_params
        return self
    def predict_proba(self, X):
        log_gamma = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_gamma[:, k] = np.log(self.pi[k] + 1e-15)
            for d in range(self.D):
                log_gamma[:, k] += np.log(self.theta[k][d][X[:, d]] + 1e-15)
        max_log_gamma = np.max(log_gamma, axis=1, keepdims=True)
        gamma_unnorm = np.exp(log_gamma - max_log_gamma)
        return gamma_unnorm / np.sum(gamma_unnorm, axis=1, keepdims=True)
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

if __name__ == '__main__':
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_data_step2.csv')
    df = pd.read_csv(data_path)
    raw_data_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    raw_df = pd.read_csv(raw_data_path, sep='\t', low_memory=False)
    qea_2_col = next((c for c in raw_df.columns if c.startswith('QEA_2')), None)
    qeb_2_col = next((c for c in raw_df.columns if c.startswith('QEB_2')), None)
    def robust_map_qea(val):
        if pd.isna(val) or str(val).strip() == '.': return np.nan
        v = str(val).lower()
        if 'significantly negative' in v: return -2
        if 'slightly negative' in v: return -1
        if 'no impact' in v: return 0
        if 'slightly positive' in v: return 1
        if 'significantly positive' in v: return 2
        if 'not sure' in v: return np.nan
        return np.nan
    if qea_2_col:
        df['Job_Security_Current'] = raw_df[qea_2_col].apply(robust_map_qea)
    if qeb_2_col:
        df['Job_Security_Future'] = raw_df[qeb_2_col].apply(robust_map_qea)
    def map_to_cat(x):
        if pd.isna(x):
            return 5
        return int(x + 2)
    X_df = pd.DataFrame()
    X_df['Job_Security_Current_Cat'] = df['Job_Security_Current'].apply(map_to_cat)
    X_df['Job_Security_Future_Cat'] = df['Job_Security_Future'].apply(map_to_cat)
    X = X_df.values
    models = {}
    results = []
    for n_classes in [2, 3, 4]:
        model = SimpleLCA(n_components=n_classes, n_init=10, random_state=42)
        model.fit(X)
        preds = model.predict_proba(X)
        entropy = -np.sum(preds * np.log(preds + 1e-15))
        norm_entropy = 1 - (entropy / (len(X) * np.log(n_classes)))
        models[n_classes] = model
        results.append({'n_classes': n_classes, 'BIC': model.bic_, 'AIC': model.aic_, 'Entropy': norm_entropy})
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    best_n = 3
    best_model = models[best_n]
    df['LCA_Class'] = best_model.predict(X)
    desc_stats = df.groupby('LCA_Class')[['Job_Security_Current', 'Job_Security_Future']].mean()
    class_means = desc_stats.sum(axis=1, skipna=True)
    sorted_classes = class_means.sort_values().index.tolist()
    class_name_map = {sorted_classes[0]: 'Anxiously Declining', sorted_classes[1]: 'Stagnant Neutral', sorted_classes[2]: 'Resiliently Optimistic'}
    df['LCA_Class_Name'] = df['LCA_Class'].map(class_name_map)
    output_path = os.path.join(data_dir, 'processed_data_step3.csv')
    df.to_csv(output_path, index=False)
    print('Updated dataset saved to ' + output_path)