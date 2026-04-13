# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class LCA:
    def __init__(self, n_classes, K=None, n_init=20, max_iter=1000, tol=1e-6, random_state=None):
        self.n_classes = n_classes
        self.K_provided = K
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        self.N, self.M = X.shape
        if self.K_provided is not None:
            self.K = self.K_provided
        else:
            self.K = [int(np.nanmax(X[:, m])) + 1 for m in range(self.M)]
            
        X_onehot = []
        valid_mask = []
        for m in range(self.M):
            valid = ~np.isnan(X[:, m])
            valid_mask.append(valid)
            onehot = np.zeros((self.N, self.K[m]))
            onehot[valid, X[valid, m].astype(int)] = 1
            X_onehot.append(onehot)
            
        best_ll = -np.inf
        best_pi = None
        best_theta = None
        best_gamma = None
        
        for init in range(self.n_init):
            pi = np.random.dirichlet(np.ones(self.n_classes))
            theta = []
            for m in range(self.M):
                theta_m = np.random.dirichlet(np.ones(self.K[m]), size=self.n_classes)
                theta.append(theta_m)
                
            ll_old = -np.inf
            
            for it in range(self.max_iter):
                log_gamma = np.zeros((self.N, self.n_classes))
                log_gamma += np.log(pi + 1e-15)
                for m in range(self.M):
                    log_gamma[valid_mask[m]] += (X_onehot[m][valid_mask[m]] @ np.log(theta[m].T + 1e-15))
                    
                max_log_gamma = np.max(log_gamma, axis=1, keepdims=True)
                gamma_unnorm = np.exp(log_gamma - max_log_gamma)
                sum_gamma = np.sum(gamma_unnorm, axis=1, keepdims=True)
                gamma = gamma_unnorm / sum_gamma
                
                ll = np.sum(max_log_gamma + np.log(sum_gamma))
                
                if ll - ll_old < self.tol:
                    break
                ll_old = ll
                
                N_c = np.sum(gamma, axis=0) + 1e-6
                pi = N_c / (self.N + self.n_classes * 1e-6)
                
                for m in range(self.M):
                    gamma_valid = gamma[valid_mask[m]]
                    X_valid = X_onehot[m][valid_mask[m]]
                    num = gamma_valid.T @ X_valid + 1e-6
                    den = np.sum(gamma_valid, axis=0, keepdims=True).T + self.K[m] * 1e-6
                    theta[m] = num / den
                    
            if ll > best_ll:
                best_ll = ll
                best_pi = pi
                best_theta = theta
                best_gamma = gamma
                
        self.pi_ = best_pi
        self.theta_ = best_theta
        self.gamma_ = best_gamma
        self.ll_ = best_ll
        
        n_params = self.n_classes - 1 + self.n_classes * sum(k - 1 for k in self.K)
        self.bic_ = -2 * self.ll_ + n_params * np.log(self.N)
        self.aic_ = -2 * self.ll_ + 2 * n_params
        
        ent = -np.sum(self.gamma_ * np.log(self.gamma_ + 1e-15))
        if self.n_classes > 1:
            self.entropy_ = 1 - ent / (self.N * np.log(self.n_classes))
        else:
            self.entropy_ = 1.0
            
        return self

def main():
    data_dir = "data/"
    file_path = os.path.join(data_dir, "processed_data.csv")
    
    df = pd.read_csv(file_path, low_memory=False)
    
    if 'QEA_2_Not_Sure' not in df.columns:
        df['QEA_2_Not_Sure'] = 0
    if 'QEB_2_Not_Sure' not in df.columns:
        df['QEB_2_Not_Sure'] = 0
        
    X1_distinct = df['QEA_2'] + 2
    X1_distinct = np.where(df['QEA_2_Not_Sure'] == 1, 5, X1_distinct)
    
    X1_missing = df['QEA_2'] + 2
    X1_missing = np.where(df['QEA_2_Not_Sure'] == 1, np.nan, X1_missing)
    
    X2_distinct = df['QEB_2'] + 2
    X2_distinct = np.where(df['QEB_2_Not_Sure'] == 1, 5, X2_distinct)
    
    X2_missing = df['QEB_2'] + 2
    X2_missing = np.where(df['QEB_2_Not_Sure'] == 1, np.nan, X2_missing)
    
    X_distinct = np.column_stack((X1_distinct, X2_distinct))
    X_missing = np.column_stack((X1_missing, X2_missing))
    
    results_distinct = []
    results_missing = []
    models_distinct = {}
    models_missing = {}
    
    for c in range(2, 6):
        lca_d = LCA(n_classes=c, K=[6, 6], n_init=20, random_state=42)
        lca_d.fit(X_distinct)
        results_distinct.append({'Classes': c, 'BIC': lca_d.bic_, 'AIC': lca_d.aic_, 'Entropy': lca_d.entropy_})
        models_distinct[c] = lca_d
        
        lca_m = LCA(n_classes=c, K=[5, 5], n_init=20, random_state=42)
        lca_m.fit(X_missing)
        results_missing.append({'Classes': c, 'BIC': lca_m.bic_, 'AIC': lca_m.aic_, 'Entropy': lca_m.entropy_})
        models_missing[c] = lca_m
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    classes = [r['Classes'] for r in results_distinct]
    bic_d = [r['BIC'] for r in results_distinct]
    aic_d = [r['AIC'] for r in results_distinct]
    ent_d = [r['Entropy'] for r in results_distinct]
    
    ax1 = axes[0]
    ax1.plot(classes, bic_d, marker='o', color='tab:blue', label='BIC')
    ax1.plot(classes, aic_d, marker='s', color='tab:orange', label='AIC')
    ax1.set_xlabel('Number of Classes')
    ax1.set_ylabel('Information Criterion')
    ax1.set_title('Model 1: "Not sure" as Distinct Category')
    ax1.set_xticks(classes)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(classes, ent_d, marker='^', color='tab:green', label='Entropy')
    ax1_twin.set_ylabel('Entropy')
    ax1_twin.set_ylim(0, 1.05)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    bic_m = [r['BIC'] for r in results_missing]
    aic_m = [r['AIC'] for r in results_missing]
    ent_m = [r['Entropy'] for r in results_missing]
    
    ax2 = axes[1]
    ax2.plot(classes, bic_m, marker='o', color='tab:blue', label='BIC')
    ax2.plot(classes, aic_m, marker='s', color='tab:orange', label='AIC')
    ax2.set_xlabel('Number of Classes')
    ax2.set_ylabel('Information Criterion')
    ax2.set_title('Model 2: "Not sure" as Missing')
    ax2.set_xticks(classes)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(classes, ent_m, marker='^', color='tab:green', label='Entropy')
    ax2_twin.set_ylabel('Entropy')
    ax2_twin.set_ylim(0, 1.05)
    
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    fig.tight_layout()
    timestamp = int(time.time())
    plot_path = os.path.join(data_dir, "lca_fit_indices_" + str(timestamp) + ".png")
    fig.savefig(plot_path, dpi=300)
    
    valid_models = [r for r in results_distinct if r['Entropy'] > 0.8]
    if valid_models:
        best_c = min(valid_models, key=lambda x: x['BIC'])['Classes']
    else:
        best_c = min(results_distinct, key=lambda x: x['BIC'])['Classes']
        
    best_model = models_distinct[best_c]
    
    class_assignments = np.argmax(best_model.gamma_, axis=1) + 1
    df['LCA_Class'] = class_assignments.astype(float)
    
    all_missing = np.isnan(X_distinct).all(axis=1)
    df.loc[all_missing, 'LCA_Class'] = np.nan
    
    out_path = os.path.join(data_dir, "processed_data.csv")
    df.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()