# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def tetrachoric_corr_matrix(df_bin):
    vals = df_bin.values
    n_cols = vals.shape[1]
    R = np.eye(n_cols)
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            x = vals[:, i]
            y = vals[:, j]
            valid = ~(np.isnan(x) | np.isnan(y))
            x_v = x[valid]
            y_v = y[valid]
            if len(x_v) == 0:
                r = 0
            else:
                a = np.sum((x_v == 0) & (y_v == 0)) + 0.5
                b = np.sum((x_v == 0) & (y_v == 1)) + 0.5
                c = np.sum((x_v == 1) & (y_v == 0)) + 0.5
                d = np.sum((x_v == 1) & (y_v == 1)) + 0.5
                ratio = (a * d) / (b * c)
                r = np.cos(np.pi / (1 + np.sqrt(ratio)))
            R[i, j] = r
            R[j, i] = r
    return R

def make_psd(R):
    evals, evecs = np.linalg.eigh(R)
    evals[evals < 0] = 1e-8
    R_psd = evecs @ np.diag(evals) @ evecs.T
    d = np.diag(R_psd)
    R_psd = R_psd / np.sqrt(np.outer(d, d))
    return R_psd

def parallel_analysis(df_bin, n_iter=30):
    n_rows, n_cols = df_bin.shape
    marginal_probs = df_bin.mean().fillna(0.5).values
    random_evals = np.zeros((n_iter, n_cols))
    for i in range(n_iter):
        rand_data = np.random.rand(n_rows, n_cols) < marginal_probs
        rand_df = pd.DataFrame(rand_data.astype(int))
        R_rand = tetrachoric_corr_matrix(rand_df)
        R_rand_psd = make_psd(R_rand)
        evals, _ = np.linalg.eigh(R_rand_psd)
        random_evals[i, :] = np.sort(evals)[::-1]
    return np.mean(random_evals, axis=0)

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R)

if __name__ == '__main__':
    data_dir = "data/"
    input_path = os.path.join(data_dir, "cleaned_dataset.csv")
    df = pd.read_csv(input_path, low_memory=False)
    qhd_cols = [c for c in df.columns if re.match(r'^QHD_\d+\b', c)]
    qhd_cols = sorted(qhd_cols, key=lambda x: int(re.search(r'^QHD_(\d+)\b', x).group(1)))
    R_tet = tetrachoric_corr_matrix(df[qhd_cols])
    R_psd = make_psd(R_tet)
    evals, evecs = np.linalg.eigh(R_psd)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    pa_evals = parallel_analysis(df[qhd_cols], n_iter=30)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(evals) + 1), evals, marker='o', linestyle='-', label='Actual Data')
    plt.plot(range(1, len(pa_evals) + 1), pa_evals, marker='x', linestyle='--', color='red', label='Parallel Analysis (Random)')
    plt.title('Scree Plot & Parallel Analysis of Tetrachoric Correlation Matrix')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = os.path.join(data_dir, "scree_plot_" + str(timestamp) + ".png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    n_factors = 2
    L = evecs[:, :n_factors] * np.sqrt(evals[:n_factors])
    L_rot = varimax(L)
    pos_loading_f0 = np.sum(L_rot[:6, 0])
    pos_loading_f1 = np.sum(L_rot[:6, 1])
    if pos_loading_f0 > pos_loading_f1:
        pos_idx, neg_idx = 0, 1
    else:
        pos_idx, neg_idx = 1, 0
    if np.sum(L_rot[:6, pos_idx]) < 0:
        L_rot[:, pos_idx] = -L_rot[:, pos_idx]
    if np.sum(L_rot[-6:, neg_idx]) < 0:
        L_rot[:, neg_idx] = -L_rot[:, neg_idx]
    Z = df[qhd_cols].copy()
    Z = (Z - Z.mean()) / Z.std()
    Z = Z.fillna(0)
    R_inv = np.linalg.pinv(R_psd)
    W = R_inv @ L_rot
    scores = Z.values @ W
    df['Positive_Affect'] = scores[:, pos_idx]
    df['Negative_Affect'] = scores[:, neg_idx]
    communalities = np.sum(L_rot**2, axis=1)
    uniquenesses = 1 - communalities
    factor_assignment = np.argmax(np.abs(L_rot), axis=1)
    def compute_omega(factor_idx, factor_assignment, L_rot, uniquenesses):
        items = (factor_assignment == factor_idx)
        if np.sum(items) == 0: return 0
        loadings = np.abs(L_rot[items, factor_idx])
        sum_l = np.sum(loadings)
        sum_u = np.sum(uniquenesses[items])
        return (sum_l**2) / (sum_l**2 + sum_u)
    omega_pos = compute_omega(pos_idx, factor_assignment, L_rot, uniquenesses)
    omega_neg = compute_omega(neg_idx, factor_assignment, L_rot, uniquenesses)
    qkb_pairs = []
    for i in range(1, 12):
        c1 = [c for c in df.columns if re.match(rf'^QKB_1_{i}\b', c)]
        c2 = [c for c in df.columns if re.match(rf'^QKB_2_{i}\b', c)]
        if c1 and c2:
            qkb_pairs.append((c1[0], c2[0]))
    gap_cols = []
    for i, (c1, c2) in enumerate(qkb_pairs):
        gap_col = 'QKB_Gap_' + str(i+1)
        df[gap_col] = df[c1] - df[c2]
        gap_cols.append(gap_col)
    imputer = SimpleImputer(strategy='mean')
    gap_data_imputed = imputer.fit_transform(df[gap_cols])
    pca = PCA(n_components=1)
    org_support_pc = pca.fit_transform(gap_data_imputed).flatten()
    corr_with_gap = np.corrcoef(org_support_pc, gap_data_imputed[:, 0])[0, 1]
    if corr_with_gap > 0:
        org_support_pc = -org_support_pc
    df['Organizational_Support_Index'] = org_support_pc
    output_path = os.path.join(data_dir, "cleaned_dataset_step2.csv")
    df.to_csv(output_path, index=False)