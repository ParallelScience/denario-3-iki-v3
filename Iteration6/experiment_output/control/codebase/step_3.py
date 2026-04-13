# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def tetrachoric_corr_matrix(df_bin):
    cols = df_bin.columns
    n = len(cols)
    R = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            x = df_bin.iloc[:, i]
            y = df_bin.iloc[:, j]
            crosstab = pd.crosstab(x, y)
            a = crosstab.get(1, {}).get(1, 0) + 0.5
            b = crosstab.get(0, {}).get(1, 0) + 0.5
            c = crosstab.get(1, {}).get(0, 0) + 0.5
            d = crosstab.get(0, {}).get(0, 0) + 0.5
            omega = (a * d) / (b * c)
            r = np.cos(np.pi / (1 + np.sqrt(omega)))
            R[i, j] = r
            R[j, i] = r
    return R

def make_psd(matrix):
    vals, vecs = np.linalg.eigh(matrix)
    vals[vals < 1e-4] = 1e-4
    return vecs @ np.diag(vals) @ vecs.T

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
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return np.dot(Phi, R)

if __name__ == '__main__':
    data_path = os.path.join('data', 'cleaned_dataset_with_classes.csv')
    df = pd.read_csv(data_path, low_memory=False)
    qhd_cols = [c for c in df.columns if c.startswith('QHD_')]
    if len(qhd_cols) == 0:
        qhd_cols = [c for c in df.columns if c.startswith('QHD')]
    print('Found ' + str(len(qhd_cols)) + ' QHD items for EFA.')
    short_names = []
    for c in qhd_cols:
        if ':' in c and '-' in c:
            name = c.split(':')[1].split('-')[0].strip()
        elif ':' in c:
            name = c.split(':')[1].strip()
        else:
            name = c
        short_names.append(name)
    df_bin = df[qhd_cols].copy()
    print('Computing tetrachoric correlation matrix...')
    R = tetrachoric_corr_matrix(df_bin)
    print('Extracting factors...')
    vals, vecs = np.linalg.eigh(R)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    L = vecs[:, :2] * np.sqrt(vals[:2])
    L_rot = varimax(L)
    pos_words = ['optimistic', 'empowered', 'motivated', 'excited', 'curious', 'confident']
    pos_indices = [i for i, name in enumerate(short_names) if any(pw in name.lower() for pw in pos_words)]
    if len(pos_indices) > 0:
        sum_f0 = np.sum(L_rot[pos_indices, 0])
        sum_f1 = np.sum(L_rot[pos_indices, 1])
        if sum_f0 > sum_f1:
            pos_idx, neg_idx = 0, 1
        else:
            pos_idx, neg_idx = 1, 0
    else:
        pos_idx, neg_idx = 0, 1
    L_final = np.zeros_like(L_rot)
    L_final[:, 0] = L_rot[:, pos_idx]
    L_final[:, 1] = L_rot[:, neg_idx]
    print('\nCalculating McDonald\'s Omega...')
    omega_dict = {}
    for factor_name, f_idx in [('Positive Affect', 0), ('Negative Affect', 1)]:
        item_indices = [i for i in range(len(qhd_cols)) if np.abs(L_final[i, f_idx]) > np.abs(L_final[i, 1 - f_idx])]
        loadings = np.abs(L_final[item_indices, f_idx])
        communality = L_final[item_indices, 0]**2 + L_final[item_indices, 1]**2
        u2 = 1 - communality
        sum_loadings = np.sum(loadings)
        sum_u2 = np.sum(u2)
        omega = (sum_loadings**2) / (sum_loadings**2 + sum_u2)
        omega_dict[factor_name] = omega
        print('McDonald\'s Omega for ' + factor_name + ': ' + str(round(omega, 4)) + ' (based on ' + str(len(item_indices)) + ' items)')
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 10))
    sns.heatmap(L_final, annot=True, cmap='coolwarm', center=0, yticklabels=short_names, xticklabels=['Positive Affect', 'Negative Affect'], vmin=-1, vmax=1, fmt='.2f')
    plt.title('Factor Loadings of QHD Items (Varimax Rotation)')
    plt.tight_layout()
    plot_filename = os.path.join('data', 'qhd_factor_loadings_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('\nFactor loading heatmap saved to ' + plot_filename)
    plt.close()
    R_psd = make_psd(R)
    R_inv = np.linalg.inv(R_psd)
    W = R_inv @ L_final
    Z = StandardScaler().fit_transform(df_bin)
    scores = Z @ W
    df['Positive_Affect'] = scores[:, 0]
    df['Negative_Affect'] = scores[:, 1]
    qf_cols = [c for c in df.columns if c.startswith('QF_')]
    print('\nCorrelations between Negative Affect and QF deterrent items:')
    for c in qf_cols:
        valid = df[[c, 'Negative_Affect']].dropna()
        if len(valid) > 0:
            corr, pval = pearsonr(valid[c], valid['Negative_Affect'])
            c_name = c[:50] + ('...' if len(c) > 50 else '')
            print(c_name + ': r = ' + str(round(corr, 4)) + ', p = ' + str(round(pval, 4)))
    output_path = os.path.join('data', 'cleaned_dataset_step3.csv')
    df.to_csv(output_path, index=False)
    print('\nDataset with factor scores saved to ' + output_path)