# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from scipy.linalg import eigh, inv

def tetrachoric_corr(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    crosstab = pd.crosstab(x, y)
    crosstab = crosstab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    crosstab = crosstab + 0.5
    a = crosstab.iloc[0, 0]
    b = crosstab.iloc[0, 1]
    c = crosstab.iloc[1, 0]
    d = crosstab.iloc[1, 1]
    r = np.cos(np.pi / (1 + np.sqrt((a * d) / (b * c))))
    return r

def make_psd(matrix):
    vals, vecs = eigh(matrix)
    if np.any(vals < 0):
        vals[vals < 0] = 1e-8
        matrix = vecs @ np.diag(vals) @ vecs.T
        d = np.diag(matrix)
        matrix = matrix / np.sqrt(np.outer(d, d))
    return matrix

if __name__ == '__main__':
    data_path = os.path.join('data', 'IKI_Cleaned_Features.csv')
    df = pd.read_csv(data_path, low_memory=False)
    qhd_cols = [c for c in df.columns if c.startswith('QHD_')]
    n = len(qhd_cols)
    tet_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = tetrachoric_corr(df[qhd_cols[i]], df[qhd_cols[j]])
            tet_mat[i, j] = r
            tet_mat[j, i] = r
    tet_mat = make_psd(tet_mat)
    vals, vecs = eigh(tet_mat)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    L = vecs[:, :2] @ np.diag(np.sqrt(np.maximum(vals[:2], 0)))
    uniquenesses = 1 - np.sum(L**2, axis=1)
    factor_assignment = np.argmax(np.abs(L), axis=1)
    omegas = []
    for i in range(2):
        mask = (factor_assignment == i)
        if np.sum(mask) == 0:
            omegas.append(0.0)
        else:
            l_i = L[mask, i]
            u_i = uniquenesses[mask]
            omegas.append(np.sum(l_i)**2 / (np.sum(l_i)**2 + np.sum(u_i)))
    Z = (df[qhd_cols] - df[qhd_cols].mean()) / df[qhd_cols].std()
    Z = Z.fillna(0)
    F = Z.values @ inv(tet_mat) @ L
    sum_pos_f0 = np.sum(L[:6, 0])
    sum_pos_f1 = np.sum(L[:6, 1])
    pos_idx, neg_idx = (0, 1) if sum_pos_f0 > sum_pos_f1 else (1, 0)
    df['Positive_Affect'] = F[:, pos_idx]
    df['Negative_Affect'] = F[:, neg_idx]
    output_path = os.path.join('data', 'IKI_Cleaned_Features_Augmented.csv')
    df.to_csv(output_path, index=False)
    print('Augmented dataset saved to ' + output_path)