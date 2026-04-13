# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

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
    vals, vecs = np.linalg.eigh(matrix)
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
    print('Found ' + str(len(qhd_cols)) + ' QHD items for Affective Disposition Modeling.')
    print('Computing tetrachoric correlation matrix...')
    n = len(qhd_cols)
    tet_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = tetrachoric_corr(df[qhd_cols[i]], df[qhd_cols[j]])
            tet_mat[i, j] = r
            tet_mat[j, i] = r
    tet_mat = make_psd(tet_mat)
    print('Performing EFA (method=uls, rotation=promax)...')
    fa = FactorAnalyzer(n_factors=2, rotation='promax', method='uls', is_corr_matrix=True)
    fa.fit(np.array(tet_mat))
    L = fa.loadings_
    Phi = fa.phi_ if hasattr(fa, 'phi_') else np.eye(2)
    uniquenesses = fa.get_uniquenesses()
    factor_assignment = np.argmax(np.abs(L), axis=1)
    omegas = []
    for i in range(2):
        idx = (factor_assignment == i)
        if np.sum(idx) == 0:
            omegas.append(0.0)
            continue
        l_i = L[idx, i]
        u_i = uniquenesses[idx]
        omega = (np.sum(l_i)**2) / ((np.sum(l_i)**2) + np.sum(u_i))
        omegas.append(omega)
    Z = (df[qhd_cols] - df[qhd_cols].mean()) / df[qhd_cols].std()
    Z = Z.fillna(0)
    R_inv = np.linalg.inv(tet_mat)
    S = L @ Phi
    F = Z.values @ R_inv @ S
    sum_pos_f0 = np.sum(L[:6, 0])
    sum_pos_f1 = np.sum(L[:6, 1])
    if sum_pos_f0 > sum_pos_f1:
        pos_idx, neg_idx = 0, 1
    else:
        pos_idx, neg_idx = 1, 0
    df['Positive_Affect'] = F[:, pos_idx]
    df['Negative_Affect'] = F[:, neg_idx]
    loadings_df = pd.DataFrame(L, index=qhd_cols, columns=['Factor 1', 'Factor 2'])
    loadings_df.columns = ['Positive Affect' if i == pos_idx else 'Negative Affect' for i in range(2)]
    print('\n--- Factor Loadings ---')
    print(loadings_df.round(3))
    print('\n--- McDonald\'s Omega ---')
    print('Positive Affect: ' + str(round(omegas[pos_idx], 3)))
    print('Negative Affect: ' + str(round(omegas[neg_idx], 3)))
    output_path = os.path.join('data', 'IKI_Cleaned_Features_Augmented.csv')
    df.to_csv(output_path, index=False)
    print('\nAugmented dataset saved to ' + output_path)