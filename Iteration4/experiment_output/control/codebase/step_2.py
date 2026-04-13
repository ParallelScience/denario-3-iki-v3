# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import re
from factor_analyzer import FactorAnalyzer

def compute_tetrachoric_corr(data):
    cols = data.columns
    n = len(cols)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            crosstab = pd.crosstab(data[cols[i]], data[cols[j]])
            if crosstab.shape == (2, 2):
                p00 = crosstab.iloc[0, 0] + 0.5
                p01 = crosstab.iloc[0, 1] + 0.5
                p10 = crosstab.iloc[1, 0] + 0.5
                p11 = crosstab.iloc[1, 1] + 0.5
                alpha = (p11 * p00) / (p10 * p01)
                r = np.cos(np.pi / (1 + np.sqrt(alpha)))
            else:
                r = data[cols[i]].corr(data[cols[j]])
            corr[i, j] = r
            corr[j, i] = r
    eigvals, eigvecs = np.linalg.eigh(corr)
    if np.any(eigvals < 0):
        eigvals[eigvals < 0] = 1e-8
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return pd.DataFrame(corr, index=cols, columns=cols)

def extract_num(col):
    match = re.search(r'QHD_(\d+)', col)
    return int(match.group(1)) if match else 0

def calculate_omega(loadings_series, communalities_series):
    sum_loadings = loadings_series.sum()
    sum_uniqueness = (1 - communalities_series).sum()
    omega = (sum_loadings**2) / ((sum_loadings**2) + sum_uniqueness)
    return omega

if __name__ == '__main__':
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'processed_data.csv')
    df = pd.read_csv(input_path, low_memory=False)
    qhd_cols = [c for c in df.columns if c.startswith('QHD_')]
    qhd_cols_sorted = sorted(qhd_cols, key=extract_num)
    print('Computing tetrachoric correlation matrix for ' + str(len(qhd_cols_sorted)) + ' items...')
    tet_corr = compute_tetrachoric_corr(df[qhd_cols_sorted])
    print('Performing EFA with oblimin rotation (2 factors)...')
    fa = FactorAnalyzer(n_factors=2, rotation='oblimin', is_corr_matrix=True)
    fa.fit(tet_corr)
    loadings = pd.DataFrame(fa.loadings_, index=qhd_cols_sorted, columns=['Factor_1', 'Factor_2'])
    communalities = pd.Series(fa.get_communalities(), index=qhd_cols_sorted, name='Communality')
    variance = fa.get_factor_variance()
    var_df = pd.DataFrame({'SS Loadings': variance[0], 'Proportion Var': variance[1], 'Cumulative Var': variance[2]}, index=['Factor_1', 'Factor_2'])
    R_inv = np.linalg.pinv(tet_corr.values)
    Lambda = fa.loadings_
    Phi = fa.phi_ if hasattr(fa, 'phi_') and fa.phi_ is not None else np.eye(2)
    W = R_inv @ Lambda @ Phi
    Z = (df[qhd_cols_sorted] - df[qhd_cols_sorted].mean()) / df[qhd_cols_sorted].std()
    Z = Z.fillna(0)
    scores = Z.values @ W
    scores_df = pd.DataFrame(scores, columns=['Factor_1', 'Factor_2'], index=df.index)
    positive_items = qhd_cols_sorted[:6]
    f1_pos_mean = loadings.loc[positive_items, 'Factor_1'].mean()
    f2_pos_mean = loadings.loc[positive_items, 'Factor_2'].mean()
    if f2_pos_mean > f1_pos_mean:
        df['Positive_Affect'] = scores_df['Factor_2']
        df['Negative_Affect'] = scores_df['Factor_1']
        loadings.columns = ['Negative_Affect', 'Positive_Affect']
    else:
        df['Positive_Affect'] = scores_df['Factor_1']
        df['Negative_Affect'] = scores_df['Factor_2']
        loadings.columns = ['Positive_Affect', 'Negative_Affect']
    factor_assignment = loadings.abs().idxmax(axis=1)
    pos_items = factor_assignment[factor_assignment == 'Positive_Affect'].index
    neg_items = factor_assignment[factor_assignment == 'Negative_Affect'].index
    omega_pos = calculate_omega(loadings.loc[pos_items, 'Positive_Affect'], communalities.loc[pos_items])
    omega_neg = calculate_omega(loadings.loc[neg_items, 'Negative_Affect'], communalities.loc[neg_items])
    print('\n--- Factor Loadings ---')
    print(loadings.round(3).to_string())
    print('\n--- Communalities ---')
    print(communalities.round(3).to_string())
    print('\n--- Variance Explained ---')
    print(var_df.round(3).to_string())
    print('\n--- Reliability (McDonald\'s Omega) ---')
    print('Positive Affect Omega: ' + str(round(omega_pos, 3)))
    print('Negative Affect Omega: ' + str(round(omega_neg, 3)))
    df.to_csv(input_path, index=False)
    print('\nUpdated dataset with factor scores saved to ' + input_path)
    loadings_path = os.path.join(data_dir, 'qhd_factor_loadings.csv')
    loadings.to_csv(loadings_path)
    print('Factor loadings saved to ' + loadings_path)