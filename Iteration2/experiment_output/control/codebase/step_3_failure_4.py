# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import sklearn.utils.validation

def patched_check_array(array, *args, **kwargs):
    kwargs.pop('force_all_finite', None)
    return original_check_array(array, *args, **kwargs)

original_check_array = sklearn.utils.validation.check_array
sklearn.utils.validation.check_array = patched_check_array

def tetrachoric_corr(x, y):
    c00 = np.sum((x == 0) & (y == 0)) + 0.5
    c01 = np.sum((x == 0) & (y == 1)) + 0.5
    c10 = np.sum((x == 1) & (y == 0)) + 0.5
    c11 = np.sum((x == 1) & (y == 1)) + 0.5
    ratio = (c00 * c11) / (c01 * c10)
    r = np.cos(np.pi / (1 + np.sqrt(ratio)))
    return r

def compute_tetrachoric_matrix(df):
    cols = df.columns
    n = len(cols)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            r = tetrachoric_corr(df[cols[i]], df[cols[j]])
            corr[i, j] = r
            corr[j, i] = r
    return pd.DataFrame(corr, index=cols, columns=cols)

def main():
    try:
        from factor_analyzer import FactorAnalyzer
    except ImportError as e:
        print('ModuleNotFoundError: ' + str(e))
        sys.exit(1)
    data_dir = 'data/'
    file_path = os.path.join(data_dir, 'processed_data.csv')
    df = pd.read_csv(file_path, low_memory=False)
    qhd_cols = [c for c in df.columns if c.startswith('QHD_')]
    df_qhd = df[qhd_cols].dropna()
    tet_corr = compute_tetrachoric_matrix(df_qhd)
    fa = FactorAnalyzer(n_factors=2, rotation='oblimin', is_corr_matrix=True)
    fa.fit(tet_corr)
    loadings = pd.DataFrame(fa.loadings_, index=qhd_cols, columns=['Factor_1', 'Factor_2'])
    communalities = pd.Series(fa.get_communalities(), index=qhd_cols)
    first_6 = qhd_cols[:6] if len(qhd_cols) >= 6 else qhd_cols
    last_10 = qhd_cols[6:] if len(qhd_cols) > 6 else qhd_cols
    sum_abs_f1_first6 = loadings.loc[first_6, 'Factor_1'].abs().sum()
    sum_abs_f2_first6 = loadings.loc[first_6, 'Factor_2'].abs().sum()
    if sum_abs_f1_first6 > sum_abs_f2_first6:
        pos_factor = 'Factor_1'
        neg_factor = 'Factor_2'
    else:
        pos_factor = 'Factor_2'
        neg_factor = 'Factor_1'
    invert_pos = loadings.loc[first_6, pos_factor].mean() < 0
    invert_neg = loadings.loc[last_10, neg_factor].mean() < 0
    if invert_pos:
        loadings[pos_factor] = -loadings[pos_factor]
    if invert_neg:
        loadings[neg_factor] = -loadings[neg_factor]
    loadings = loadings.rename(columns={pos_factor: 'Positive_Affect', neg_factor: 'Negative_Affect'})
    Z = (df_qhd - df_qhd.mean()) / (df_qhd.std() + 1e-8)
    R_inv = np.linalg.pinv(tet_corr.values)
    L = fa.loadings_
    Phi = fa.phi_ if hasattr(fa, 'phi_') and fa.phi_ is not None else np.eye(2)
    S = L @ Phi
    W = R_inv @ S
    scores = Z.values @ W
    scores_df = pd.DataFrame(scores, index=df_qhd.index, columns=['Factor_1', 'Factor_2'])
    if invert_pos:
        scores_df[pos_factor] = -scores_df[pos_factor]
    if invert_neg:
        scores_df[neg_factor] = -scores_df[neg_factor]
    scores_df = scores_df.rename(columns={pos_factor: 'Positive_Affect', neg_factor: 'Negative_Affect'})
    df['Positive_Affect'] = np.nan
    df['Negative_Affect'] = np.nan
    df.loc[scores_df.index, 'Positive_Affect'] = scores_df['Positive_Affect']
    df.loc[scores_df.index, 'Negative_Affect'] = scores_df['Negative_Affect']
    df.to_csv(file_path, index=False)

if __name__ == '__main__':
    main()