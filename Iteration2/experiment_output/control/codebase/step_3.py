# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import math
try:
    from factor_analyzer import FactorAnalyzer
except ImportError:
    sys.exit(1)
def tetrachoric_corr(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    x_val = x[valid]
    y_val = y[valid]
    if len(np.unique(x_val)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0
    crosstab = pd.crosstab(x_val, y_val)
    if crosstab.shape != (2, 2):
        corr = np.corrcoef(x_val, y_val)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    a = crosstab.iloc[0, 0]
    b = crosstab.iloc[0, 1]
    c = crosstab.iloc[1, 0]
    d = crosstab.iloc[1, 1]
    if a == 0 or b == 0 or c == 0 or d == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    ratio = (a * d) / (b * c)
    r = math.cos(math.pi / (1 + math.sqrt(ratio)))
    return r
def get_tetrachoric_matrix(df):
    cols = df.columns
    n = len(cols)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = tetrachoric_corr(df[cols[i]], df[cols[j]])
            mat[i, j] = r
            mat[j, i] = r
    return pd.DataFrame(mat, index=cols, columns=cols)
def main():
    data_dir = "data/"
    file_path = os.path.join(data_dir, "processed_data.csv")
    df = pd.read_csv(file_path, low_memory=False)
    qhd_cols = [c for c in df.columns if c.startswith('QHD_')]
    current_items = qhd_cols.copy()
    iteration = 1
    while True:
        df_efa = df[current_items].dropna()
        tetra_corr = get_tetrachoric_matrix(df_efa)
        fa = FactorAnalyzer(n_factors=2, rotation='oblimin', is_corr_matrix=True, method='minres')
        fa.fit(tetra_corr)
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        loadings_df = pd.DataFrame(loadings, index=current_items, columns=['Factor1', 'Factor2'])
        loadings_df['Communality'] = communalities
        drop_items = []
        for item in current_items:
            comm = loadings_df.loc[item, 'Communality']
            l1 = abs(loadings_df.loc[item, 'Factor1'])
            l2 = abs(loadings_df.loc[item, 'Factor2'])
            if comm < 0.3:
                drop_items.append(item)
            elif l1 > 0.3 and l2 > 0.3:
                drop_items.append(item)
        if not drop_items:
            break
        current_items = [item for item in current_items if item not in drop_items]
        if len(current_items) < 4:
            break
        iteration += 1
    factor_corr = fa.phi_[0, 1] if hasattr(fa, 'phi_') and fa.phi_ is not None else 0.0
    R_inv = np.linalg.inv(tetra_corr.values)
    W = R_inv @ loadings
    std = df[current_items].std()
    std = std.replace(0, 1.0)
    Z = (df[current_items] - df[current_items].mean()) / std
    Z = Z.fillna(0)
    scores = Z.values @ W
    ref_item = qhd_cols[0] if qhd_cols[0] in current_items else current_items[0]
    corr_f1 = np.corrcoef(scores[:, 0], df[ref_item].fillna(0))[0, 1]
    corr_f2 = np.corrcoef(scores[:, 1], df[ref_item].fillna(0))[0, 1]
    pos_idx, neg_idx = (0, 1) if corr_f1 > corr_f2 else (1, 0)
    df['Positive_Affect'] = scores[:, pos_idx]
    df['Negative_Affect'] = scores[:, neg_idx]
    if abs(factor_corr) > 0.5:
        df['AI_Sentiment'] = df['Positive_Affect'] - df['Negative_Affect']
    df.to_csv(file_path, index=False)
if __name__ == '__main__':
    main()