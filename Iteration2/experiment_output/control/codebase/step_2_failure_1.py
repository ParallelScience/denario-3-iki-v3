# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import re
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer

def tetrachoric_corr(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.nan
    c00 = np.sum((x == 0) & (y == 0)) + 0.5
    c01 = np.sum((x == 0) & (y == 1)) + 0.5
    c10 = np.sum((x == 1) & (y == 0)) + 0.5
    c11 = np.sum((x == 1) & (y == 1)) + 0.5
    ratio = (c00 * c11) / (c01 * c10)
    r = np.cos(np.pi / (1 + np.sqrt(ratio)))
    return r

def get_tetrachoric_matrix(df_items):
    cols = df_items.columns
    n = len(cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = tetrachoric_corr(df_items.iloc[:, i].values, df_items.iloc[:, j].values)
            mat[i, j] = r
            mat[j, i] = r
    return pd.DataFrame(mat, index=cols, columns=cols)

def get_qhd_num(col):
    match = re.search(r'QHD_(\d+)', col)
    return int(match.group(1)) if match else 999

def main():
    data_dir = "data/"
    input_path = os.path.join(data_dir, "processed_data.csv")
    df = pd.read_csv(input_path, low_memory=False)
    qhd_cols = sorted([c for c in df.columns if c.startswith('QHD_')], key=get_qhd_num)
    tetra_corr = get_tetrachoric_matrix(df[qhd_cols])
    fa = FactorAnalyzer(n_factors=2, rotation='promax', method='minres')
    fa.fit(tetra_corr)
    loadings = pd.DataFrame(fa.loadings_, index=qhd_cols, columns=['Factor_1', 'Factor_2'])
    communalities = pd.Series(fa.get_communalities(), index=qhd_cols)
    pos_keywords = ['optimistic', 'empowered', 'motivated', 'excited', 'curious', 'confident']
    neg_keywords = ['indifferent', 'uncertain', 'cautious', 'skeptical', 'worried', 'resistant', 'confused', 'overwhelmed', 'threatened', 'anxious']
    pos_cols = []
    neg_cols = []
    for c in qhd_cols:
        c_lower = c.lower()
        if any(k in c_lower for k in pos_keywords):
            pos_cols.append(c)
        elif any(k in c_lower for k in neg_keywords):
            neg_cols.append(c)
    if not pos_cols or not neg_cols:
        pos_cols = qhd_cols[:6]
        neg_cols = qhd_cols[6:]
    f1_pos_mean = loadings.loc[pos_cols, 'Factor_1'].mean()
    f2_pos_mean = loadings.loc[pos_cols, 'Factor_2'].mean()
    if f1_pos_mean > f2_pos_mean:
        pos_factor = 'Factor_1'
        neg_factor = 'Factor_2'
    else:
        pos_factor = 'Factor_2'
        neg_factor = 'Factor_1'
    if loadings.loc[pos_cols, pos_factor].mean() < 0:
        loadings[pos_factor] = -loadings[pos_factor]
    if loadings.loc[neg_cols, neg_factor].mean() < 0:
        loadings[neg_factor] = -loadings[neg_factor]
    loadings = loadings.rename(columns={pos_factor: 'Positive_Affect', neg_factor: 'Negative_Affect'})
    R_inv = np.linalg.pinv(tetra_corr.values)
    W = R_inv @ loadings[['Positive_Affect', 'Negative_Affect']].values
    std = df[qhd_cols].std()
    std[std == 0] = 1e-6
    Z = (df[qhd_cols] - df[qhd_cols].mean()) / std
    Z = Z.fillna(0)
    factor_scores = Z.values @ W
    df['Positive_Affect'] = factor_scores[:, 0]
    df['Negative_Affect'] = factor_scores[:, 1]
    inter_corr = df['Positive_Affect'].corr(df['Negative_Affect'])
    df['AI_Sentiment'] = df['Positive_Affect'] - df['Negative_Affect']
    df.to_csv(input_path, index=False)

if __name__ == '__main__':
    main()