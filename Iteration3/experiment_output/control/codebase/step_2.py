# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np

def tetrachoric_corr(x, y):
    crosstab = pd.crosstab(x, y)
    if crosstab.shape != (2, 2):
        corr = np.corrcoef(x, y)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    crosstab = crosstab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    a = crosstab.loc[1, 1]
    b = crosstab.loc[1, 0]
    c = crosstab.loc[0, 1]
    d = crosstab.loc[0, 0]
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ratio = (a * d) / (b * c)
    rtet = np.cos(np.pi / (1 + np.sqrt(ratio)))
    return rtet

def tetrachoric_corr_matrix(df):
    cols = df.columns
    n = len(cols)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = tetrachoric_corr(df[cols[i]], df[cols[j]])
            corr[i, j] = r
            corr[j, i] = r
    return pd.DataFrame(corr, index=cols, columns=cols)

if __name__ == '__main__':
    from factor_analyzer import FactorAnalyzer
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_data_step1.csv')
    df = pd.read_csv(data_path)
    qhd_cols = [c for c in df.columns if c.startswith('QHD_')]
    print("Computing tetrachoric correlation matrix...")
    tet_corr = tetrachoric_corr_matrix(df[qhd_cols])
    print("Performing EFA...")
    fa = FactorAnalyzer(n_factors=2, rotation='promax', is_corr_matrix=True)
    fa.fit(tet_corr)
    loadings = pd.DataFrame(fa.loadings_, index=qhd_cols, columns=['Factor1', 'Factor2'])
    print("\nFactor Loadings:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(loadings)
    if hasattr(fa, 'get_factor_variance'):
        variance = fa.get_factor_variance()
    else:
        ss_loadings = np.sum(fa.loadings_**2, axis=0)
        prop_var = ss_loadings / len(qhd_cols)
        cum_var = np.cumsum(prop_var)
        variance = (ss_loadings, prop_var, cum_var)
    var_df = pd.DataFrame(variance, index=['SS Loadings', 'Proportion Var', 'Cumulative Var'], columns=['Factor1', 'Factor2'])
    print("\nVariance Explained:")
    print(var_df)
    factor1_items = loadings[loadings['Factor1'].abs() > loadings['Factor2'].abs()].index.tolist()
    factor2_items = loadings[loadings['Factor2'].abs() > loadings['Factor1'].abs()].index.tolist()
    opt_col = [c for c in qhd_cols if 'Optimistic' in c][0]
    if loadings.loc[opt_col, 'Factor1'] > loadings.loc[opt_col, 'Factor2']:
        pos_factor = 'Factor1'
        neg_factor = 'Factor2'
        pos_items = factor1_items
        neg_items = factor2_items
    else:
        pos_factor = 'Factor2'
        neg_factor = 'Factor1'
        pos_items = factor2_items
        neg_items = factor1_items
    print("\nPositive Affect items (" + str(len(pos_items)) + "):")
    print([c.split(':')[1].split('-')[0].strip() for c in pos_items])
    print("Negative Affect items (" + str(len(neg_items)) + "):")
    print([c.split(':')[1].split('-')[0].strip() for c in neg_items])
    def calculate_omega(loadings_series, uniquenesses_series):
        sum_loadings_sq = loadings_series.sum() ** 2
        sum_uniquenesses = uniquenesses_series.sum()
        return sum_loadings_sq / (sum_loadings_sq + sum_uniquenesses)
    uniquenesses_vals = fa.get_uniquenesses()
    uniquenesses = pd.Series(uniquenesses_vals, index=qhd_cols)
    omega_pos = calculate_omega(loadings.loc[pos_items, pos_factor].abs(), uniquenesses.loc[pos_items])
    omega_neg = calculate_omega(loadings.loc[neg_items, neg_factor].abs(), uniquenesses.loc[neg_items])
    print("\nMcDonald's Omega for Positive Affect: " + str(round(omega_pos, 3)))
    print("McDonald's Omega for Negative Affect: " + str(round(omega_neg, 3)))
    S = fa.loadings_ @ fa.phi_ if hasattr(fa, 'phi_') and fa.phi_ is not None else fa.loadings_
    W = np.linalg.pinv(tet_corr.values) @ S
    Z = (df[qhd_cols] - df[qhd_cols].mean()) / df[qhd_cols].std()
    Z = Z.fillna(0)
    factor_scores = Z.values @ W
    if pos_factor == 'Factor1':
        pos_scores = factor_scores[:, 0]
        neg_scores = factor_scores[:, 1]
    else:
        pos_scores = factor_scores[:, 1]
        neg_scores = factor_scores[:, 0]
    df['Positive_Affect'] = (pos_scores - np.nanmean(pos_scores)) / np.nanstd(pos_scores)
    df['Negative_Affect'] = (neg_scores - np.nanmean(neg_scores)) / np.nanstd(neg_scores)
    print("\nSummary of Positive Affect Score:")
    print(df['Positive_Affect'].describe())
    print("\nSummary of Negative Affect Score:")
    print(df['Negative_Affect'].describe())
    output_path = os.path.join(data_dir, 'processed_data_step2.csv')
    df.to_csv(output_path, index=False)
    print("\nProcessed data with factor scores saved to " + output_path)