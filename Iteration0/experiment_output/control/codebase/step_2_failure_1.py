# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

def tetrachoric_corr(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    c11 = np.sum((x == 1) & (y == 1))
    c10 = np.sum((x == 1) & (y == 0))
    c01 = np.sum((x == 0) & (y == 1))
    c00 = np.sum((x == 0) & (y == 0))
    if c11 == 0 or c10 == 0 or c01 == 0 or c00 == 0:
        c11 += 0.5
        c10 += 0.5
        c01 += 0.5
        c00 += 0.5
    omega = (c11 * c00) / (c10 * c01)
    rtet = np.cos(np.pi / (1 + np.sqrt(omega)))
    return rtet

def get_tetrachoric_corr_matrix(df):
    cols = df.columns
    n = len(cols)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            r = tetrachoric_corr(df.iloc[:, i].values, df.iloc[:, j].values)
            corr[i, j] = r
            corr[j, i] = r
    return pd.DataFrame(corr, index=cols, columns=cols)

def cronbach_alpha(df):
    df = df.dropna()
    if len(df) < 2 or df.shape[1] < 2:
        return 0.0
    item_vars = df.var(axis=0, ddof=1)
    t_var = df.sum(axis=1).var(ddof=1)
    k = df.shape[1]
    if t_var == 0:
        return 0.0
    return (k / (k - 1)) * (1 - item_vars.sum() / t_var)

if __name__ == '__main__':
    raw_path = 'data/data_raw_encoded.csv'
    dummy_path = 'data/data_dummy_encoded.csv'
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_dummy = pd.read_csv(dummy_path, low_memory=False)
    qhd_cols = [c for c in df_raw.columns if c.startswith('QHD_') and not c.endswith('_Not_Sure')]
    qhd_data = df_raw[qhd_cols].copy()
    qhd_data_clean = qhd_data.dropna()
    try:
        kmo_all, kmo_model = calculate_kmo(qhd_data_clean)
        print('KMO Statistic: ' + str(round(kmo_model, 4)))
    except Exception as e:
        print('KMO calculation failed: ' + str(e))
    try:
        chi_square_value, p_value = calculate_bartlett_sphericity(qhd_data_clean)
        print('Bartlett\'s Test p-value: ' + str(p_value))
    except Exception as e:
        print('Bartlett\'s Test calculation failed: ' + str(e))
    tet_corr = get_tetrachoric_corr_matrix(qhd_data)
    eigenvalues, _ = np.linalg.eigh(tet_corr.values)
    eigenvalues = np.sort(eigenvalues)[::-1]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
    plt.title('Scree Plot of Eigenvalues (Tetrachoric Correlation)')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'data/scree_plot_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('Scree plot saved to ' + plot_filename)
    n_factors = 2
    fa = FactorAnalyzer(n_factors=n_factors, rotation='promax', is_corr_matrix=True)
    fa.fit(tet_corr.values)
    loadings = pd.DataFrame(fa.loadings_, index=qhd_cols, columns=['Factor_' + str(i+1) for i in range(n_factors)])
    print('\nFactor Loadings:')
    print(loadings.round(3).to_string())
    factor_items = {'Factor_' + str(i+1): [] for i in range(n_factors)}
    for item in qhd_cols:
        best_factor = loadings.loc[item].abs().idxmax()
        factor_items[best_factor].append(item)
    for factor, items in factor_items.items():
        if len(items) > 0:
            alpha = cronbach_alpha(qhd_data[items])
            print(factor + ' (' + str(len(items)) + ' items): ' + str(round(alpha, 4)))
    Z = (qhd_data - qhd_data.mean()) / qhd_data.std()
    Z = Z.fillna(0)
    R_inv = np.linalg.pinv(tet_corr.values)
    if hasattr(fa, 'phi_') and fa.phi_ is not None:
        S = fa.loadings_.dot(fa.phi_)
    else:
        S = fa.loadings_
    W = R_inv.dot(S)
    scores = Z.values.dot(W)
    df_raw['Affective_Disposition_Factor_1'] = scores[:, 0]
    df_raw['Affective_Disposition_Factor_2'] = scores[:, 1]
    df_dummy['Affective_Disposition_Factor_1'] = scores[:, 0]
    df_dummy['Affective_Disposition_Factor_2'] = scores[:, 1]
    df_raw.to_csv(raw_path, index=False)
    df_dummy.to_csv(dummy_path, index=False)
    print('Factor scores successfully saved to ' + raw_path + ' and ' + dummy_path)