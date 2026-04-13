# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def get_full_col(prefix, df):
    for c in df.columns:
        if c.startswith(prefix + ':') or c == prefix:
            return c
    return None

if __name__ == '__main__':
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'cleaned_dataset_step3.csv')
    df = pd.read_csv(data_path, low_memory=False)
    main_effects = ['QKB_2_8', 'QKB_2_11', 'QKB_2_9', 'QKB_1_4', 'QKB_1_11', 'QF_4', 'QF_7', 'QF_6', 'QF_3', 'QF_9', 'QKC_1', 'QKC_2', 'QKC_3', 'QKC_4', 'QGO_5']
    feature_cols = []
    for me in main_effects:
        col = get_full_col(me, df)
        if col:
            feature_cols.append(col)
        else:
            print('Warning: ' + me + ' not found in columns.')
    cols_to_keep = feature_cols + ['LCA_Class']
    df_model = df[cols_to_keep].copy()
    df_model = df_model.dropna(subset=['LCA_Class'])
    imputer = SimpleImputer(strategy='median')
    df_model[feature_cols] = imputer.fit_transform(df_model[feature_cols])
    class_mapping = {3: 0, 1: 1, 2: 2}
    df_model['Target'] = df_model['LCA_Class'].map(class_mapping)
    interaction_cols = []
    qkb_list = ['QKB_2_8', 'QKB_2_11', 'QKB_2_9', 'QKB_1_4', 'QKB_1_11']
    qf_list = ['QF_4', 'QF_7']
    for qkb in qkb_list:
        for qf in qf_list:
            qkb_col = get_full_col(qkb, df)
            qf_col = get_full_col(qf, df)
            if qkb_col and qf_col:
                inter_name = qkb + '_x_' + qf
                df_model[inter_name] = df_model[qkb_col] * df_model[qf_col]
                interaction_cols.append(inter_name)
    X_cols = feature_cols + interaction_cols
    X = df_model[X_cols]
    X = sm.add_constant(X)
    y = df_model['Target']
    model = sm.MNLogit(y, X)
    result = model.fit(maxiter=1000, disp=False)
    class_1_col = 1
    b = result.params[class_1_col]
    se = result.bse[class_1_col]
    p = result.pvalues[class_1_col]
    lower = b - 1.96 * se
    upper = b + 1.96 * se
    inter_pvals = p[interaction_cols]
    reject, pvals_corrected, _, _ = multipletests(inter_pvals, alpha=0.05, method='fdr_bh')
    odds_ratios = np.exp(b)
    lower_or = np.exp(lower)
    upper_or = np.exp(upper)
    plot_features = [c for c in X_cols if c != 'const']
    or_vals = odds_ratios[plot_features]
    or_lower = lower_or[plot_features]
    or_upper = upper_or[plot_features]
    errors_lower = or_vals - or_lower
    errors_upper = or_upper - or_vals
    sort_idx = np.argsort(or_vals.values)
    sorted_features = np.array(plot_features)[sort_idx]
    sorted_or = or_vals.values[sort_idx]
    sorted_err_lower = errors_lower.values[sort_idx]
    sorted_err_upper = errors_upper.values[sort_idx]
    def short_name(col):
        if '_x_' in col:
            return col
        return col.split(':')[0]
    short_features = [short_name(c) for c in sorted_features]
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 12))
    plt.errorbar(sorted_or, range(len(short_features)), xerr=[sorted_err_lower, sorted_err_upper], fmt='o', color='tab:blue', ecolor='tab:gray', capsize=4, elinewidth=2, markersize=6)
    plt.axvline(1, color='red', linestyle='--', linewidth=1.5)
    plt.yticks(range(len(short_features)), short_features, fontsize=10)
    plt.xlabel('Odds Ratio (log scale)', fontsize=12)
    plt.xscale('log')
    plt.title('Odds Ratios for Resiliently Optimistic vs Anxiously Declining', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'forest_plot_odds_ratios_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Forest plot saved to ' + plot_filename)