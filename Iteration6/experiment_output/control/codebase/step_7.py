# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from scipy.special import logsumexp

def get_full_col(prefix, df):
    for c in df.columns:
        if c.startswith(prefix + ':') or c == prefix:
            return c
    return None

def run_lca(data, n_classes, n_init=10, max_iter=1000, tol=1e-7):
    N, M = data.shape
    C_max = int(np.max(data) + 1)
    one_hots = [np.eye(C_max)[data[:, m]] for m in range(M)]
    best_ll = -np.inf
    best_model = None
    for init in range(n_init):
        pi = np.random.dirichlet(np.ones(n_classes))
        theta = np.random.dirichlet(np.ones(C_max), size=(n_classes, M))
        ll_old = -np.inf
        for iteration in range(max_iter):
            log_prob_y_given_c = np.zeros((N, n_classes))
            for m in range(M):
                log_prob_y_given_c += np.log(theta[:, m, data[:, m]].T + 1e-15)
            log_joint = np.log(pi + 1e-15) + log_prob_y_given_c
            log_marginal = logsumexp(log_joint, axis=1)
            ll = np.sum(log_marginal)
            gamma = np.exp(log_joint - log_marginal[:, np.newaxis])
            N_c = np.sum(gamma, axis=0)
            pi = N_c / N
            for m in range(M):
                counts = (one_hots[m].T @ gamma).T
                theta[:, m, :] = (counts + 1e-6) / (N_c[:, np.newaxis] + C_max * 1e-6)
            if np.abs(ll - ll_old) < tol:
                break
            ll_old = ll
        if ll > best_ll:
            best_ll = ll
            best_model = {'pi': pi, 'theta': theta, 'gamma': gamma, 'll': ll, 'n_iter': iteration}
    return best_model

def get_lca_classes(y1, y2, k=3):
    data = np.column_stack((y1, y2))
    model = run_lca(data, k, n_init=10, max_iter=1000)
    C_max = int(np.max(data) + 1)
    if C_max > 4:
        pos_prob = np.sum(model['theta'][:, :, 3:5], axis=(1, 2))
    else:
        pos_prob = np.sum(model['theta'][:, :, 3:5], axis=(1, 2))
    sort_idx = np.argsort(-pos_prob)
    gamma_sorted = model['gamma'][:, sort_idx]
    classes = np.argmax(gamma_sorted, axis=1) + 1
    return classes

def fit_mnlogit(df_subset, target_col, feature_cols, qkb_list, qf_list, df_full):
    X_df = df_subset[feature_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_df[feature_cols] = imputer.fit_transform(X_df[feature_cols])
    for qkb in qkb_list:
        for qf in qf_list:
            qkb_col = get_full_col(qkb, df_full)
            qf_col = get_full_col(qf, df_full)
            if qkb_col in X_df.columns and qf_col in X_df.columns:
                inter_name = qkb + '_x_' + qf
                X_df[inter_name] = X_df[qkb_col] * X_df[qf_col]
    qkb14_col = get_full_col('QKB_1_4', df_full)
    qkb111_col = get_full_col('QKB_1_11', df_full)
    if qkb14_col in X_df.columns and qkb111_col in X_df.columns:
        inter_14_111 = 'QKB_1_4_x_QKB_1_11'
        X_df[inter_14_111] = X_df[qkb14_col] * X_df[qkb111_col]
    X = sm.add_constant(X_df)
    y = df_subset[target_col]
    model = sm.MNLogit(y, X)
    result = model.fit(maxiter=1000, disp=False)
    return result, X

if __name__ == '__main__':
    np.random.seed(42)
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'cleaned_dataset_step3.csv')
    df = pd.read_csv(data_path, low_memory=False)
    main_effects = ['QKB_2_8', 'QKB_2_11', 'QKB_2_9', 'QKB_1_4', 'QKB_1_11', 'QF_4', 'QF_7', 'QF_6', 'QF_3', 'QF_9', 'QKC_1', 'QKC_2', 'QKC_3', 'QKC_4', 'QGO_5']
    feature_cols = []
    for me in main_effects:
        col = get_full_col(me, df)
        if col:
            feature_cols.append(col)
    qkb_list = ['QKB_2_8', 'QKB_2_11', 'QKB_2_9', 'QKB_1_4', 'QKB_1_11']
    qf_list = ['QF_4', 'QF_7']
    df_v1 = df.copy()
    df_v1 = df_v1.dropna(subset=['LCA_Class'])
    class_mapping = {3: 0, 1: 1, 2: 2}
    df_v1['Target'] = df_v1['LCA_Class'].map(class_mapping)
    result_v1, X_v1 = fit_mnlogit(df_v1, 'Target', feature_cols, qkb_list, qf_list, df)
    print("Calculating VIF for the full model (Variant 1)...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_v1.columns
    vif_list = []
    X_vals = X_v1.values
    for i in range(X_v1.shape[1]):
        try:
            vif = variance_inflation_factor(X_vals, i)
        except Exception:
            vif = np.nan
        vif_list.append(vif)
    vif_data["VIF"] = vif_list
    print("\nVIF Table (flagging VIF > 5):")
    for idx, row in vif_data.iterrows():
        flag = " *** HIGH VIF ***" if row['VIF'] > 5 and row['Feature'] != 'const' else ""
        feat_name = str(row['Feature'])
        if len(feat_name) > 57:
            feat_name = feat_name[:54] + "..."
        print(feat_name.ljust(60) + str(round(row['VIF'], 4)).rjust(10) + flag)
    qea2_col = next((col for col in df.columns if col.startswith('QEA_2')), None)
    qeb2_col = next((col for col in df.columns if col.startswith('QEB_2')), None)
    valid_idx = df[qea2_col].notna() & df[qeb2_col].notna()
    df_v2 = df[valid_idx].copy()
    mapping = {-2.0: 0, -1.0: 1, 0.0: 2, 1.0: 3, 2.0: 4}
    y1_v2 = df_v2[qea2_col].round(1).map(mapping).astype(int).values
    y2_v2 = df_v2[qeb2_col].round(1).map(mapping).astype(int).values
    print("\nRunning LCA for Variant 2 (excluding 'Not sure')...")
    classes_v2 = get_lca_classes(y1_v2, y2_v2, k=3)
    df_v2['LCA_Class_v2'] = classes_v2
    df_v2['Target'] = df_v2['LCA_Class_v2'].map(class_mapping)
    result_v2, X_v2 = fit_mnlogit(df_v2, 'Target', feature_cols, qkb_list, qf_list, df)
    df_v3 = df.copy()
    mode_qea2 = df_v3[qea2_col].mode()[0]
    mode_qeb2 = df_v3[qeb2_col].mode()[0]
    df_v3[qea2_col] = df_v3[qea2_col].fillna(mode_qea2)
    df_v3[qeb2_col] = df_v3[qeb2_col].fillna(mode_qeb2)
    y1_v3 = df_v3[qea2_col].round(1).map(mapping).astype(int).values
    y2_v3 = df_v3[qeb2_col].round(1).map(mapping).astype(int).values
    print("Running LCA for Variant 3 (recoding 'Not sure' to mode)...")
    classes_v3 = get_lca_classes(y1_v3, y2_v3, k=3)
    df_v3['LCA_Class_v3'] = classes_v3
    df_v3['Target'] = df_v3['LCA_Class_v3'].map(class_mapping)
    result_v3, X_v3 = fit_mnlogit(df_v3, 'Target', feature_cols, qkb_list, qf_list, df)
    print("\n--- Sensitivity Analysis ---")
    print("Variant 1 (Full sample) N = " + str(len(df_v1)))
    print("Variant 2 (Excluding 'Not sure') N = " + str(len(df_v2)))
    print("Variant 3 (Recoding 'Not sure' to mode) N = " + str(len(df_v3)))
    or_v1 = np.exp(result_v1.params[1])
    or_v2 = np.exp(result_v2.params[1])
    or_v3 = np.exp(result_v3.params[1])
    print("\nComparison of Odds Ratios (Resiliently Optimistic vs Anxiously Declining):")
    header = "Feature".ljust(60) + "V1 (Full)".rjust(15) + "V2 (Excl NS)".rjust(15) + "V3 (Mode NS)".rjust(15)
    print(header)
    print("-" * len(header))
    for feat in X_v1.columns:
        if feat == 'const':
            continue
        val1 = str(round(or_v1[feat], 3))
        val2 = str(round(or_v2[feat], 3))
        val3 = str(round(or_v3[feat], 3))
        feat_name = feat
        if len(feat_name) > 57:
            feat_name = feat_name[:54] + "..."
        print(feat_name.ljust(60) + val1.rjust(15) + val2.rjust(15) + val3.rjust(15))