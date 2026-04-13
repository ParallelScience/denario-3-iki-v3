# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
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
    qkb14_col = get_full_col('QKB_1_4', df)
    qkb111_col = get_full_col('QKB_1_11', df)
    qkb211_col = get_full_col('QKB_2_11', df)
    inter_14_111 = 'QKB_1_4_x_QKB_1_11'
    df_model[inter_14_111] = df_model[qkb14_col] * df_model[qkb111_col]
    interaction_cols.append(inter_14_111)
    X_cols = feature_cols + interaction_cols
    X = df_model[X_cols]
    X = sm.add_constant(X)
    y = df_model['Target']
    model = sm.MNLogit(y, X)
    result = model.fit(maxiter=1000, disp=False)
    X_mean = X.mean().copy()
    scenarios = [
        {"name": "No Training\nNo Involvement", "QKB_1_4": -2, "QKB_1_11": -2},
        {"name": "Training Only", "QKB_1_4": 2, "QKB_1_11": -2},
        {"name": "Involvement Only", "QKB_1_4": -2, "QKB_1_11": 2},
        {"name": "Both", "QKB_1_4": 2, "QKB_1_11": 2}
    ]
    probs = []
    ci_lower = []
    ci_upper = []
    np.random.seed(42)
    n_samples = 10000
    cov = result.cov_params()
    mean_params = result.params.values.flatten(order='F')
    params_sampled_flat = np.random.multivariate_normal(mean_params, cov, n_samples)
    n_features = X.shape[1]
    n_classes_minus_1 = result.params.shape[1]
    params_sampled = np.zeros((n_samples, n_features, n_classes_minus_1))
    for i in range(n_samples):
        params_sampled[i] = params_sampled_flat[i].reshape((n_features, n_classes_minus_1), order='F')
    for sc in scenarios:
        x_sc = X_mean.copy()
        x_sc[qkb14_col] = sc["QKB_1_4"]
        x_sc[qkb111_col] = sc["QKB_1_11"]
        if qkb211_col:
            x_sc[qkb211_col] = sc["QKB_1_11"]
        for qkb in qkb_list:
            for qf in qf_list:
                qkb_col = get_full_col(qkb, df)
                qf_col = get_full_col(qf, df)
                if qkb_col and qf_col:
                    inter_name = qkb + '_x_' + qf
                    x_sc[inter_name] = x_sc[qkb_col] * x_sc[qf_col]
        x_sc[inter_14_111] = x_sc[qkb14_col] * x_sc[qkb111_col]
        x_val = x_sc.values
        eta = np.dot(x_val, result.params.values)
        exp_eta = np.exp(eta)
        p1 = exp_eta[0] / (1 + np.sum(exp_eta))
        probs.append(p1)
        eta_sampled = np.einsum('sfc,f->sc', params_sampled, x_val)
        exp_eta_sampled = np.exp(eta_sampled)
        p1_sampled = exp_eta_sampled[:, 0] / (1 + np.sum(exp_eta_sampled, axis=1))
        ci_lower.append(np.percentile(p1_sampled, 2.5))
        ci_upper.append(np.percentile(p1_sampled, 97.5))
    plt.rcParams['text.usetex'] = False
    labels = ['No Training', 'Regular Training']
    no_involvement_means = [probs[0], probs[1]]
    no_involvement_err = [[probs[0] - ci_lower[0], probs[1] - ci_lower[1]], [ci_upper[0] - probs[0], ci_upper[1] - probs[1]]]
    involvement_means = [probs[2], probs[3]]
    involvement_err = [[probs[2] - ci_lower[2], probs[3] - ci_lower[3]], [ci_upper[2] - probs[2], ci_upper[3] - probs[3]]]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, no_involvement_means, width, yerr=no_involvement_err, capsize=6, label='No Involvement', color='tab:red', alpha=0.8, edgecolor='black')
    rects2 = ax.bar(x + width/2, involvement_means, width, yerr=involvement_err, capsize=6, label='Employee Involvement', color='tab:green', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Predicted Probability of "Resiliently Optimistic"', fontsize=12)
    ax.set_title('Policy Simulation: Training and Involvement Effects', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Development Process', fontsize=11, title_fontsize=12)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(str(round(height, 2)), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    autolabel(rects1)
    autolabel(rects2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'policy_simulation_probabilities_' + str(timestamp) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)