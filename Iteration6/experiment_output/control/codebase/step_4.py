# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time
import textwrap

if __name__ == '__main__':
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'cleaned_dataset_step3.csv')
    df = pd.read_csv(data_path, low_memory=False)
    prefixes = ['QKB_1_', 'QKB_2_', 'QGO_', 'QF_', 'QKC_']
    predictor_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    predictor_cols = [c for c in predictor_cols if 'QF_13' not in c and 'Other' not in c]
    print('Number of predictors identified: ' + str(len(predictor_cols)))
    target_col = 'LCA_Class'
    if target_col not in df.columns:
        raise ValueError('Target column ' + target_col + ' not found in dataset.')
    valid_idx = df[target_col].notna()
    df = df[valid_idx].copy()
    y = df[target_col].astype(int).values
    X_raw = df[predictor_cols].copy()
    for c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print('\nFitting Overall Elastic Net Model (N=' + str(len(df)) + ')...')
    clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.2, multi_class='multinomial', max_iter=5000, random_state=42)
    clf.fit(X_scaled, y)
    if 1 in clf.classes_:
        class1_idx = list(clf.classes_).index(1)
        coef_overall = clf.coef_[class1_idx]
    else:
        raise ValueError('Class 1 not found in the target variable.')
    if np.max(np.abs(coef_overall)) == 0:
        print('C=0.2 resulted in all zero coefficients. Retrying with C=1.0...')
        clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, multi_class='multinomial', max_iter=5000, random_state=42)
        clf.fit(X_scaled, y)
        coef_overall = clf.coef_[class1_idx]
    results = {'Overall': coef_overall}
    def get_stratum_name(val):
        s = str(val)
        if '-' in s:
            parts = s.split('-')
            if parts[0].strip().isdigit() and len(parts) > 1:
                name = parts[1].strip()
            else:
                name = parts[0].strip()
        else:
            name = s.strip()
        if len(name) > 20:
            name = name[:17] + '...'
        return name
    strata_masks = {}
    job_col = next((c for c in df.columns if c.upper() == 'HIDDG'), None)
    if job_col:
        counts = df[job_col].value_counts()
        for val, count in counts.items():
            if count > 100:
                name = get_stratum_name(val)
                key = 'Job: ' + name
                suffix = 1
                while key in strata_masks:
                    key = 'Job: ' + name + '_' + str(suffix)
                    suffix += 1
                strata_masks[key] = (df[job_col] == val)
    else:
        print("Warning: Job profile column (HIDDG) not found.")
    sec_col = next((c for c in df.columns if c.upper() == 'QDB'), None)
    if sec_col:
        counts = df[sec_col].value_counts()
        for val, count in counts.items():
            if count > 100:
                name = get_stratum_name(val)
                key = 'Sec: ' + name
                suffix = 1
                while key in strata_masks:
                    key = 'Sec: ' + name + '_' + str(suffix)
                    suffix += 1
                strata_masks[key] = (df[sec_col] == val)
    else:
        print("Warning: Sector column (QDB) not found.")
    print('\nSample sizes per stratum:')
    print('Overall: ' + str(len(df)))
    for name, mask in strata_masks.items():
        print(name + ': ' + str(mask.sum()))
    print('\nFitting Stratified Models...')
    for name, mask in strata_masks.items():
        X_strat = X_scaled[mask]
        y_strat = y[mask]
        if len(np.unique(y_strat)) > 1:
            clf_strat = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.2, multi_class='multinomial', max_iter=5000, random_state=42)
            clf_strat.fit(X_strat, y_strat)
            if np.max(np.abs(clf_strat.coef_)) == 0:
                clf_strat = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, multi_class='multinomial', max_iter=5000, random_state=42)
                clf_strat.fit(X_strat, y_strat)
            if 1 in clf_strat.classes_:
                idx = list(clf_strat.classes_).index(1)
                results[name] = clf_strat.coef_[idx]
            else:
                results[name] = np.zeros(len(predictor_cols))
        else:
            results[name] = np.zeros(len(predictor_cols))
    coef_df = pd.DataFrame(results, index=predictor_cols)
    coef_df['abs_overall'] = coef_df['Overall'].abs()
    top_features = coef_df[coef_df['abs_overall'] > 0].sort_values('abs_overall', ascending=False).head(20).index
    if len(top_features) == 0:
        print('Warning: No non-zero coefficients found. Plotting top 20 by absolute value anyway.')
        top_features = coef_df.sort_values('abs_overall', ascending=False).head(20).index
    top_features_list = list(top_features)
    mandatory_features = [c for c in predictor_cols if c.startswith('QKB_1_4:') or c.startswith('QKB_1_11:') or c == 'QKB_1_4' or c == 'QKB_1_11']
    for mf in mandatory_features:
        if mf not in top_features_list:
            top_features_list.append(mf)
    plot_df = coef_df.loc[top_features_list].drop(columns=['abs_overall'])
    def clean_feature_name(name):
        parts = name.split(':')
        if len(parts) > 1:
            code = parts[0].strip()
            desc = parts[1].strip()
            return code + ': ' + desc
        else:
            return name
    clean_names = [textwrap.fill(clean_feature_name(c), width=50) for c in plot_df.index]
    plot_df.index = clean_names
    print('\nTop Elastic Net Coefficients for \'Resiliently Optimistic\' Class (Overall Model):')
    for feat, name in zip(top_features_list, clean_names):
        print(name.replace('\n', ' ') + ': ' + str(round(coef_df.loc[feat, 'Overall'], 4)))
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(20, 14))
    sns.heatmap(plot_df, cmap='coolwarm', center=0, annot=True, fmt='.2f', cbar_kws={'label': 'Elastic Net Coefficient (Class 1)'})
    plt.title('Top Elastic Net Coefficients for "Resiliently Optimistic" Class across Strata')
    plt.ylabel('Granular Policy / Enabler')
    plt.xlabel('Stratum')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_filename = os.path.join(data_dir, 'elastic_net_coefficients_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filename, dpi=300)
    print('\nCoefficient plot saved to ' + plot_filename)