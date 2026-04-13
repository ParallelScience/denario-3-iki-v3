# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm

def generate_plots():
    plt.rcParams['text.usetex'] = False
    raw_path = 'data/data_raw_encoded.csv'
    dummy_path = 'data/data_dummy_encoded.csv'
    res_path = 'data/mnlogit_results.csv'
    print('Loading data...')
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_dummy = pd.read_csv(dummy_path, low_memory=False)
    res_df = pd.read_csv(res_path)
    valid_target = df_dummy['LCA_Class_Name'].notna()
    df_dummy = df_dummy[valid_target].copy()
    df_raw = df_raw[valid_target].copy()
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    print('Generating (a) Latent Class Profiles...')
    ax_a = axs[0, 0]
    ind_cols = ['QEA_2: Security of your current role - What impact does AI currently have in the following areas? Please rate the positive or negative impact in each area?', 'QEB_2: Security of your current role - What impact do you think will AI have on your work in the next 3 years? Please rate the positive or negative impact you believe AI will have in each area?']
    profile_data = df_raw.groupby('LCA_Class_Name')[ind_cols].mean()
    markers = ['o', 's', '^', 'D', 'v']
    for i, c in enumerate(profile_data.index):
        ax_a.plot(['Current (QEA_2)', 'Expected (QEB_2)'], profile_data.loc[c], marker=markers[i % len(markers)], label=c, linewidth=2, markersize=8)
    ax_a.set_ylim(-2.2, 2.2)
    ax_a.set_ylabel('Mean Impact Score (-2 to +2)')
    ax_a.set_title('(a) Latent Class Trajectory Profiles')
    ax_a.grid(True, linestyle='--', alpha=0.7)
    ax_a.legend(title='Latent Class')
    print('Generating (b) Marginal Effects Plot...')
    ax_b = axs[0, 1]
    qkb_11_cols = [c for c in df_dummy.columns if c.startswith('QKB_1_11:') and not c.endswith('_Not_Sure')]
    qkb_4_cols = [c for c in df_dummy.columns if c.startswith('QKB_1_4:') and not c.endswith('_Not_Sure')]
    qkb_11_col = qkb_11_cols[0]
    qkb_4_col = qkb_4_cols[0]
    base_predictors = [qkb_11_col, qkb_4_col, 'Company_Culture_Index', 'Affective_Disposition_Factor_1', 'Affective_Disposition_Factor_2']
    hiddg_cols = sorted([c for c in df_dummy.columns if c.startswith('HIDDG_')])[1:]
    qdb_cols = sorted([c for c in df_dummy.columns if c.startswith('QDB_')])[1:]
    qda_cols = sorted([c for c in df_dummy.columns if c.startswith('QDA_')])[1:]
    all_predictors = base_predictors + hiddg_cols + qdb_cols + qda_cols
    X_raw = df_dummy[all_predictors].copy()
    for col in X_raw.columns:
        if X_raw[col].isna().any():
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())
    for col in base_predictors:
        X_raw[col] = (X_raw[col] - X_raw[col].mean()) / (X_raw[col].std() + 1e-09)
    X_raw['Interaction_QKB11_QKB4'] = X_raw[qkb_11_col] * X_raw[qkb_4_col]
    X = sm.add_constant(X_raw)
    X_mean = X.mean()
    qkb11_vals = np.linspace(-2, 2, 50)
    qkb4_levels = [-1, 0, 1]
    qkb4_labels = ['Low Training (-1 SD)', 'Mean Training (0 SD)', 'High Training (+1 SD)']
    colors = ['#d62728', '#7f7f7f', '#2ca02c']
    model_classes = res_df['Class'].unique()
    target_class = 'Resiliently Optimistic'
    if target_class in model_classes:
        for qkb4_val, label, color in zip(qkb4_levels, qkb4_labels, colors):
            y_probs = []
            for qkb11_val in qkb11_vals:
                x_vec = X_mean.copy()
                x_vec[qkb_11_col] = qkb11_val
                x_vec[qkb_4_col] = qkb4_val
                x_vec['Interaction_QKB11_QKB4'] = qkb11_val * qkb4_val
                etas = {}
                for c in model_classes:
                    coefs = res_df[res_df['Class'] == c].set_index('Feature')['Coef']
                    coefs = coefs.reindex(x_vec.index).fillna(0)
                    eta = np.sum(x_vec * coefs)
                    etas[c] = eta
                exp_etas = {c: np.exp(etas[c]) for c in model_classes}
                sum_exp = 1.0 + sum(exp_etas.values())
                prob = exp_etas[target_class] / sum_exp
                y_probs.append(prob)
            ax_b.plot(qkb11_vals, y_probs, label=label, color=color, linewidth=2)
        ax_b.set_xlabel('Employee Involvement in Dev (Standardized)')
        ax_b.set_ylabel('Probability of "' + target_class + '"')
        ax_b.set_title('(b) Marginal Effects of Involvement x Training')
        ax_b.legend()
        ax_b.grid(True, linestyle='--', alpha=0.7)
    else:
        ax_b.text(0.5, 0.5, 'Class "' + target_class + '" not found in model', ha='center', va='center')
    print('Generating (c) Coefficient Forest Plot...')
    ax_c = axs[1, 0]
    key_features = ['Interaction_QKB11_QKB4', qkb_11_col, qkb_4_col, 'Company_Culture_Index', 'Affective_Disposition_Factor_1', 'Affective_Disposition_Factor_2']
    feature_labels = ['Involvement x Training', 'Employee Involvement', 'Regular Training', 'Company Culture', 'Positive Affect (F1)', 'Negative Affect (F2)']
    key_features = key_features[::-1]
    feature_labels = feature_labels[::-1]
    plot_classes = [c for c in ['Resiliently Optimistic', 'Anxiously Declining'] if c in model_classes]
    y_pos = np.arange(len(key_features))
    offsets = [-0.15, 0.15] if len(plot_classes) == 2 else [0]
    for i, c in enumerate(plot_classes):
        c_data = res_df[(res_df['Class'] == c) & (res_df['Feature'].isin(key_features))].copy()
        c_data['Feature'] = pd.Categorical(c_data['Feature'], categories=key_features, ordered=True)
        c_data = c_data.sort_values('Feature')
        ors = c_data['OR'].values
        ci_lower = c_data['CI_Lower'].values
        ci_upper = c_data['CI_Upper'].values
        ax_c.errorbar(ors, y_pos + offsets[i], xerr=[ors - ci_lower, ci_upper - ors], fmt='o', label=c, linewidth=2, capsize=4)
    ax_c.axvline(1.0, color='black', linestyle='--', alpha=0.7)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(feature_labels)
    ax_c.set_xlabel('Odds Ratio (95% CI)')
    ax_c.set_title('(c) Predictors of Class Membership (Ref: Stagnant Neutral)')
    ax_c.legend()
    ax_c.grid(True, axis='x', linestyle='--', alpha=0.7)
    print('Generating (d) Bar Chart of Class Proportions...')
    ax_d = axs[1, 1]
    class_counts = df_raw['LCA_Class_Name'].value_counts(normalize=True) * 100
    class_counts = class_counts.sort_values(ascending=True)
    bars = ax_d.barh(class_counts.index, class_counts.values, color='skyblue', edgecolor='black')
    ax_d.set_xlabel('Percentage of Respondents (%)')
    ax_d.set_title('(d) Latent Class Membership Proportions')
    ax_d.grid(True, axis='x', linestyle='--', alpha=0.7)
    for bar in bars:
        width = bar.get_width()
        ax_d.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(round(width, 1)) + '%', ha='left', va='center')
    ax_d.set_xlim(0, max(class_counts.values) + 15)
    plt.tight_layout()
    plot_filename = 'data/results_summary_1_' + str(int(time.time())) + '.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print('Multi-panel figure saved to ' + plot_filename)
    print('\n--- Summary Statistics for Researcher ---')
    print('Class Proportions:')
    print(class_counts.to_string())
    print('\nMarginal Effects (Probability of Resiliently Optimistic):')
    for qkb4_val, label in zip(qkb4_levels, qkb4_labels):
        probs_print = []
        for qkb11_val in [-2, 0, 2]:
            x_vec = X_mean.copy()
            x_vec[qkb_11_col] = qkb11_val
            x_vec[qkb_4_col] = qkb4_val
            x_vec['Interaction_QKB11_QKB4'] = qkb11_val * qkb4_val
            etas = {}
            for c in model_classes:
                coefs = res_df[res_df['Class'] == c].set_index('Feature')['Coef']
                coefs = coefs.reindex(x_vec.index).fillna(0)
                eta = np.sum(x_vec * coefs)
                etas[c] = eta
            exp_etas = {c: np.exp(etas[c]) for c in model_classes}
            sum_exp = 1.0 + sum(exp_etas.values())
            prob = exp_etas[target_class] / sum_exp
            probs_print.append(prob)
        print(label + ': Inv=-2SD -> ' + str(round(probs_print[0], 4)) + ', Inv=0SD -> ' + str(round(probs_print[1], 4)) + ', Inv=+2SD -> ' + str(round(probs_print[2], 4)))
    print('\nOdds Ratios for Key Features:')
    for c in plot_classes:
        print('\nClass: ' + c)
        c_data = res_df[(res_df['Class'] == c) & (res_df['Feature'].isin(key_features))].copy()
        c_data['Feature'] = pd.Categorical(c_data['Feature'], categories=key_features, ordered=True)
        c_data = c_data.sort_values('Feature')
        for _, row in c_data.iterrows():
            print('  ' + str(row['Feature'])[:30] + '... : OR=' + str(round(row['OR'], 3)) + ' (95% CI: ' + str(round(row['CI_Lower'], 3)) + '-' + str(round(row['CI_Upper'], 3)) + '), p=' + str(round(row['FDR_P_value'], 4)))

if __name__ == '__main__':
    generate_plots()