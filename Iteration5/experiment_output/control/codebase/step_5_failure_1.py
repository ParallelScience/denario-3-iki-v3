# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
def main():
    data_dir = 'data/'
    timestamp = int(time.time())
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_dataset_step2.csv'), low_memory=False)
    raw_df = pd.read_csv('/home/node/work/projects/iki_v2/IKI-Data-Raw.csv', sep='\t', low_memory=False)
    for col in ['HidQDC', 'HIDDG', 'QDB', 'QDA']:
        if col not in df.columns and col in raw_df.columns:
            df[col] = raw_df[col]
    targets = ['Job_Security_Index_Current', 'Job_Security_Index_Future']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_strs = ['Not_Sure', 'Income_Val', 'Job_Security_Index']
    num_cols = [c for c in num_cols if not any(x in c for x in exclude_strs)]
    corrs = []
    for col in num_cols:
        valid = df[[col] + targets].dropna()
        if len(valid) > 100 and valid[col].var() > 0:
            corr_curr, _ = spearmanr(valid[col], valid['Job_Security_Index_Current'])
            corr_fut, _ = spearmanr(valid[col], valid['Job_Security_Index_Future'])
            corrs.append({'Feature': col, 'Corr_Current': corr_curr, 'Corr_Future': corr_fut})
    corr_df = pd.DataFrame(corrs)
    corr_df['Max_Abs_Corr'] = corr_df[['Corr_Current', 'Corr_Future']].abs().max(axis=1)
    top_features = corr_df.sort_values('Max_Abs_Corr', ascending=False).head(15)['Feature'].tolist()
    plot_data = corr_df[corr_df['Feature'].isin(top_features)].set_index('Feature')[['Corr_Current', 'Corr_Future']]
    plt.figure(figsize=(10, 12))
    sns.heatmap(plot_data, annot=True, cmap='coolwarm', center=0, vmin=-0.6, vmax=0.6, fmt='.2f')
    plt.title('Spearman Rank Correlations: Top Predictors vs Job Security')
    plt.tight_layout()
    heatmap_path = os.path.join(data_dir, 'heatmap_correlations_1_' + str(timestamp) + '.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print('Heatmap saved to ' + heatmap_path)
    print('=== Top 15 Spearman Correlations ===')
    print(corr_df.sort_values('Max_Abs_Corr', ascending=False).head(15).to_string(index=False))
    print('\n')
    def get_ols_coefs(target_col, task_trans_col):
        valid_idx = df[target_col].notna()
        df_mod = df[valid_idx].copy()
        features = ['Organizational_Support_Index', 'Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', task_trans_col]
        missing_feats = [f for f in features if f not in df_mod.columns]
        if missing_feats:
            for f in missing_feats:
                df_mod[f] = 0
        df_mod[features] = df_mod[features].fillna(df_mod[features].mean())
        scaler = StandardScaler()
        df_mod[features] = scaler.fit_transform(df_mod[features])
        X = df_mod[features].copy()
        X['QKB_Interaction'] = X['QKB_1_4'] * X['QKB_1_11']
        X['Task_Trans_Sq'] = X[task_trans_col] ** 2
        X = sm.add_constant(X)
        y = df_mod[target_col]
        mod = sm.OLS(y, X)
        res = mod.fit()
        coefs = res.params.drop('const')
        conf_int = res.conf_int().drop('const')
        res_df = pd.DataFrame({'Feature': coefs.index, 'Coef': coefs.values, 'CI_Lower': conf_int[0].values, 'CI_Upper': conf_int[1].values})
        return res_df
    coefs_curr = get_ols_coefs('Job_Security_Index_Current', 'Nature_of_Work_Change_Index')
    coefs_fut = get_ols_coefs('Job_Security_Index_Future', 'Future_Nature_of_Work_Change_Index')
    print('=== Standardized Regression Coefficients (OLS) ===')
    print('Current Job Security:')
    print(coefs_curr.to_string(index=False))
    print('\nFuture Job Security:')
    print(coefs_fut.to_string(index=False))
    print('\n')
    plt.figure(figsize=(10, 8))
    y_ticks = np.arange(len(coefs_curr['Feature']))
    offset = 0.2
    plt.errorbar(coefs_curr['Coef'], y_ticks - offset, xerr=[coefs_curr['Coef'] - coefs_curr['CI_Lower'], coefs_curr['CI_Upper'] - coefs_curr['Coef']], fmt='o', label='Current', capsize=5, color='royalblue')
    plt.errorbar(coefs_fut['Coef'], y_ticks + offset, xerr=[coefs_fut['Coef'] - coefs_fut['CI_Lower'], coefs_fut['CI_Upper'] - coefs_fut['Coef']], fmt='s', label='Future', capsize=5, color='darkorange')
    plt.yticks(y_ticks, coefs_curr['Feature'])
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel('Standardized Coefficient')
    plt.title('Forest Plot of Standardized Regression Coefficients')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    forest_path = os.path.join(data_dir, 'forest_plot_2_' + str(timestamp) + '.png')
    plt.savefig(forest_path, dpi=300)
    plt.close()
    print('Forest plot saved to ' + forest_path)
    rf_curr_path = os.path.join(data_dir, 'rf_importance_current.csv')
    rf_fut_path = os.path.join(data_dir, 'rf_importance_future.csv')
    if os.path.exists(rf_curr_path) and os.path.exists(rf_fut_path):
        rf_curr = pd.read_csv(rf_curr_path).head(15)
        rf_fut = pd.read_csv(rf_fut_path).head(15)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        sns.barplot(x='Importance', y='Feature', data=rf_curr, ax=axes[0], color='skyblue')
        axes[0].set_title('RF Permutation Importance (Current Job Security)')
        sns.barplot(x='Importance', y='Feature', data=rf_fut, ax=axes[1], color='lightgreen')
        axes[1].set_title('RF Permutation Importance (Future Job Security)')
        plt.tight_layout()
        rf_path = os.path.join(data_dir, 'rf_importance_3_' + str(timestamp) + '.png')
        plt.savefig(rf_path, dpi=300)
        plt.close()
    med_path = os.path.join(data_dir, 'mediation_results.csv')
    if os.path.exists(med_path):
        med_df = pd.read_csv(med_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, row in med_df.iterrows():
            if i >= 2: break
            ax = axes[i]
            ax.axis('off')
            ax.text(0.2, 0.3, 'Organizational\nSupport (X)', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue'))
            ax.text(0.5, 0.8, 'Task\nTransformation (M)', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen'))
            ax.text(0.8, 0.3, 'Job\nSecurity (Y)', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral'))
            ax.annotate('', xy=(0.45, 0.7), xytext=(0.25, 0.4), arrowprops=dict(arrowstyle='->', lw=2))
            ax.text(0.3, 0.6, 'a = ' + str(round(row['a'], 3)), ha='center', va='center', rotation=45)
            ax.annotate('', xy=(0.75, 0.4), xytext=(0.55, 0.7), arrowprops=dict(arrowstyle='->', lw=2))
            ax.text(0.7, 0.6, 'b = ' + str(round(row['b'], 3)), ha='center', va='center', rotation=-45)
            ax.annotate('', xy=(0.7, 0.3), xytext=(0.3, 0.3), arrowprops=dict(arrowstyle='->', lw=2))
            ax.text(0.5, 0.2, "c' = " + str(round(row['c_prime'], 3)) + '\n(Indirect = ' + str(round(row['indirect'], 3)) + ')', ha='center', va='center')
            ax.set_title('Mediation Path: ' + row['Model'] + ' Job Security')
        plt.tight_layout()
        med_plot_path = os.path.join(data_dir, 'mediation_paths_4_' + str(timestamp) + '.png')
        plt.savefig(med_plot_path, dpi=300)
        plt.close()
    subgroups = ['HidQDC', 'HIDDG', 'QDA']
    fig, axes = plt.subplots(len(subgroups), 1, figsize=(12, 6 * len(subgroups)))
    for i, grp in enumerate(subgroups):
        if grp not in df.columns: continue
        plot_data = []
        for tgt in targets:
            valid = df[[grp, tgt]].dropna()
            valid = valid[valid[grp] != 'Unknown']
            stats = valid.groupby(grp)[tgt].agg(['mean', 'count', 'std']).reset_index()
            stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
            stats['Target'] = 'Current' if 'Current' in tgt else 'Future'
            plot_data.append(stats)
        if not plot_data: continue
        plot_df = pd.concat(plot_data)
        curr_stats = plot_df[plot_df['Target'] == 'Current'].set_index(grp)
        order = curr_stats.sort_values('mean', ascending=False).index.tolist()
        ax = axes[i]
        width = 0.35
        x = np.arange(len(order))
        curr_data = plot_df[plot_df['Target'] == 'Current'].set_index(grp).reindex(order)
        fut_data = plot_df[plot_df['Target'] == 'Future'].set_index(grp).reindex(order)
        ax.bar(x - width/2, curr_data['mean'], width, yerr=curr_data['ci'], label='Current', capsize=5, color='skyblue')
        ax.bar(x + width/2, fut_data['mean'], width, yerr=fut_data['ci'], label='Future', capsize=5, color='lightgreen')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha='right')
        ax.set_ylabel('Mean Job Security Index')
        ax.set_title('Job Security by ' + grp)
        ax.legend()
    plt.tight_layout()
    subgroup_path = os.path.join(data_dir, 'subgroup_bars_5_' + str(timestamp) + '.png')
    plt.savefig(subgroup_path, dpi=300)
    plt.close()
if __name__ == '__main__':
    main()