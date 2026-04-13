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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
def map_likert(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'not sure' in s or "don't know" in s: return np.nan
    if 'significantly negative' in s: return -2
    if 'slightly negative' in s: return -1
    if 'no impact' in s: return 0
    if 'slightly positive' in s: return 1
    if 'significantly positive' in s: return 2
    if 'large decrease' in s: return -2
    if 'moderate decrease' in s: return -1
    if 'no major change' in s: return 0
    if 'moderate increase' in s: return 1
    if 'large increase' in s: return 2
    if 'not at all important' in s: return -2
    if 'slightly important' in s: return -1
    if 'moderately important' in s: return 0
    if 'very important' in s: return 1
    if 'extremely important' in s: return 2
    try:
        v = float(s)
        if 1 <= v <= 5: return int(v) - 3
    except ValueError: pass
    return np.nan
def truncate_label(label, max_len=40):
    label_str = str(label)
    if len(label_str) > max_len: return label_str[:max_len-3] + '...'
    return label_str
def perform_mediation(df, x_col, m_col, y_col):
    data = df[[x_col, m_col, y_col]].dropna()
    X = data[x_col].values
    M = data[m_col].values
    Y = data[y_col].values
    n = len(data)
    var_x = np.var(X, ddof=1)
    if var_x == 0: return None
    a = np.cov(X, M)[0, 1] / var_x
    X_mat = np.column_stack((X, M, np.ones(n)))
    coeffs = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    c_prime = coeffs[0]
    b = coeffs[1]
    c = np.cov(X, Y)[0, 1] / var_x
    indirect_effect = a * b
    return {'a': a, 'b': b, 'c_prime': c_prime, 'c': c, 'indirect': indirect_effect}
def compute_rf_importance(df, target_col, name, data_dir):
    potential_num_cols = ['QDC', 'QDE_Year', 'QC', 'Income_Rank', 'QGI', 'QGS', 'Organizational_Support_Index', 'Positive_Affect', 'Negative_Affect', 'Nature_of_Work_Change_Index', 'Future_Nature_of_Work_Change_Index']
    prefixes = ['QHD_', 'QGM_', 'QF_', 'QKC_', 'QED_1_', 'QKB_1_', 'QKB_2_', 'QGO_']
    for col in df.columns:
        if any(col.startswith(p) for p in prefixes): potential_num_cols.append(col)
    potential_cat_cols = ['QDA', 'QDB', 'HidQDC', 'QDD', 'QDH', 'QDG', 'HIDDG', 'Global Employee Size', 'Global Annual Revenue', 'Market Capitalization', 'QGG', 'QGN', 'QA_1_1', 'QKD']
    num_cols = [c for c in potential_num_cols if c in df.columns and c != target_col]
    cat_cols = [c for c in potential_cat_cols if c in df.columns and c != target_col]
    num_cols = list(set(num_cols))
    cat_cols = list(set(cat_cols))
    valid_idx = df[target_col].notna()
    df_valid = df[valid_idx]
    X = df_valid[num_cols + cat_cols]
    y = df_valid[target_col]
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf)])
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    feature_names = num_cols + cat_cols
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean, 'Std': result.importances_std})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
    imp_df.to_csv(os.path.join(data_dir, 'rf_importance_' + name.lower() + '.csv'), index=False)
    return imp_df
def main():
    data_dir = 'data/'
    timestamp = int(time.time())
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_dataset_step2.csv'), low_memory=False)
    raw_df = pd.read_csv('/home/node/work/projects/iki_v2/IKI-Data-Raw.csv', sep='\t', low_memory=False)
    def find_col(prefix):
        return next((col for col in raw_df.columns if col.startswith(prefix)), None)
    qea_2_col = find_col('QEA_2')
    qeb_2_col = find_col('QEB_2')
    qkb_1_4_col = find_col('QKB_1_4')
    qkb_1_11_col = find_col('QKB_1_11')
    qgp_cols = [find_col('QGP_' + str(i)) for i in range(1, 4)]
    qgu_cols = [find_col('QGU_' + str(i)) for i in range(1, 4)]
    if qea_2_col: df['Job_Security_Index_Current'] = raw_df[qea_2_col].apply(map_likert)
    if qeb_2_col: df['Job_Security_Index_Future'] = raw_df[qeb_2_col].apply(map_likert)
    if qkb_1_4_col: df['QKB_1_4'] = raw_df[qkb_1_4_col].apply(map_likert)
    if qkb_1_11_col: df['QKB_1_11'] = raw_df[qkb_1_11_col].apply(map_likert)
    if all(qgp_cols): df['Nature_of_Work_Change_Index'] = raw_df[qgp_cols].map(map_likert).mean(axis=1)
    if all(qgu_cols): df['Future_Nature_of_Work_Change_Index'] = raw_df[qgu_cols].map(map_likert).mean(axis=1)
    for prefix in ['HidQDC', 'HIDDG', 'QDB', 'QDA']:
        col = find_col(prefix)
        if col: df[prefix] = raw_df[col]
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
    plot_data.index = [truncate_label(idx, 45) for idx in plot_data.index]
    plt.figure(figsize=(12, 12))
    sns.heatmap(plot_data, annot=True, cmap='coolwarm', center=0, vmin=-0.6, vmax=0.6, fmt='.2f')
    plt.title('Spearman Rank Correlations: Top Predictors vs Job Security')
    plt.tight_layout()
    heatmap_path = os.path.join(data_dir, 'heatmap_correlations_1_' + str(timestamp) + '.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Heatmap saved to ' + heatmap_path)
    def get_ols_coefs(target_col, task_trans_col):
        valid_idx = df[target_col].notna()
        df_mod = df[valid_idx].copy()
        features = ['Organizational_Support_Index', 'Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', task_trans_col]
        missing_feats = [f for f in features if f not in df_mod.columns]
        for f in missing_feats: df_mod[f] = 0
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
        return pd.DataFrame({'Feature': coefs.index, 'Coef': coefs.values, 'CI_Lower': conf_int[0].values, 'CI_Upper': conf_int[1].values})
    coefs_curr = get_ols_coefs('Job_Security_Index_Current', 'Nature_of_Work_Change_Index')
    coefs_fut = get_ols_coefs('Job_Security_Index_Future', 'Future_Nature_of_Work_Change_Index')
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
    plt.savefig(forest_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Forest plot saved to ' + forest_path)
    rf_curr_path = os.path.join(data_dir, 'rf_importance_current.csv')
    rf_fut_path = os.path.join(data_dir, 'rf_importance_future.csv')
    if not os.path.exists(rf_curr_path): rf_curr = compute_rf_importance(df, 'Job_Security_Index_Current', 'Current', data_dir)
    else: rf_curr = pd.read_csv(rf_curr_path).head(15)
    if not os.path.exists(rf_fut_path): rf_fut = compute_rf_importance(df, 'Job_Security_Index_Future', 'Future', data_dir)
    else: rf_fut = pd.read_csv(rf_fut_path).head(15)
    rf_curr['Feature'] = rf_curr['Feature'].apply(lambda x: truncate_label(x, 35))
    rf_fut['Feature'] = rf_fut['Feature'].apply(lambda x: truncate_label(x, 35))
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    sns.barplot(x='Importance', y='Feature', data=rf_curr.head(15), ax=axes[0], color='skyblue')
    axes[0].set_title('RF Permutation Importance (Current Job Security)')
    sns.barplot(x='Importance', y='Feature', data=rf_fut.head(15), ax=axes[1], color='lightgreen')
    axes[1].set_title('RF Permutation Importance (Future Job Security)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'rf_importance_3_' + str(timestamp) + '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    med_path = os.path.join(data_dir, 'mediation_results.csv')
    if not os.path.exists(med_path):
        mediation_results = []
        res_curr = perform_mediation(df, 'Organizational_Support_Index', 'Nature_of_Work_Change_Index', 'Job_Security_Index_Current')
        if res_curr: res_curr['Model'] = 'Current'; mediation_results.append(res_curr)
        res_fut = perform_mediation(df, 'Organizational_Support_Index', 'Future_Nature_of_Work_Change_Index', 'Job_Security_Index_Future')
        if res_fut: res_fut['Model'] = 'Future'; mediation_results.append(res_fut)
        if mediation_results: med_df = pd.DataFrame(mediation_results); med_df.to_csv(med_path, index=False)
        else: med_df = pd.DataFrame()
    else: med_df = pd.read_csv(med_path)
    if not med_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, row in med_df.iterrows():
            if i >= 2: break
            ax = axes[i]; ax.axis('off')
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
        plt.savefig(os.path.join(data_dir, 'mediation_paths_4_' + str(timestamp) + '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    subgroups = ['HidQDC', 'HIDDG', 'QDA']
    fig, axes = plt.subplots(len(subgroups), 1, figsize=(12, 8 * len(subgroups)))
    for i, grp in enumerate(subgroups):
        if grp not in df.columns: continue
        plot_data = []
        for tgt in targets:
            valid = df[[grp, tgt]].copy()
            valid[grp] = valid[grp].fillna('Unknown').astype(str)
            valid = valid.dropna(subset=[tgt])
            valid = valid[valid[grp] != 'Unknown']
            valid = valid[valid[grp] != 'nan']
            if valid.empty: continue
            stats = valid.groupby(grp)[tgt].agg(['mean', 'count', 'std']).reset_index()
            stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
            stats['Target'] = 'Current' if 'Current' in tgt else 'Future'
            plot_data.append(stats)
        if not plot_data: continue
        plot_df = pd.concat(plot_data)
        if plot_df.empty: continue
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
        ax.set_xticklabels([truncate_label(l, 30) for l in order], rotation=45, ha='right')
        ax.set_ylabel('Mean Job Security Index')
        ax.set_title('Job Security by ' + grp)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'subgroup_bars_5_' + str(timestamp) + '.png'), dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    main()