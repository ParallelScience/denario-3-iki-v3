# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

if __name__ == '__main__':
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_data_step3.csv')
    raw_data_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'

    df = pd.read_csv(data_path)
    raw_df = pd.read_csv(raw_data_path, sep='\t', low_memory=False)

    qda_cols = [c for c in raw_df.columns if c.upper().startswith('QDA')]
    qdb_cols = [c for c in raw_df.columns if c.upper().startswith('QDB')]
    hiddg_cols = [c for c in raw_df.columns if 'HIDDG' in c.upper()]
    if not hiddg_cols:
        hiddg_cols = [c for c in raw_df.columns if c.upper().startswith('QDG')]

    df['HIDDG'] = raw_df[hiddg_cols[0]]
    df['QDB'] = raw_df[qdb_cols[0]]
    df['QDA'] = raw_df[qda_cols[0]]

    df_filtered = df[df['LCA_Class_Name'].isin(['Resiliently Optimistic', 'Anxiously Declining'])].copy()
    df_filtered['Target'] = (df_filtered['LCA_Class_Name'] == 'Resiliently Optimistic').astype(int)

    qkb_1_4_col = [c for c in df.columns if c.startswith('QKB_1_4')][0]
    qkb_1_11_col = [c for c in df.columns if c.startswith('QKB_1_11')][0]

    cols_to_keep = ['Target', 'Positive_Affect', 'Negative_Affect', 'HIDDG', 'QDB', 'QDA', qkb_1_4_col, qkb_1_11_col, 'QGM_Composite']
    df_model = df_filtered[cols_to_keep].dropna().copy()
    df_model.rename(columns={qkb_1_4_col: 'QKB_1_4', qkb_1_11_col: 'QKB_1_11'}, inplace=True)
    df_model['QDA_code'] = df_model['QDA'].astype('category').cat.codes

    print('Number of observations: ' + str(len(df_model)))

    df_model['QKB_1_4_x_QKB_1_11'] = df_model['QKB_1_4'] * df_model['QKB_1_11']

    exog_cols_dual = ['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'QKB_1_4_x_QKB_1_11', 'HIDDG', 'QDB']
    exog_df_dual = pd.get_dummies(df_model[exog_cols_dual], columns=['HIDDG', 'QDB'], drop_first=True, dtype=float)
    exog_df_dual.insert(0, 'Intercept', 1.0)

    logit_dual = sm.Logit(df_model['Target'], exog_df_dual)
    result_dual = logit_dual.fit(cov_type='cluster', cov_kwds={'groups': df_model['QDA_code']}, disp=0)

    print('\n' + '='*80)
    print('=== Dual-Pillar Hypothesis (Interaction QKB_1_4 * QKB_1_11) ===')
    print('='*80)
    print(result_dual.summary())

    df_model['QGM_Centered'] = df_model['QGM_Composite'] - df_model['QGM_Composite'].mean()
    df_model['QKB_1_11_Centered'] = df_model['QKB_1_11'] - df_model['QKB_1_11'].mean()
    df_model['QKB_1_11_x_QGM'] = df_model['QKB_1_11_Centered'] * df_model['QGM_Centered']

    exog_cols_mod = ['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11_Centered', 'QGM_Centered', 'QKB_1_11_x_QGM', 'HIDDG', 'QDB']
    exog_df_mod = pd.get_dummies(df_model[exog_cols_mod], columns=['HIDDG', 'QDB'], drop_first=True, dtype=float)
    exog_df_mod.insert(0, 'Intercept', 1.0)

    logit_mod = sm.Logit(df_model['Target'], exog_df_mod)
    result_mod = logit_mod.fit(cov_type='cluster', cov_kwds={'groups': df_model['QDA_code']}, disp=0)

    print('\n' + '='*80)
    print('=== Moderation Analysis: Informed Skeptic (Interaction QKB_1_11 * QGM_Composite) ===')
    print('='*80)
    print(result_mod.summary())

    beta_qkb = result_mod.params['QKB_1_11_Centered']
    beta_int = result_mod.params['QKB_1_11_x_QGM']

    cov_matrix = result_mod.cov_params()
    var_qkb = cov_matrix.loc['QKB_1_11_Centered', 'QKB_1_11_Centered']
    var_int = cov_matrix.loc['QKB_1_11_x_QGM', 'QKB_1_11_x_QGM']
    cov_qkb_int = cov_matrix.loc['QKB_1_11_Centered', 'QKB_1_11_x_QGM']

    qgm_mean = df_model['QGM_Composite'].mean()
    qgm_sd = df_model['QGM_Composite'].std()

    qgm_vals = {'Low (-1 SD)': -qgm_sd, 'Mean': 0, 'High (+1 SD)': qgm_sd}

    print('\n' + '='*80)
    print('=== Simple Slopes of QKB_1_11 at different levels of QGM_Composite ===')
    print('='*80)
    for label, qgm_val in qgm_vals.items():
        slope = beta_qkb + beta_int * qgm_val
        se_slope = np.sqrt(var_qkb + (qgm_val**2) * var_int + 2 * qgm_val * cov_qkb_int)
        z_val = slope / se_slope
        p_val = 2 * (1 - stats.norm.cdf(np.abs(z_val)))
        print(label + ' (Centered QGM = ' + str(round(qgm_val, 2)) + ', Raw QGM = ' + str(round(qgm_mean + qgm_val, 2)) + '):')
        print('  Slope: ' + str(round(slope, 4)) + ', SE: ' + str(round(se_slope, 4)) + ', z: ' + str(round(z_val, 4)) + ', p-value: ' + str(round(p_val, 4)))

    z_crit = 1.96
    z_crit_sq = z_crit**2
    A = beta_int**2 - z_crit_sq * var_int
    B = 2 * beta_qkb * beta_int - 2 * z_crit_sq * cov_qkb_int
    C = beta_qkb**2 - z_crit_sq * var_qkb
    discriminant = B**2 - 4 * A * C

    print('\n' + '='*80)
    print('=== Johnson-Neyman Interval for QGM_Composite ===')
    print('='*80)
    if discriminant < 0:
        print('No real roots found. The effect of QKB_1_11 is either always significant or never significant across all values of QGM.')
    else:
        root1 = (-B - np.sqrt(discriminant)) / (2 * A)
        root2 = (-B + np.sqrt(discriminant)) / (2 * A)
        jn_lower = min(root1, root2)
        jn_upper = max(root1, root2)
        raw_jn_lower = jn_lower + qgm_mean
        raw_jn_upper = jn_upper + qgm_mean
        print('The effect of Employee Involvement (QKB_1_11) on the probability of being Resiliently Optimistic')
        print('is statistically significant (p < 0.05) outside the interval of Centered QGM: [' + str(round(jn_lower, 4)) + ', ' + str(round(jn_upper, 4)) + ']')
        print('In terms of Raw QGM_Composite, the interval is: [' + str(round(raw_jn_lower, 4)) + ', ' + str(round(raw_jn_upper, 4)) + ']')

    output_path = os.path.join(data_dir, 'processed_data_step5.csv')
    df_model.to_csv(output_path, index=False)
    print('\nData saved to ' + output_path)