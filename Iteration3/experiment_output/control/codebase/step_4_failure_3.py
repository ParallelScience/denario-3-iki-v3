# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    data_dir = 'data'
    data_path = os.path.join(data_dir, 'processed_data_step3.csv')
    raw_data_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    df = pd.read_csv(data_path)
    raw_df = pd.read_csv(raw_data_path, sep='\t', low_memory=False)
    raw_cols_lower = {c.lower(): c for c in raw_df.columns}
    hiddg_col_raw = raw_cols_lower.get('hiddg')
    qdb_col_raw = raw_cols_lower.get('qdb')
    qda_col_raw = raw_cols_lower.get('qda')
    if not hiddg_col_raw:
        possible_hiddg = [c for c in raw_df.columns if 'hiddg' in c.lower() or 'dg' in c.lower()]
        hiddg_col_raw = possible_hiddg[0] if possible_hiddg else raw_cols_lower.get('qdg')
    df['HIDDG'] = raw_df[hiddg_col_raw]
    df['QDB'] = raw_df[qdb_col_raw]
    df['QDA'] = raw_df[qda_col_raw]
    df_filtered = df[df['LCA_Class_Name'].isin(['Resiliently Optimistic', 'Anxiously Declining'])].copy()
    df_filtered['Target'] = (df_filtered['LCA_Class_Name'] == 'Resiliently Optimistic').astype(int)
    qkb_1_4_col = [c for c in df.columns if c.startswith('QKB_1_4')][0]
    qkb_1_11_col = [c for c in df.columns if c.startswith('QKB_1_11')][0]
    cols_to_keep = ['Target', 'Positive_Affect', 'Negative_Affect', 'HIDDG', 'QDB', 'QDA', qkb_1_4_col, qkb_1_11_col]
    df_model = df_filtered[cols_to_keep].dropna().copy()
    df_model.rename(columns={qkb_1_4_col: 'QKB_1_4', qkb_1_11_col: 'QKB_1_11'}, inplace=True)
    exog_df = df_model[['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'HIDDG', 'QDB']]
    exog_df = pd.get_dummies(exog_df, columns=['HIDDG', 'QDB'], drop_first=True, dtype=float)
    exog_df.insert(0, 'Intercept', 1.0)
    exog_vc = pd.get_dummies(df_model['QDA'], drop_first=False, dtype=float)
    ident = np.zeros(exog_vc.shape[1], dtype=int)
    try:
        model = BinomialBayesMixedGLM(endog=df_model['Target'], exog=exog_df, exog_vc=exog_vc, ident=ident)
        result = model.fit_vb()
        fe_names = model.exog_names
        fe_mean = result.fe_mean if hasattr(result, 'fe_mean') else result.params[:len(fe_names)]
        fe_sd = result.fe_sd if hasattr(result, 'fe_sd') else result.bse[:len(fe_names)]
        z_scores = fe_mean / fe_sd
        pvals = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        results_df = pd.DataFrame({'Coefficient': fe_mean, 'Odds Ratio': np.exp(fe_mean), 'OR CI Lower': np.exp(fe_mean - 1.96 * fe_sd), 'OR CI Upper': np.exp(fe_mean + 1.96 * fe_sd), 'p-value': pvals, 'FDR p-value': pvals_corrected, 'Significant (FDR < 0.05)': reject}, index=fe_names)
        results_df.to_csv(os.path.join(data_dir, 'logistic_regression_results.csv'))
    except Exception:
        try:
            gee_model = sm.GEE(df_model['Target'], exog_df, groups=df_model['QDA'], family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
            gee_result = gee_model.fit()
            results_df = pd.DataFrame({'Coefficient': gee_result.params, 'Odds Ratio': np.exp(gee_result.params), 'p-value': gee_result.pvalues})
            results_df.to_csv(os.path.join(data_dir, 'logistic_regression_results.csv'))
        except Exception:
            exog_df_fixed = pd.get_dummies(df_model[['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'HIDDG', 'QDB', 'QDA']], drop_first=True, dtype=float)
            exog_df_fixed.insert(0, 'Intercept', 1.0)
            logit_model = sm.Logit(df_model['Target'], exog_df_fixed)
            logit_result = logit_model.fit(disp=0)
            results_df = pd.DataFrame({'Coefficient': logit_result.params, 'Odds Ratio': np.exp(logit_result.params), 'p-value': logit_result.pvalues})
            results_df.to_csv(os.path.join(data_dir, 'logistic_regression_results.csv'))