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
    df['HIDDG'] = raw_df['HIDDG']
    df['QDB'] = raw_df['QDB']
    df['QDA'] = raw_df['QDA']
    df_filtered = df[df['LCA_Class_Name'].isin(['Resiliently Optimistic', 'Anxiously Declining'])].copy()
    df_filtered['Target'] = (df_filtered['LCA_Class_Name'] == 'Resiliently Optimistic').astype(int)
    qkb_1_4_col = [c for c in df.columns if c.startswith('QKB_1_4')][0]
    qkb_1_11_col = [c for c in df.columns if c.startswith('QKB_1_11')][0]
    cols_to_keep = ['Target', 'Positive_Affect', 'Negative_Affect', 'HIDDG', 'QDB', 'QDA', qkb_1_4_col, qkb_1_11_col]
    df_model = df_filtered[cols_to_keep].dropna().copy()
    df_model.rename(columns={qkb_1_4_col: 'QKB_1_4', qkb_1_11_col: 'QKB_1_11'}, inplace=True)
    print('Number of observations after filtering and dropping NaNs: ' + str(len(df_model)))
    exog_df = df_model[['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'HIDDG', 'QDB']]
    exog_df = pd.get_dummies(exog_df, columns=['HIDDG', 'QDB'], drop_first=True, dtype=float)
    exog_df.insert(0, 'Intercept', 1.0)
    exog_vc = pd.get_dummies(df_model['QDA'], drop_first=False, dtype=float)
    ident = np.zeros(exog_vc.shape[1], dtype=int)
    try:
        print('\nFitting Bayesian Mixed GLM (Random Intercept for QDA)...')
        model = BinomialBayesMixedGLM(endog=df_model['Target'], exog=exog_df, exog_vc=exog_vc, ident=ident)
        result = model.fit_vb()
        fe_names = model.exog_names
        if hasattr(result, 'fe_mean') and hasattr(result, 'fe_sd'):
            fe_mean = result.fe_mean
            fe_sd = result.fe_sd
        else:
            fe_mean = result.params[:len(fe_names)]
            if hasattr(result, 'bse'):
                fe_sd = result.bse[:len(fe_names)]
            else:
                raise AttributeError('Standard errors not found in result object.')
        z_scores = fe_mean / fe_sd
        pvals = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        odds_ratios = np.exp(fe_mean)
        or_ci_lower = np.exp(fe_mean - 1.96 * fe_sd)
        or_ci_upper = np.exp(fe_mean + 1.96 * fe_sd)
        results_df = pd.DataFrame({'Coefficient': fe_mean, 'Odds Ratio': odds_ratios, 'OR CI Lower': or_ci_lower, 'OR CI Upper': or_ci_upper, 'p-value': pvals, 'FDR p-value': pvals_corrected, 'Significant (FDR < 0.05)': reject}, index=fe_names)
        print('\nMain-Effects Logistic Regression Results (Bayesian Mixed GLM):')
        print(results_df.round(4))
        results_df.to_csv(os.path.join(data_dir, 'logistic_regression_results.csv'))
        try:
            if hasattr(result, 'predict'):
                preds_prob = result.predict()
            else:
                lin_pred = np.dot(exog_df, fe_mean)
                preds_prob = 1 / (1 + np.exp(-lin_pred))
            preds_class = (preds_prob > 0.5).astype(int)
            cm = confusion_matrix(df_model['Target'], preds_class)
            print('\nConfusion Matrix:')
            print(cm)
        except Exception as e:
            print('Could not compute confusion matrix: ' + str(e))
    except Exception as e:
        print('\nBayesian Mixed GLM failed (' + str(e) + '). Falling back to GEE...')
        try:
            gee_model = sm.GEE(df_model['Target'], exog_df, groups=df_model['QDA'], family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
            gee_result = gee_model.fit()
            pvals = gee_result.pvalues
            params = gee_result.params
            conf_int = gee_result.conf_int()
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            results_df = pd.DataFrame({'Coefficient': params, 'Odds Ratio': np.exp(params), 'OR CI Lower': np.exp(conf_int[0]), 'OR CI Upper': np.exp(conf_int[1]), 'p-value': pvals, 'FDR p-value': pvals_corrected, 'Significant (FDR < 0.05)': reject})
            print('\nMain-Effects Logistic Regression Results (GEE):')
            print(results_df.round(4))
            results_df.to_csv(os.path.join(data_dir, 'logistic_regression_results.csv'))
            preds_prob = gee_result.predict(exog_df)
            preds_class = (preds_prob > 0.5).astype(int)
            cm = confusion_matrix(df_model['Target'], preds_class)
            print('\nConfusion Matrix:')
            print(cm)
        except Exception as e2:
            print('\nGEE failed (' + str(e2) + '). Falling back to standard Logit...')
            exog_df_fixed = pd.get_dummies(df_model[['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'HIDDG', 'QDB', 'QDA']], drop_first=True, dtype=float)
            exog_df_fixed.insert(0, 'Intercept', 1.0)
            logit_model = sm.Logit(df_model['Target'], exog_df_fixed)
            logit_result = logit_model.fit(disp=0)
            pvals = logit_result.pvalues
            params = logit_result.params
            conf_int = logit_result.conf_int()
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            results_df = pd.DataFrame({'Coefficient': params, 'Odds Ratio': np.exp(params), 'OR CI Lower': np.exp(conf_int[0]), 'OR CI Upper': np.exp(conf_int[1]), 'p-value': pvals, 'FDR p-value': pvals_corrected, 'Significant (FDR < 0.05)': reject})
            print('\nMain-Effects Logistic Regression Results (Standard Logit):')
            print(results_df.round(4))
            results_df.to_csv(os.path.join(data_dir, 'logistic_regression_results.csv'))
            preds_prob = logit_result.predict(exog_df_fixed)
            preds_class = (preds_prob > 0.5).astype(int)
            cm = confusion_matrix(df_model['Target'], preds_class)
            print('\nConfusion Matrix:')
            print(cm)