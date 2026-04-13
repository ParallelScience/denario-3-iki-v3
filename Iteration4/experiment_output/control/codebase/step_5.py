# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
import warnings

warnings.filterwarnings('ignore')

def fit_model(y, X):
    model = sm.MNLogit(y, X)
    try:
        res = model.fit(disp=False, method='bfgs', maxiter=5000)
        if np.isnan(res.llf):
            raise ValueError('NaN log-likelihood')
        return res
    except:
        try:
            res = model.fit(disp=False, method='newton', maxiter=5000)
            if np.isnan(res.llf):
                raise ValueError('NaN log-likelihood')
            return res
        except:
            return model.fit_regularized(disp=False, maxiter=5000, alpha=0.1, L1_wt=0.0)

def main():
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'processed_data.csv')
    print('Loading dataset from ' + input_path + '...')
    df = pd.read_csv(input_path, low_memory=False)
    qdb_col = next((c for c in df.columns if c.startswith('QDB')), 'QDB')
    def is_high_complex(x):
        if pd.isna(x): return 0
        x_str = str(x).lower()
        if 'high tech' in x_str or 'life sciences' in x_str or 'financial services' in x_str:
            return 1
        return 0
    df['High_Complexity'] = df[qdb_col].apply(is_high_complex)
    print('\n--- Demographic Composition of "Anxiously Declining" Class ---')
    not_sure_class = 'Anxiously Declining'
    df_ns = df[df['LCA_Class'] == not_sure_class]
    if len(df_ns) > 0:
        print('Total respondents in ' + not_sure_class + ': ' + str(len(df_ns)))
        qdg_col = next((c for c in df.columns if c.lower() == 'qdg'), None)
        if qdg_col:
            print('\nJob Level (QDG):')
            print((df_ns[qdg_col].value_counts(normalize=True) * 100).round(1).astype(str) + '%')
        qdh_col = next((c for c in df.columns if c.lower() == 'qdh'), None)
        if qdh_col:
            print('\nDepartment (QDH):')
            print((df_ns[qdh_col].value_counts(normalize=True) * 100).round(1).astype(str) + '%')
        gen_col = next((c for c in df.columns if c.lower() == 'hidqdc'), None)
        if gen_col:
            print('\nGeneration (HidQDC):')
            print((df_ns[gen_col].value_counts(normalize=True) * 100).round(1).astype(str) + '%')
    else:
        print('No respondents found in ' + not_sure_class + ' class.')
    qkb_1_4_col = next((c for c in df.columns if c.startswith('QKB_1_4')), 'QKB_1_4')
    qkb_1_11_col = next((c for c in df.columns if c.startswith('QKB_1_11')), 'QKB_1_11')
    hiddg_col = next((c for c in df.columns if c.startswith('HIDDG')), 'HIDDG')
    df_model = df.dropna(subset=['LCA_Class']).copy()
    df_model['Positive_Affect'] = df_model['Positive_Affect'].fillna(0)
    df_model['Negative_Affect'] = df_model['Negative_Affect'].fillna(0)
    df_model['HIDDG'] = df_model[hiddg_col].fillna('Unknown')
    df_model['QKB_1_4'] = df_model[qkb_1_4_col].fillna(df_model[qkb_1_4_col].median())
    df_model['QKB_1_11'] = df_model[qkb_1_11_col].fillna(df_model[qkb_1_11_col].median())
    df_model = pd.get_dummies(df_model, columns=['HIDDG'], drop_first=True, dtype=float)
    hiddg_dummies = [c for c in df_model.columns if c.startswith('HIDDG_')]
    df_model['QKB_1_4_x_QKB_1_11'] = df_model['QKB_1_4'] * df_model['QKB_1_11']
    class_mapping = {'Stagnant Neutral': 0, 'Resiliently Optimistic': 1, 'Anxiously Declining': 2}
    y = df_model['LCA_Class'].map(class_mapping)
    X0_cols = ['Positive_Affect', 'Negative_Affect', 'High_Complexity', 'QKB_1_4', 'QKB_1_11', 'QKB_1_4_x_QKB_1_11'] + hiddg_dummies
    X0 = sm.add_constant(df_model[X0_cols])
    df_model['HC_x_QKB_1_4'] = df_model['High_Complexity'] * df_model['QKB_1_4']
    df_model['HC_x_QKB_1_11'] = df_model['High_Complexity'] * df_model['QKB_1_11']
    df_model['HC_x_QKB_1_4_x_QKB_1_11'] = df_model['High_Complexity'] * df_model['QKB_1_4_x_QKB_1_11']
    X1_cols = X0_cols + ['HC_x_QKB_1_4', 'HC_x_QKB_1_11', 'HC_x_QKB_1_4_x_QKB_1_11']
    X1 = sm.add_constant(df_model[X1_cols])
    print('\n--- Likelihood Ratio Test: Sector Complexity Interactions ---')
    try:
        model0 = fit_model(y, X0)
        llf0 = model0.llf if hasattr(model0, 'llf') else np.nan
        df0 = model0.df_model if hasattr(model0, 'df_model') else np.nan
        model1 = fit_model(y, X1)
        llf1 = model1.llf if hasattr(model1, 'llf') else np.nan
        df1 = model1.df_model if hasattr(model1, 'df_model') else np.nan
        if not np.isnan(llf0) and not np.isnan(llf1):
            lr_stat = 2 * (llf1 - llf0)
            df_diff = df1 - df0
            if df_diff <= 0: df_diff = 1
            p_val = chi2.sf(lr_stat, df_diff)
            print('Null Model Log-Likelihood: ' + str(round(llf0, 2)) + ' (df=' + str(df0) + ')')
            print('Alt Model Log-Likelihood:  ' + str(round(llf1, 2)) + ' (df=' + str(df1) + ')')
            print('LR Statistic: ' + str(round(lr_stat, 2)))
            print('Degrees of Freedom: ' + str(df_diff))
            print('p-value: ' + str(round(p_val, 6)))
        else:
            print('LRT could not be computed due to regularized fit (no valid log-likelihood).')
            lr_stat, p_val, df_diff = np.nan, np.nan, np.nan
    except Exception as e:
        print('Error during LRT: ' + str(e))
        lr_stat, p_val, df_diff = np.nan, np.nan, np.nan
    print('\n--- Average Marginal Effects (AMEs) for Dual-Pillar Predictors ---')
    df_model['QDB'] = df_model[qdb_col].fillna('Unknown')
    df_model_full = pd.get_dummies(df_model, columns=['QDB'], drop_first=True, dtype=float)
    qdb_dummies = [c for c in df_model_full.columns if c.startswith('QDB_')]
    X_full_cols = ['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'QKB_1_4_x_QKB_1_11'] + hiddg_dummies + qdb_dummies
    X_full = sm.add_constant(df_model_full[X_full_cols])
    cols_to_drop = [col for col in X_full.columns if col != 'const' and X_full[col].nunique() <= 1]
    X_full = X_full.drop(columns=cols_to_drop)
    try:
        model_full = fit_model(y, X_full)
        preds = model_full.predict(X_full)
        params = model_full.params
        if isinstance(params, pd.DataFrame):
            params_vals = params.values
        else:
            params_vals = params
        params_full = np.column_stack((np.zeros(params_vals.shape[0]), params_vals))
        feature_names = X_full.columns.tolist()
        idx_q4 = feature_names.index('QKB_1_4')
        idx_q11 = feature_names.index('QKB_1_11')
        idx_int = feature_names.index('QKB_1_4_x_QKB_1_11')
        N = len(X_full)
        n_classes = preds.shape[1]
        ame_q4 = np.zeros(n_classes)
        ame_q11 = np.zeros(n_classes)
        for i in range(N):
            p_i = preds.iloc[i].values
            q11_val = X_full.iloc[i]['QKB_1_11']
            q4_val = X_full.iloc[i]['QKB_1_4']
            beta_eff_q4 = params_full[idx_q4, :] + params_full[idx_int, :] * q11_val
            beta_eff_q11 = params_full[idx_q11, :] + params_full[idx_int, :] * q4_val
            exp_beta_q4 = np.sum(p_i * beta_eff_q4)
            exp_beta_q11 = np.sum(p_i * beta_eff_q11)
            ame_q4 += p_i * (beta_eff_q4 - exp_beta_q4)
            ame_q11 += p_i * (beta_eff_q11 - exp_beta_q11)
        ame_q4 /= N
        ame_q11 /= N
        class_names_dict = {0: 'Stagnant Neutral', 1: 'Resiliently Optimistic', 2: 'Anxiously Declining'}
        ame_results = []
        for c in range(n_classes):
            c_name = class_names_dict.get(c, str(c))
            ame_results.append({'Class': c_name, 'Predictor': 'QKB_1_4 (Regular Training)', 'AME': ame_q4[c]})
            ame_results.append({'Class': c_name, 'Predictor': 'QKB_1_11 (Employee Involvement)', 'AME': ame_q11[c]})
        ame_df = pd.DataFrame(ame_results)
        print(ame_df.round(5).to_string(index=False))
        ame_path = os.path.join(data_dir, 'average_marginal_effects.csv')
        ame_df.to_csv(ame_path, index=False)
        print('\nAME estimates saved to ' + ame_path)
    except Exception as e:
        print('Error calculating AMEs: ' + str(e))
    diag_stats = {'LRT_Statistic': lr_stat if 'lr_stat' in locals() else np.nan, 'LRT_pvalue': p_val if 'p_val' in locals() else np.nan, 'LRT_df': df_diff if 'df_diff' in locals() else np.nan}
    diag_df = pd.DataFrame([diag_stats])
    diag_path = os.path.join(data_dir, 'sensitivity_diagnostics.csv')
    diag_df.to_csv(diag_path, index=False)
    print('Diagnostic statistics saved to ' + diag_path)

if __name__ == '__main__':
    main()