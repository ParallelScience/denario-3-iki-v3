# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", message="Inverting hessian failed")
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'processed_data.csv')
    print('Loading dataset from ' + input_path + '...')
    df = pd.read_csv(input_path, low_memory=False)
    qkb_1_4_cols = [c for c in df.columns if c.startswith('QKB_1_4')]
    qkb_1_11_cols = [c for c in df.columns if c.startswith('QKB_1_11')]
    hiddg_cols = [c for c in df.columns if c.startswith('HIDDG')]
    qdb_cols = [c for c in df.columns if c.startswith('QDB')]
    if not qkb_1_4_cols or not qkb_1_11_cols or not hiddg_cols or not qdb_cols:
        print('Error: Required columns not found.')
        sys.exit(1)
    qkb_1_4_col = qkb_1_4_cols[0]
    qkb_1_11_col = qkb_1_11_cols[0]
    hiddg_col = hiddg_cols[0]
    qdb_col = qdb_cols[0]
    df.rename(columns={qkb_1_4_col: 'QKB_1_4', qkb_1_11_col: 'QKB_1_11', hiddg_col: 'HIDDG', qdb_col: 'QDB'}, inplace=True)
    cols_to_keep = ['LCA_Class', 'Positive_Affect', 'Negative_Affect', 'HIDDG', 'QDB', 'QKB_1_4', 'QKB_1_11']
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        print('Error: Missing columns: ' + str(missing_cols))
        sys.exit(1)
    df_model = df.dropna(subset=['LCA_Class'])[cols_to_keep].copy()
    df_model['Positive_Affect'] = df_model['Positive_Affect'].fillna(0)
    df_model['Negative_Affect'] = df_model['Negative_Affect'].fillna(0)
    df_model['HIDDG'] = df_model['HIDDG'].fillna('Unknown')
    df_model['QDB'] = df_model['QDB'].fillna('Unknown')
    df_model['QKB_1_4'] = df_model['QKB_1_4'].fillna(df_model['QKB_1_4'].median())
    df_model['QKB_1_11'] = df_model['QKB_1_11'].fillna(df_model['QKB_1_11'].median())
    df_model = pd.get_dummies(df_model, columns=['HIDDG', 'QDB'], drop_first=True, dtype=float)
    df_model['QKB_1_4_x_QKB_1_11'] = df_model['QKB_1_4'] * df_model['QKB_1_11']
    X_cols = [c for c in df_model.columns if c != 'LCA_Class']
    X = df_model[X_cols]
    X = sm.add_constant(X)
    cols_to_drop = []
    for col in X.columns:
        if col != 'const' and X[col].nunique() <= 1:
            cols_to_drop.append(col)
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vifs = []
    for i in range(len(X.columns)):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        vifs.append(vif)
    vif_data['VIF'] = vifs
    vif_path = os.path.join(data_dir, 'vif_diagnostics.csv')
    vif_data.to_csv(vif_path, index=False)
    class_mapping = {'Stagnant Neutral': 0, 'Resiliently Optimistic': 1, 'Anxiously Declining': 2}
    y = df_model['LCA_Class'].map(class_mapping)
    unique_classes = np.sort(y.unique())
    model = sm.MNLogit(y, X)
    try:
        result = model.fit_regularized(maxiter=5000, alpha=0.1, L1_wt=0.0, disp=False)
    except Exception:
        result = model.fit(maxiter=5000, method='bfgs', disp=False)
    if hasattr(result, 'summary'):
        try:
            print('\n--- Multinomial Logistic Regression Summary ---')
            print(result.summary())
        except Exception:
            print('\n--- Multinomial Logistic Regression Summary ---')
            print('Summary generation failed (likely due to singular Hessian).')
    else:
        print('\n--- Multinomial Logistic Regression Summary ---')
        print('Regularized fit used. Summary not available.')
    print('\n--- Multinomial Logistic Regression Coefficients ---')
    all_res_dfs = []
    modeled_classes = [c for c in unique_classes if c != 0]
    try:
        params_df = pd.DataFrame(result.params)
        if params_df.shape[1] == 1 and len(modeled_classes) > 1:
            modeled_classes = [modeled_classes[0]]
        bse_df = pd.DataFrame(result.bse) if hasattr(result, 'bse') and result.bse is not None else pd.DataFrame(np.nan, index=params_df.index, columns=params_df.columns)
        tvalues_df = pd.DataFrame(result.tvalues) if hasattr(result, 'tvalues') and result.tvalues is not None else pd.DataFrame(np.nan, index=params_df.index, columns=params_df.columns)
        pvalues_df = pd.DataFrame(result.pvalues) if hasattr(result, 'pvalues') and result.pvalues is not None else pd.DataFrame(np.nan, index=params_df.index, columns=params_df.columns)
        for i, class_val in enumerate(modeled_classes):
            class_name = {1: 'Resiliently Optimistic', 2: 'Anxiously Declining'}.get(class_val, str(class_val))
            print('\nEquation for ' + class_name + ' (Reference: Stagnant Neutral):')
            col_idx = i if i < params_df.shape[1] else 0
            coef = params_df.iloc[:, col_idx]
            se = bse_df.iloc[:, col_idx] if col_idx < bse_df.shape[1] else pd.Series(np.nan, index=coef.index)
            z = tvalues_df.iloc[:, col_idx] if col_idx < tvalues_df.shape[1] else pd.Series(np.nan, index=coef.index)
            p = pvalues_df.iloc[:, col_idx] if col_idx < pvalues_df.shape[1] else pd.Series(np.nan, index=coef.index)
            or_val = np.exp(coef)
            res_df = pd.DataFrame({'Coef': coef, 'Std.Err': se, 'z': z, 'P>|z|': p, 'Odds Ratio': or_val})
            print(res_df.round(4).to_string())
            res_df['Class'] = class_name
            res_df['Feature'] = res_df.index
            all_res_dfs.append(res_df)
    except Exception as e:
        print('Error extracting coefficients: ' + str(e))
    if all_res_dfs:
        full_res_df = pd.concat(all_res_dfs, ignore_index=True)
        res_path = os.path.join(data_dir, 'mnlogit_results.csv')
        full_res_df.to_csv(res_path, index=False)
    print('\n--- Predicted Probabilities for Policy-Lift Combinations ---')
    try:
        means = X.mean()
        scenarios = []
        scenario_names = ['Low/Low (1,1)', 'High/Low (5,1)', 'Low/High (1,5)', 'High/High (5,5)']
        for q4, q11 in [(1, 1), (5, 1), (1, 5), (5, 5)]:
            scenario = means.copy()
            if 'QKB_1_4' in scenario: scenario['QKB_1_4'] = q4
            if 'QKB_1_11' in scenario: scenario['QKB_1_11'] = q11
            if 'QKB_1_4_x_QKB_1_11' in scenario: scenario['QKB_1_4_x_QKB_1_11'] = q4 * q11
            scenarios.append(scenario)
        scenarios_df = pd.DataFrame(scenarios)
        preds = result.predict(scenarios_df)
        preds.index = scenario_names
        pred_col_names = []
        for c in unique_classes:
            if c == 0: pred_col_names.append('Stagnant Neutral')
            elif c == 1: pred_col_names.append('Resiliently Optimistic')
            elif c == 2: pred_col_names.append('Anxiously Declining')
            else: pred_col_names.append(str(c))
        if preds.shape[1] == len(pred_col_names): preds.columns = pred_col_names
        print(preds.round(4).to_string())
        preds_path = os.path.join(data_dir, 'policy_lift_probabilities.csv')
        preds.to_csv(preds_path)
    except Exception as e:
        print('Error computing policy lift probabilities: ' + str(e))

if __name__ == '__main__':
    main()