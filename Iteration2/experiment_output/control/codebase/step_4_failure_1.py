# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import warnings
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'processed_data.csv')
    print('Loading dataset from ' + input_path + '...')
    df = pd.read_csv(input_path, low_memory=False)
    qea_2_col = next((c for c in df.columns if c.startswith('QEA_2')), None)
    qeb_2_col = next((c for c in df.columns if c.startswith('QEB_2')), None)
    qgp_cols = [c for c in df.columns if c.startswith('QGP_1') or c.startswith('QGP_2') or c.startswith('QGP_3')]
    qgu_cols = [c for c in df.columns if c.startswith('QGU_1') or c.startswith('QGU_2') or c.startswith('QGU_3')]
    if qea_2_col:
        df['Job_Security_Current'] = pd.to_numeric(df[qea_2_col], errors='coerce')
    if qeb_2_col:
        df['Job_Security_Future'] = pd.to_numeric(df[qeb_2_col], errors='coerce')
    if len(qgp_cols) == 3:
        df['Nature_of_Work_Change'] = df[qgp_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    if len(qgu_cols) == 3:
        df['Future_Nature_of_Work_Change'] = df[qgu_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    outcomes = ['Job_Security_Current', 'Job_Security_Future', 'Nature_of_Work_Change', 'Future_Nature_of_Work_Change']
    cat_prefixes = ['QDA', 'QDB', 'HidQDC', 'QDH', 'QDG', 'HIDDG', 'QDD', 'QGG', 'Global Employee Size', 'Global Annual Revenue', 'Market Capitalization']
    cat_cols = []
    for prefix in cat_prefixes:
        col = next((c for c in df.columns if c.startswith(prefix)), None)
        if col:
            cat_cols.append(col)
    print('One-hot encoding categorical columns: ' + ', '.join([c.split(':')[0] for c in cat_cols]))
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    numeric_cols = df_encoded.select_dtypes(include=[np.number, bool]).columns.tolist()
    for c in numeric_cols:
        if df_encoded[c].dtype == bool:
            df_encoded[c] = df_encoded[c].astype(int)
    exclude_prefixes = ['Respid', 'QEA_2', 'QEB_2', 'QGP_1', 'QGP_2', 'QGP_3', 'QGU_1', 'QGU_2', 'QGU_3', 'LCA_Class', 'Income_Min']
    exclude_cols = outcomes.copy()
    for prefix in exclude_prefixes:
        exclude_cols.extend([c for c in df_encoded.columns if c.startswith(prefix)])
    predictors = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('Unnamed')]
    print('Total predictors identified: ' + str(len(predictors)))
    print('\n' + '='*50)
    print('--- Bivariate Analysis (Spearman Rank Correlations) ---')
    print('='*50)
    corr_results = []
    for outcome in outcomes:
        if outcome not in df_encoded.columns:
            print('Outcome ' + outcome + ' not found in dataset. Skipping.')
            continue
        y = df_encoded[outcome]
        valid_y_mask = ~y.isna()
        pvals = []
        corrs = []
        valid_preds = []
        for pred in predictors:
            x = df_encoded[pred]
            valid_mask = valid_y_mask & ~x.isna()
            if valid_mask.sum() > 30:
                corr, pval = spearmanr(x[valid_mask], y[valid_mask])
                if not np.isnan(corr) and not np.isnan(pval):
                    corrs.append(corr)
                    pvals.append(pval)
                    valid_preds.append(pred)
        if len(pvals) > 0:
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            res_df = pd.DataFrame({'Outcome': outcome, 'Predictor': valid_preds, 'Spearman_rho': corrs, 'p_value': pvals, 'FDR_p_value': pvals_corrected})
            res_df['Abs_rho'] = res_df['Spearman_rho'].abs()
            res_df = res_df.sort_values('Abs_rho', ascending=False).drop(columns=['Abs_rho'])
            corr_results.append(res_df)
            print('\nTop 20 Predictors for ' + outcome + ':')
            print_df = res_df.head(20).copy()
            print_df['Predictor'] = print_df['Predictor'].apply(lambda x: x.split(':')[0] if ':' in x else x)
            print(print_df.to_string(index=False))
    if corr_results:
        all_corr_df = pd.concat(corr_results, ignore_index=True)
        corr_out_path = os.path.join(data_dir, 'bivariate_correlations.csv')
        all_corr_df.to_csv(corr_out_path, index=False)
        print('\nCorrelations saved to ' + corr_out_path)
    print('\n' + '='*50)
    print('--- Random Forest Feature Importance ---')
    print('='*50)
    rf_results = []
    rf_outcomes = ['Job_Security_Current', 'Job_Security_Future']
    for outcome in rf_outcomes:
        if outcome not in df_encoded.columns:
            continue
        print('\nTraining Random Forest for ' + outcome + '...')
        df_valid = df_encoded.dropna(subset=[outcome])
        y = df_valid[outcome].values
        X_df = df_valid[predictors]
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X_df)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
        cv_results = cross_validate(rf, X, y, cv=5, scoring='r2', return_estimator=False, n_jobs=8)
        mean_r2 = np.mean(cv_results['test_score'])
        print('5-fold CV Mean R^2: ' + str(round(mean_r2, 4)))
        rf.fit(X, y)
        print('Calculating permutation importance (this may take a moment)...')
        perm_importance = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=8)
        imp_df = pd.DataFrame({'Outcome': outcome, 'Predictor': predictors, 'Importance_Mean': perm_importance.importances_mean, 'Importance_Std': perm_importance.importances_std})
        imp_df = imp_df.sort_values('Importance_Mean', ascending=False)
        rf_results.append(imp_df)
        print('\nTop 20 Predictors by Permutation Importance for ' + outcome + ':')
        print_imp_df = imp_df.head(20).copy()
        print_imp_df['Predictor'] = print_imp_df['Predictor'].apply(lambda x: x.split(':')[0] if ':' in x else x)
        print(print_imp_df.to_string(index=False))
    if rf_results:
        all_rf_df = pd.concat(rf_results, ignore_index=True)
        rf_out_path = os.path.join(data_dir, 'rf_feature_importance.csv')
        all_rf_df.to_csv(rf_out_path, index=False)
        print('\nFeature importances saved to ' + rf_out_path)

if __name__ == '__main__':
    main()