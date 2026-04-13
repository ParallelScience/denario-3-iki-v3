# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

os.environ['OMP_NUM_THREADS'] = '1'

def perform_mediation_bootstrap(df, x_col, m_col, y_col, n_boot=1000):
    data = df[[x_col, m_col, y_col]].dropna()
    X = data[x_col].values
    M = data[m_col].values
    Y = data[y_col].values
    n = len(data)
    cov_xm = np.cov(X, M)[0, 1]
    var_x = np.var(X, ddof=1)
    a = cov_xm / var_x
    X_mat = np.column_stack((X, M, np.ones(n)))
    coeffs = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    c_prime = coeffs[0]
    b = coeffs[1]
    cov_xy = np.cov(X, Y)[0, 1]
    c = cov_xy / var_x
    indirect_effect = a * b
    boot_indirect = np.zeros(n_boot)
    np.random.seed(42)
    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        X_b = X[idx]
        M_b = M[idx]
        Y_b = Y[idx]
        var_x_b = np.var(X_b, ddof=1)
        if var_x_b == 0:
            continue
        a_b = np.cov(X_b, M_b)[0, 1] / var_x_b
        X_mat_b = np.column_stack((X_b, M_b, np.ones(n)))
        coeffs_b = np.linalg.lstsq(X_mat_b, Y_b, rcond=None)[0]
        b_b = coeffs_b[1]
        boot_indirect[i] = a_b * b_b
    ci_lower = np.percentile(boot_indirect, 2.5)
    ci_upper = np.percentile(boot_indirect, 97.5)
    return {'a': a, 'b': b, 'c_prime': c_prime, 'c': c, 'indirect': indirect_effect, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

def print_subgroup_analysis(df, group_col, target_col, data_dir):
    if group_col not in df.columns or target_col not in df.columns:
        return
    data = df[[group_col, target_col]].dropna()
    grouped = data.groupby(group_col)[target_col].agg(['mean', 'count', 'std'])
    grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['std'] / np.sqrt(grouped['count'])
    grouped = grouped.sort_values(by='mean', ascending=False)
    grouped.to_csv(os.path.join(data_dir, 'subgroup_' + target_col + '_' + group_col + '.csv'))

def main():
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_dataset_step2.csv'), low_memory=False)
    raw_df = pd.read_csv('/home/node/work/projects/iki_v2/IKI-Data-Raw.csv', sep='\t', low_memory=False)
    for col in ['HidQDC', 'HIDDG', 'QDB', 'QDA']:
        if col not in df.columns and col in raw_df.columns:
            df[col] = raw_df[col]
    targets = {'Current': 'Job_Security_Index_Current', 'Future': 'Job_Security_Index_Future'}
    potential_num_cols = ['QDC', 'QDE_Year', 'QC', 'Income_Rank', 'QGI', 'QGS', 'Organizational_Support_Index', 'Positive_Affect', 'Negative_Affect', 'Nature_of_Work_Change_Index', 'Future_Nature_of_Work_Change_Index', 'Task_Transformation_Repetitive', 'Task_Transformation_Creative', 'Task_Transformation_Complex']
    prefixes = ['QHD_', 'QGM_', 'QF_', 'QKC_', 'QED_1_', 'QKB_1_', 'QKB_2_', 'QGO_']
    for col in df.columns:
        if any(col.startswith(p) for p in prefixes):
            potential_num_cols.append(col)
    potential_cat_cols = ['QDA', 'QDB', 'HidQDC', 'QDD', 'QDH', 'QDG', 'HIDDG', 'Global Employee Size', 'Global Annual Revenue', 'Market Capitalization', 'QGG', 'QGN', 'QA_1_1', 'QKD']
    num_cols = [c for c in potential_num_cols if c in df.columns and c not in targets.values()]
    cat_cols = [c for c in potential_cat_cols if c in df.columns and c not in targets.values()]
    num_cols = list(set(num_cols))
    cat_cols = list(set(cat_cols))
    for name, target_col in targets.items():
        if target_col not in df.columns:
            continue
        valid_idx = df[target_col].notna()
        df_valid = df[valid_idx]
        X = df_valid[num_cols + cat_cols]
        y = df_valid[target_col]
        numeric_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rf)])
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X, y, cv=cv, scoring=['r2', 'neg_mean_squared_error'], n_jobs=1)
        model.fit(X, y)
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=8)
        importances = result.importances_mean
        feature_names = num_cols + cat_cols
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances, 'Std': result.importances_std})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
        imp_df.to_csv(os.path.join(data_dir, 'rf_importance_' + name.lower() + '.csv'), index=False)
    mediation_results = []
    if 'Organizational_Support_Index' in df.columns and 'Nature_of_Work_Change_Index' in df.columns and 'Job_Security_Index_Current' in df.columns:
        res_curr = perform_mediation_bootstrap(df, 'Organizational_Support_Index', 'Nature_of_Work_Change_Index', 'Job_Security_Index_Current', n_boot=1000)
        res_curr['Model'] = 'Current'
        mediation_results.append(res_curr)
    if 'Organizational_Support_Index' in df.columns and 'Future_Nature_of_Work_Change_Index' in df.columns and 'Job_Security_Index_Future' in df.columns:
        res_fut = perform_mediation_bootstrap(df, 'Organizational_Support_Index', 'Future_Nature_of_Work_Change_Index', 'Job_Security_Index_Future', n_boot=1000)
        res_fut['Model'] = 'Future'
        mediation_results.append(res_fut)
    if mediation_results:
        pd.DataFrame(mediation_results).to_csv(os.path.join(data_dir, 'mediation_results.csv'), index=False)
    subgroup_cols = ['HidQDC', 'HIDDG', 'QDB', 'QDA']
    for target in ['Job_Security_Index_Current', 'Job_Security_Index_Future']:
        for col in subgroup_cols:
            print_subgroup_analysis(df, col, target, data_dir)

if __name__ == '__main__':
    main()