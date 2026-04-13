# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

def find_col(df, prefix):
    for c in df.columns:
        if c.startswith(prefix):
            return c
    return None

def main():
    data_dir = 'data/'
    file_path = os.path.join(data_dir, 'processed_data.csv')
    print('Loading processed dataset...')
    df = pd.read_csv(file_path, low_memory=False)
    if 'LCA_Class' not in df.columns or df['LCA_Class'].isna().all():
        print('LCA_Class not found or empty. Recreating using KMeans (3 classes)...')
        qea2 = find_col(df, 'QEA_2:')
        qeb2 = find_col(df, 'QEB_2:')
        X_lca = df[[qea2, qeb2]].fillna(0)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['LCA_Class'] = kmeans.fit_predict(X_lca)
    else:
        print('LCA_Class found in dataset.')
    cols_to_keep = {'QDA': find_col(df, 'QDA:'), 'QDB': find_col(df, 'QDB:'), 'HIDDG': find_col(df, 'HIDDG:'), 'QKB_1_4': find_col(df, 'QKB_1_4:'), 'QKB_1_11': find_col(df, 'QKB_1_11:'), 'QKB_Index_Reduced': 'QKB_Index_Reduced', 'QGO_Index': 'QGO_Index', 'Positive_Affect': 'Positive_Affect', 'Negative_Affect': 'Negative_Affect', 'LCA_Class': 'LCA_Class'}
    df_reg = pd.DataFrame()
    for k, v in cols_to_keep.items():
        if v and v in df.columns:
            df_reg[k] = df[v]
        else:
            print('Warning: Column for ' + k + ' not found!')
    initial_len = len(df_reg)
    df_reg = df_reg.dropna().copy()
    print('Dropped ' + str(initial_len - len(df_reg)) + ' rows with missing values. Remaining: ' + str(len(df_reg)))
    num_cols = ['QKB_1_4', 'QKB_1_11', 'QKB_Index_Reduced', 'QGO_Index', 'Positive_Affect', 'Negative_Affect']
    for c in num_cols:
        df_reg[c] = pd.to_numeric(df_reg[c], errors='coerce')
    df_reg = df_reg.dropna().copy()
    df_reg['QKB_1_4_centered'] = df_reg['QKB_1_4'] - df_reg['QKB_1_4'].mean()
    df_reg['QKB_1_11_centered'] = df_reg['QKB_1_11'] - df_reg['QKB_1_11'].mean()
    df_reg['Interaction_QKB_4_11'] = df_reg['QKB_1_4_centered'] * df_reg['QKB_1_11_centered']
    df_reg = pd.get_dummies(df_reg, columns=['QDB', 'HIDDG'], drop_first=True, dtype=float)
    exclude_from_X = ['LCA_Class', 'QDA', 'QKB_1_4', 'QKB_1_11']
    X_cols = [c for c in df_reg.columns if c not in exclude_from_X]
    X = df_reg[X_cols]
    X = sm.add_constant(X)
    y_factorized, y_unique = pd.factorize(df_reg['LCA_Class'])
    print('\nLCA_Class mapping for MNLogit:')
    for i, val in enumerate(y_unique):
        print('  Class ' + str(i) + ' = ' + str(val))
    y = y_factorized
    print('\n--- Variance Inflation Factors (VIF) ---')
    try:
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        print(vif_data.to_string(index=False))
    except Exception as e:
        print('Error calculating VIF: ' + str(e))
    print('\n--- Multinomial Logistic Regression ---')
    groups = df_reg['QDA']
    group_ids = pd.factorize(groups)[0]
    model = sm.MNLogit(y, X)
    try:
        result = model.fit(cov_type='cluster', cov_kwds={'groups': group_ids}, disp=0, maxiter=100, method='bfgs')
        print(result.summary())
        if hasattr(result, 'prsquared'):
            print('McFadden\'s Pseudo R-squared: ' + str(result.prsquared))
    except Exception as e:
        print('Error fitting MNLogit with cluster-robust SE: ' + str(e))
        try:
            result = model.fit(disp=0, maxiter=100, method='bfgs')
            print(result.summary())
            if hasattr(result, 'prsquared'):
                print('McFadden\'s Pseudo R-squared: ' + str(result.prsquared))
        except Exception as e2:
            print('Error fitting MNLogit: ' + str(e2))
    if 'result' in locals():
        with open(os.path.join(data_dir, 'regression_summary.txt'), 'w') as f:
            f.write(result.summary().as_text())
            if hasattr(result, 'prsquared'):
                f.write('\nMcFadden\'s Pseudo R-squared: ' + str(result.prsquared) + '\n')
    print('\n--- Mediation Analysis (Bootstrap) ---')
    print('Path: QKB_Index_Reduced -> Negative_Affect -> LCA_Class')
    n_boot = 500
    indirect_effects = {c: [] for c in np.unique(y) if c != 0}
    X_med = X.copy()
    X_a = X_med.drop(columns=['Negative_Affect'])
    y_a = df_reg['Negative_Affect']
    X_b = X_med.copy()
    np.random.seed(42)
    successful_boots = 0
    for i in range(n_boot):
        idx = np.random.choice(len(df_reg), len(df_reg), replace=True)
        X_a_boot = X_a.iloc[idx]
        y_a_boot = y_a.iloc[idx]
        X_b_boot = X_b.iloc[idx]
        y_boot = y[idx]
        try:
            valid_cols = X_b_boot.columns[X_b_boot.var() > 1e-08]
            if 'QKB_Index_Reduced' not in valid_cols or 'Negative_Affect' not in valid_cols:
                continue
            X_a_boot_valid = X_a_boot[[c for c in valid_cols if c != 'Negative_Affect']]
            X_b_boot_valid = X_b_boot[valid_cols]
            model_a = sm.OLS(y_a_boot, X_a_boot_valid).fit()
            a_coeff = model_a.params['QKB_Index_Reduced']
            model_b = sm.MNLogit(y_boot, X_b_boot_valid).fit(disp=0, maxiter=50, method='newton')
            b_coeffs = model_b.params.loc['Negative_Affect']
            for c in b_coeffs.index:
                indirect_effects[c].append(a_coeff * b_coeffs[c])
            successful_boots += 1
        except:
            continue
    print('Successful bootstrap samples: ' + str(successful_boots) + ' / ' + str(n_boot))
    for c, effects in indirect_effects.items():
        if len(effects) > 0:
            effects = np.array(effects)
            mean_ie = np.mean(effects)
            ci_lower = np.percentile(effects, 2.5)
            ci_upper = np.percentile(effects, 97.5)
            print('Class ' + str(c) + ' relative to Class 0:')
            print('  Indirect Effect: ' + str(mean_ie))
            print('  95% CI: [' + str(ci_lower) + ', ' + str(ci_upper) + ']')
        else:
            print('Class ' + str(c) + ': Mediation analysis failed.')

if __name__ == '__main__':
    main()