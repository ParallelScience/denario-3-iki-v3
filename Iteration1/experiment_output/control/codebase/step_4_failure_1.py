# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

def main():
    data_dir = 'data/'
    data_path = os.path.join(data_dir, 'IKI_Cleaned_Features_LCA.csv')
    if not os.path.exists(data_path):
        print('Error: Data file ' + data_path + ' not found.')
        return
    df = pd.read_csv(data_path, low_memory=False)
    if 'LCA_Class' not in df.columns:
        print('Error: LCA_Class column not found in dataset.')
        return
    df = df.dropna(subset=['LCA_Class'])
    class_mapping = {2: 0, 1: 1, 3: 2}
    df['Target'] = df['LCA_Class'].map(class_mapping)
    placebo_cols = ['Global Employee Size', 'Market Capitalization']
    placebo_cols = [c for c in placebo_cols if c in df.columns]
    include_scale = False
    if placebo_cols:
        placebo_df = df[['Target'] + placebo_cols].dropna()
        if not placebo_df.empty:
            X_placebo = pd.get_dummies(placebo_df[placebo_cols], drop_first=True)
            X_placebo = sm.add_constant(X_placebo.astype(float))
            y_placebo = placebo_df['Target']
            try:
                mnlogit_placebo = sm.MNLogit(y_placebo, X_placebo).fit(disp=0)
                print('=== Placebo Test Results ===')
                print(mnlogit_placebo.summary())
                pvalues_placebo = mnlogit_placebo.pvalues.values.flatten()
                include_scale = any(pvalues_placebo < 0.05)
            except Exception as e:
                print('Placebo test failed: ' + str(e))
        else:
            print('Placebo dataframe is empty after dropping NaNs.')
    else:
        print('Placebo columns not found.')
    print('Include scale variables in main model: ' + str(include_scale))
    qkb_1_4_col = next((c for c in df.columns if c.startswith('QKB_1_4')), None)
    qkb_1_11_col = next((c for c in df.columns if c.startswith('QKB_1_11')), None)
    main_cols = ['QKB_Enablers_Index', 'QGO_Culture_Index', 'Positive_Affect', 'Negative_Affect', 'HIDDG', 'QDB', 'QDA', 'QDG', 'Baseline_Uncertainty']
    numeric_cols = ['QKB_Enablers_Index', 'QGO_Culture_Index', 'Positive_Affect', 'Negative_Affect', 'QDG', 'Baseline_Uncertainty']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if qkb_1_4_col:
        df[qkb_1_4_col] = pd.to_numeric(df[qkb_1_4_col], errors='coerce')
        main_cols.append(qkb_1_4_col)
    if qkb_1_11_col:
        df[qkb_1_11_col] = pd.to_numeric(df[qkb_1_11_col], errors='coerce')
        main_cols.append(qkb_1_11_col)
    if include_scale:
        main_cols.extend(placebo_cols)
    main_cols = [c for c in main_cols if c in df.columns]
    model_df = df[['Target'] + main_cols].dropna()
    if model_df.empty:
        print('Error: Main model dataframe is empty after dropping NaNs.')
        return
    X_main = pd.DataFrame(index=model_df.index)
    for c in numeric_cols:
        if c in model_df.columns:
            X_main[c] = model_df[c]
    X_main['Training'] = model_df[qkb_1_4_col] if qkb_1_4_col and qkb_1_4_col in model_df.columns else 0
    X_main['Involvement'] = model_df[qkb_1_11_col] if qkb_1_11_col and qkb_1_11_col in model_df.columns else 0
    if 'QGO_Culture_Index' in X_main.columns and 'QDG' in X_main.columns:
        X_main['Culture_x_JobLevel'] = X_main['QGO_Culture_Index'] * X_main['QDG']
    if 'Training' in X_main.columns and 'Baseline_Uncertainty' in X_main.columns:
        X_main['Training_x_Uncertainty'] = X_main['Training'] * X_main['Baseline_Uncertainty']
    if 'Training' in X_main.columns and 'Involvement' in X_main.columns:
        X_main['Training_x_Involvement'] = X_main['Training'] * X_main['Involvement']
    cat_cols = ['HIDDG', 'QDB', 'QDA']
    if include_scale:
        cat_cols.extend(placebo_cols)
    for col in cat_cols:
        if col in model_df.columns:
            dummies = pd.get_dummies(model_df[col], prefix=col, drop_first=True)
            X_main = pd.concat([X_main, dummies], axis=1)
    X_main = sm.add_constant(X_main.astype(float))
    y_main = model_df['Target']
    non_const_cols = [c for c in X_main.columns if c == 'const' or X_main[c].nunique() > 1]
    X_main = X_main[non_const_cols]
    variances = X_main.var()
    low_var_cols = variances[variances < 0.0001].index.tolist()
    if 'const' in low_var_cols:
        low_var_cols.remove('const')
    if low_var_cols:
        X_main = X_main.drop(columns=low_var_cols)
    corr_matrix = X_main.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        X_main = X_main.drop(columns=to_drop)
    try:
        mnlogit_main = sm.MNLogit(y_main, X_main).fit(disp=0, maxiter=1000)
    except Exception:
        try:
            mnlogit_main = sm.MNLogit(y_main, X_main).fit(disp=0, maxiter=1000, method='lbfgs')
        except Exception:
            mnlogit_main = sm.MNLogit(y_main, X_main).fit(disp=0, maxiter=1000, method='bfgs')
    params = mnlogit_main.params
    bse = mnlogit_main.bse
    pvalues = mnlogit_main.pvalues
    pvals_flat = pvalues.values.flatten()
    _, pvals_fdr_flat, _, _ = multipletests(pvals_flat, method='fdr_bh')
    pvals_fdr = pd.DataFrame(pvals_fdr_flat.reshape(pvalues.shape), index=pvalues.index, columns=pvalues.columns)
    results_list = []
    for col in params.columns:
        class_name = 'Resiliently Optimistic' if col == 0 else 'Anxiously Declining'
        for feature in params.index:
            coef = params.loc[feature, col]
            se = bse.loc[feature, col]
            pval = pvalues.loc[feature, col]
            pval_fdr = pvals_fdr.loc[feature, col]
            or_val = np.exp(np.clip(coef, -700, 700))
            ci_lower = np.exp(np.clip(coef - 1.95996 * se, -700, 700))
            ci_upper = np.exp(np.clip(coef + 1.95996 * se, -700, 700))
            results_list.append({'Class': class_name, 'Feature': feature, 'Coef': coef, 'Std_Err': se, 'Odds_Ratio': or_val, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper, 'P_Value': pval, 'FDR_P_Value': pval_fdr})
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(data_dir, 'MNLogit_Results.csv'), index=False)
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X_main.drop(columns=['const']), y_main, cv=cv, scoring='accuracy')
    print('5-fold CV Accuracy: ' + str(np.mean(cv_scores)))

if __name__ == '__main__':
    main()