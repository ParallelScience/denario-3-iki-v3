# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    raw_path = 'data/data_raw_encoded.csv'
    dummy_path = 'data/data_dummy_encoded.csv'
    print("Loading data...")
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_dummy = pd.read_csv(dummy_path, low_memory=False)
    if 'LCA_Class_Name' not in df_dummy.columns:
        raise ValueError("LCA_Class_Name not found in data. Please ensure Step 3 completed successfully.")
    valid_target = df_dummy['LCA_Class_Name'].notna()
    df_dummy = df_dummy[valid_target].copy()
    df_raw = df_raw[valid_target].copy()
    classes = df_dummy['LCA_Class_Name'].unique().tolist()
    if 'Stagnant Neutral' in classes:
        ref_class = 'Stagnant Neutral'
    else:
        ref_class = classes[0]
        print("Warning: 'Stagnant Neutral' not found. Using '" + str(ref_class) + "' as reference.")
    classes.remove(ref_class)
    class_mapping = {ref_class: 0}
    for i, c in enumerate(classes):
        class_mapping[c] = i + 1
    y = df_dummy['LCA_Class_Name'].map(class_mapping)
    qkb_11_cols = [c for c in df_dummy.columns if c.startswith('QKB_1_11:') and not c.endswith('_Not_Sure')]
    qkb_4_cols = [c for c in df_dummy.columns if c.startswith('QKB_1_4:') and not c.endswith('_Not_Sure')]
    qkb_11_col = qkb_11_cols[0] if qkb_11_cols else None
    qkb_4_col = qkb_4_cols[0] if qkb_4_cols else None
    if not qkb_11_col or not qkb_4_col:
        raise ValueError("QKB_1_11 or QKB_1_4 columns not found.")
    base_predictors = [qkb_11_col, qkb_4_col, 'Company_Culture_Index', 'Affective_Disposition_Factor_1', 'Affective_Disposition_Factor_2']
    hiddg_cols = [c for c in df_dummy.columns if c.startswith('HIDDG_')]
    qdb_cols = [c for c in df_dummy.columns if c.startswith('QDB_')]
    qda_cols = [c for c in df_dummy.columns if c.startswith('QDA_')]
    hiddg_cols = sorted(hiddg_cols)[1:] if len(hiddg_cols) > 1 else hiddg_cols
    qdb_cols = sorted(qdb_cols)[1:] if len(qdb_cols) > 1 else qdb_cols
    qda_cols = sorted(qda_cols)[1:] if len(qda_cols) > 1 else qda_cols
    all_predictors = base_predictors + hiddg_cols + qdb_cols + qda_cols
    X_raw = df_dummy[all_predictors].copy()
    sens_mask = df_raw[qkb_11_col].notna() & df_raw[qkb_4_col].notna() & df_raw['Company_Culture_Index'].notna()
    for col in X_raw.columns:
        if X_raw[col].isna().any():
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())
    for col in base_predictors:
        X_raw[col] = (X_raw[col] - X_raw[col].mean()) / (X_raw[col].std() + 1e-09)
    X_raw['Interaction_QKB11_QKB4'] = X_raw[qkb_11_col] * X_raw[qkb_4_col]
    X = sm.add_constant(X_raw)
    X = X.astype(float)
    print("\nCalculating VIFs...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vifs = []
    for i in range(X.shape[1]):
        try:
            vifs.append(variance_inflation_factor(X.values, i))
        except:
            vifs.append(np.nan)
    vif_data["VIF"] = vifs
    print(vif_data[vif_data['Feature'] != 'const'].sort_values('VIF', ascending=False).head(10).to_string(index=False))
    print("\nFitting Multinomial Logistic Regression (Full Model)...")
    model = sm.MNLogit(y, X)
    try:
        result = model.fit(method='bfgs', maxiter=2000, disp=False)
        if not getattr(result, 'converged', True):
            print("BFGS did not converge, trying newton...")
            result = model.fit(method='newton', maxiter=1000, disp=False)
    except Exception as e:
        print("BFGS failed (" + str(e) + "), trying newton...")
        try:
            result = model.fit(method='newton', maxiter=1000, disp=False)
        except Exception as e2:
            print("Newton failed (" + str(e2) + "), trying lbfgs...")
            result = model.fit(method='lbfgs', maxiter=2000, disp=False)
    print("Model converged: " + str(getattr(result, 'converged', 'Unknown')))
    summary_dfs = []
    for i, c in enumerate(classes):
        coef = result.params.iloc[:, i]
        stderr = result.bse.iloc[:, i]
        z = result.tvalues.iloc[:, i]
        p = result.pvalues.iloc[:, i]
        df_res = pd.DataFrame({'Class': c, 'Feature': X.columns, 'Coef': coef, 'StdErr': stderr, 'z': z, 'P_value': p})
        summary_dfs.append(df_res)
    full_summary = pd.concat(summary_dfs, ignore_index=True)
    _, fdr_p, _, _ = multipletests(full_summary['P_value'], method='fdr_bh')
    full_summary['FDR_P_value'] = fdr_p
    full_summary['OR'] = np.exp(full_summary['Coef'])
    full_summary['CI_Lower'] = np.exp(full_summary['Coef'] - 1.96 * full_summary['StdErr'])
    full_summary['CI_Upper'] = np.exp(full_summary['Coef'] + 1.96 * full_summary['StdErr'])
    out_csv = 'data/mnlogit_results.csv'
    full_summary.to_csv(out_csv, index=False)
    print("\nFull regression table saved to " + out_csv)
    key_features = ['Interaction_QKB11_QKB4', qkb_11_col, qkb_4_col, 'Company_Culture_Index', 'Affective_Disposition_Factor_1', 'Affective_Disposition_Factor_2']
    print("\n--- Key Predictors Results ---")
    key_res = full_summary[full_summary['Feature'].isin(key_features)].copy()
    key_res = key_res[['Class', 'Feature', 'Coef', 'OR', 'CI_Lower', 'CI_Upper', 'FDR_P_value']]
    for col in ['Coef', 'OR', 'CI_Lower', 'CI_Upper', 'FDR_P_value']:
        key_res[col] = key_res[col].apply(lambda x: str(round(x, 4)) if pd.notnull(x) else "NaN")
    print(key_res.to_string(index=False))
    print("\nPerforming 5-fold Cross-Validation...")
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_cv = X.drop(columns=['const'])
    scores = cross_val_score(clf, X_cv, y, cv=cv, scoring='accuracy')
    print("5-fold CV Accuracy: " + str(round(scores.mean(), 4)) + " (+/- " + str(round(scores.std() * 2, 4)) + ")")
    print("\nPerforming Sensitivity Analysis (excluding 'Not sure' / missing responses)...")
    X_sens = X[sens_mask]
    y_sens = y[sens_mask]
    print("Full model N: " + str(len(X)) + ", Sensitivity model N: " + str(len(X_sens)))
    if len(y_sens.unique()) < len(y.unique()):
        print("Warning: Sensitivity analysis drops some classes entirely. MNLogit might fail.")
    if 0 not in y_sens.unique():
        print("Warning: Reference class (0) not in sensitivity data. Comparison might be invalid.")
    model_sens = sm.MNLogit(y_sens, X_sens)
    try:
        result_sens = model_sens.fit(method='bfgs', maxiter=2000, disp=False)
        if not getattr(result_sens, 'converged', True):
            result_sens = model_sens.fit(method='newton', maxiter=1000, disp=False)
    except Exception as e:
        try:
            result_sens = model_sens.fit(method='newton', maxiter=1000, disp=False)
        except Exception as e2:
            result_sens = model_sens.fit(method='lbfgs', maxiter=2000, disp=False)
    print("\n--- Sensitivity Analysis Comparison (Coefficients) ---")
    sens_unique_y = sorted([val for val in y_sens.unique() if val != 0])
    for i, c in enumerate(classes):
        print("\nClass: " + str(c))
        coef_full = result.params.iloc[:, i]
        if class_mapping[c] in sens_unique_y:
            sens_idx = sens_unique_y.index(class_mapping[c])
            coef_sens = result_sens.params.iloc[:, sens_idx]
        else:
            coef_sens = pd.Series(np.nan, index=coef_full.index)
        comp_df = pd.DataFrame({'Feature': key_features, 'Full_Coef': [coef_full[f] for f in key_features], 'Sens_Coef': [coef_sens[f] for f in key_features]})
        comp_df['Diff'] = comp_df['Sens_Coef'] - comp_df['Full_Coef']
        for col in ['Full_Coef', 'Sens_Coef', 'Diff']:
            comp_df[col] = comp_df[col].apply(lambda x: str(round(x, 4)) if pd.notnull(x) else "NaN")
        print(comp_df.to_string(index=False))