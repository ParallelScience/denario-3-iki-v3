# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
def map_likert(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'not sure' in s or "don't know" in s: return np.nan
    if 'significantly negative' in s: return -2
    if 'slightly negative' in s: return -1
    if 'no impact' in s: return 0
    if 'slightly positive' in s: return 1
    if 'significantly positive' in s: return 2
    if 'large decrease' in s: return -2
    if 'moderate decrease' in s: return -1
    if 'no major change' in s: return 0
    if 'moderate increase' in s: return 1
    if 'large increase' in s: return 2
    if 'not at all important' in s: return -2
    if 'slightly important' in s: return -1
    if 'moderately important' in s: return 0
    if 'very important' in s: return 1
    if 'extremely important' in s: return 2
    try:
        v = float(s)
        if 1 <= v <= 5:
            return int(v) - 3
    except ValueError:
        pass
    return np.nan
def compute_vif(X):
    X_vif = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vifs = []
    for i in range(X_vif.shape[1]):
        try:
            vifs.append(variance_inflation_factor(X_vif.values, i))
        except:
            vifs.append(np.nan)
    vif_data["VIF"] = vifs
    return vif_data
def print_model_results(res, title):
    print("=== " + title + " ===")
    try:
        pvals_fdr = multipletests(res.pvalues, method='fdr_bh')[1]
        res_df = pd.DataFrame({'Coef': res.params, 'Std.Err': res.bse, 'z': res.tvalues, 'P>|z|': res.pvalues, 'FDR_P': pvals_fdr, 'CI_Lower': res.conf_int()[0], 'CI_Upper': res.conf_int()[1]})
        print(res_df.to_string())
    except Exception as e:
        print("Error formatting results: " + str(e))
        print(res.summary())
    print("\n")
def main():
    data_dir = "data/"
    df = pd.read_csv(os.path.join(data_dir, "cleaned_dataset_step2.csv"), low_memory=False)
    df_raw = pd.read_csv('/home/node/work/projects/iki_v2/IKI-Data-Raw.csv', sep='\t', low_memory=False)
    def find_col(prefix):
        return next((col for col in df_raw.columns if col.startswith(prefix)), None)
    qea_2_col = find_col('QEA_2')
    qeb_2_col = find_col('QEB_2')
    qkb_1_4_col = find_col('QKB_1_4')
    qkb_1_11_col = find_col('QKB_1_11')
    qgp_cols = [find_col(f'QGP_{i}') for i in range(1, 4)]
    qgu_cols = [find_col(f'QGU_{i}') for i in range(1, 4)]
    if qea_2_col:
        df['Not_Sure_Current'] = df_raw[qea_2_col].astype(str).str.lower().str.contains('not sure').astype(int)
        df['Job_Security_Index_Current'] = df_raw[qea_2_col].apply(map_likert)
    if qeb_2_col:
        df['Not_Sure_Future'] = df_raw[qeb_2_col].astype(str).str.lower().str.contains('not sure').astype(int)
        df['Job_Security_Index_Future'] = df_raw[qeb_2_col].apply(map_likert)
    if qkb_1_4_col: df['QKB_1_4'] = df_raw[qkb_1_4_col].apply(map_likert)
    if qkb_1_11_col: df['QKB_1_11'] = df_raw[qkb_1_11_col].apply(map_likert)
    if all(qgp_cols):
        df['Nature_of_Work_Change_Index'] = df_raw[qgp_cols].map(map_likert).mean(axis=1)
    if all(qgu_cols):
        df['Future_Nature_of_Work_Change_Index'] = df_raw[qgu_cols].map(map_likert).mean(axis=1)
    mean_org_sure = df.loc[df['Not_Sure_Current'] == 0, 'Organizational_Support_Index'].mean()
    mean_org_notsure = df.loc[df['Not_Sure_Current'] == 1, 'Organizational_Support_Index'].mean()
    print("=== Organizational Support Index Comparison ===")
    print("Mean Org Support (Sure about Current Job Security): {:.4f}".format(mean_org_sure))
    print("Mean Org Support (Not Sure about Current Job Security): {:.4f}".format(mean_org_notsure))
    print("\n")
    cat_cols = ['HIDDG', 'QDB']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)
        else:
            df[c] = 'Unknown'
    num_cols = ['Organizational_Support_Index', 'Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'Nature_of_Work_Change_Index', 'Future_Nature_of_Work_Change_Index']
    missing_num_cols = [c for c in num_cols if c not in df.columns]
    if missing_num_cols:
        for c in missing_num_cols:
            df[c] = 0
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    if 'QKD' in df.columns:
        df['QKD'] = df['QKD'].fillna(0)
    else:
        df['QKD'] = 0
    X_base = pd.DataFrame()
    X_base['Org_Support'] = df['Organizational_Support_Index']
    X_base['Pos_Affect'] = df['Positive_Affect']
    X_base['Neg_Affect'] = df['Negative_Affect']
    X_base['QKB_1_4'] = df['QKB_1_4']
    X_base['QKB_1_11'] = df['QKB_1_11']
    X_base['QKB_Interaction'] = df['QKB_1_4'] * df['QKB_1_11']
    X_base['QKD'] = df['QKD']
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    X_base = pd.concat([X_base, X_cat], axis=1)
    X_base = X_base.loc[:, X_base.var() > 0]
    X_current = X_base.copy()
    if 'Nature_of_Work_Change_Index' in df.columns:
        X_current['Task_Trans'] = df['Nature_of_Work_Change_Index']
        X_current['Task_Trans_Sq'] = df['Nature_of_Work_Change_Index'] ** 2
        X_current['QKD_x_Org_Support'] = df['QKD'] * df['Organizational_Support_Index']
        X_current['QKD_x_Task_Trans'] = df['QKD'] * df['Nature_of_Work_Change_Index']
    X_current = X_current.loc[:, X_current.var() > 0]
    vif_current = compute_vif(X_current)
    print("=== VIF for Current Job Security Predictors ===")
    print(vif_current.to_string(index=False))
    print("\nFlags (VIF > 5):")
    print(vif_current[vif_current['VIF'] > 5].to_string(index=False))
    print("\n")
    valid_idx_curr = (df['Not_Sure_Current'] == 0) & df['Job_Security_Index_Current'].notna()
    y_curr = df.loc[valid_idx_curr, 'Job_Security_Index_Current'].astype(int)
    y_curr = pd.Series(pd.Categorical(y_curr, categories=[-2, -1, 0, 1, 2], ordered=True), index=y_curr.index)
    X_curr_model = X_current.loc[valid_idx_curr]
    try:
        mod_curr = OrderedModel(y_curr, X_curr_model, distr='logit')
        res_curr = mod_curr.fit(method='bfgs', disp=False)
        print_model_results(res_curr, "Ordinal Logistic Regression: Current Job Security")
    except Exception as e:
        print("Error fitting Ordinal Model for Current Job Security: " + str(e))
    X_future = X_base.copy()
    if 'Future_Nature_of_Work_Change_Index' in df.columns:
        X_future['Task_Trans'] = df['Future_Nature_of_Work_Change_Index']
        X_future['Task_Trans_Sq'] = df['Future_Nature_of_Work_Change_Index'] ** 2
        X_future['QKD_x_Org_Support'] = df['QKD'] * df['Organizational_Support_Index']
        X_future['QKD_x_Task_Trans'] = df['QKD'] * df['Future_Nature_of_Work_Change_Index']
    X_future = X_future.loc[:, X_future.var() > 0]
    valid_idx_fut = (df['Not_Sure_Future'] == 0) & df['Job_Security_Index_Future'].notna()
    y_fut = df.loc[valid_idx_fut, 'Job_Security_Index_Future'].astype(int)
    y_fut = pd.Series(pd.Categorical(y_fut, categories=[-2, -1, 0, 1, 2], ordered=True), index=y_fut.index)
    X_fut_model = X_future.loc[valid_idx_fut]
    try:
        mod_fut = OrderedModel(y_fut, X_fut_model, distr='logit')
        res_fut = mod_fut.fit(method='bfgs', disp=False)
        print_model_results(res_fut, "Ordinal Logistic Regression: Future Job Security")
    except Exception as e:
        print("Error fitting Ordinal Model for Future Job Security: " + str(e))
    X_notsure = X_current.copy()
    X_notsure = sm.add_constant(X_notsure)
    y_notsure = df['Not_Sure_Current']
    try:
        mod_ns = sm.Logit(y_notsure, X_notsure)
        res_ns = mod_ns.fit(disp=False)
        print_model_results(res_ns, "Sensitivity Analysis: Logistic Regression Predicting 'Not Sure' (Current Job Security)")
    except Exception as e:
        print("Error fitting Logistic Model for Not Sure: " + str(e))
    print("=== Sensitivity Analysis: Multinomial Logistic Regression (Current Job Security) ===")
    y_multi = df['Job_Security_Index_Current'].copy()
    y_multi[df['Not_Sure_Current'] == 1] = 3
    valid_idx_multi = y_multi.notna()
    y_multi = y_multi[valid_idx_multi].astype(int)
    X_multi = X_current.loc[valid_idx_multi]
    X_multi = sm.add_constant(X_multi)
    try:
        y_multi_mapped = y_multi + 2
        mod_multi = sm.MNLogit(y_multi_mapped, X_multi)
        res_multi = mod_multi.fit(disp=False, maxiter=100)
        print("Multinomial Model fitted successfully. (Reference category: Significantly Negative)")
        not_sure_coefs = res_multi.params.iloc[:, -1]
        not_sure_pvals = res_multi.pvalues.iloc[:, -1]
        not_sure_bse = res_multi.bse.iloc[:, -1]
        pvals_fdr_multi = multipletests(not_sure_pvals, method='fdr_bh')[1]
        sens_df = pd.DataFrame({'Coef (Not Sure vs Sig.Neg)': not_sure_coefs, 'Std.Err': not_sure_bse, 'P-value': not_sure_pvals, 'FDR_P': pvals_fdr_multi})
        print(sens_df.to_string())
    except Exception as e:
        print("Error fitting Multinomial Model: " + str(e))
    print("\n")
if __name__ == '__main__':
    main()