# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

def main():
    data_dir = 'data/'
    input_path = os.path.join(data_dir, 'processed_data.csv')
    print("Loading dataset from " + input_path + "...")
    df = pd.read_csv(input_path, low_memory=False)
    prefixes_to_find = ['HIDDG', 'QDB', 'QDA', 'QKB_1_11', 'QKB_1_4', 'QC']
    rename_dict = {}
    for prefix in prefixes_to_find:
        col_match = next((c for c in df.columns if c.startswith(prefix)), None)
        if col_match:
            rename_dict[col_match] = prefix
    df = df.rename(columns=rename_dict)
    cols = ['LCA_Class', 'QKB_Index_Reduced', 'QGO_Index', 'Positive_Affect', 'Negative_Affect', 'HIDDG', 'QDB', 'QDA', 'QKB_1_11', 'QKB_1_4', 'QC']
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print("Error: Missing columns in dataset: " + str(missing_cols))
        return
    df_model = df[cols].dropna().copy()
    print("Data shape after dropping NaNs: " + str(df_model.shape))
    df_model['LCA_Class'] = df_model['LCA_Class'].astype(int)
    for cat_col in ['HIDDG', 'QDB', 'QDA']:
        df_model[cat_col] = df_model[cat_col].astype(str).str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
    base_class = sorted(df_model['LCA_Class'].unique())[0]
    print("Base Class for MNLogit: " + str(base_class))
    print("\n" + "="*50)
    print("--- Variance Inflation Factors (VIF) ---")
    print("="*50)
    vif_data = df_model[['QKB_Index_Reduced', 'QGO_Index', 'Positive_Affect', 'Negative_Affect', 'QKB_1_11', 'QKB_1_4']]
    cat_dummies = pd.get_dummies(df_model[['HIDDG', 'QDB', 'QDA']], drop_first=True).astype(float)
    vif_data = pd.concat([vif_data, cat_dummies], axis=1)
    vif_data = sm.add_constant(vif_data)
    vif_df = pd.DataFrame()
    vif_df["Variable"] = vif_data.columns
    vif_df["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    print(vif_df[vif_df['Variable'] != 'const'].to_string(index=False))
    print("\n" + "="*50)
    print("--- Multinomial Logistic Regression ---")
    print("="*50)
    formula = 'LCA_Class ~ QKB_Index_Reduced + QGO_Index + Positive_Affect + Negative_Affect + C(HIDDG) + C(QDB) + C(QDA) + QKB_1_11 * QKB_1_4'
    print("Fitting MNLogit model...")
    try:
        mnlogit_model = smf.mnlogit(formula, data=df_model).fit(disp=0)
    except Exception as e:
        print("Default MNLogit failed (" + str(e) + "), trying bfgs...")
        mnlogit_model = smf.mnlogit(formula, data=df_model).fit(method='bfgs', disp=0)
    print("\nMcFadden's Pseudo-R-squared: " + str(round(mnlogit_model.prsquared, 4)))
    print("\n--- Odds Ratios and 95% CIs for Interaction Term ---")
    params = mnlogit_model.params
    bse = mnlogit_model.bse
    interaction_term = [idx for idx in params.index if 'QKB_1_11' in idx and 'QKB_1_4' in idx and ':' in idx]
    if interaction_term:
        interaction_term = interaction_term[0]
        for col in params.columns:
            coef = params.loc[interaction_term, col]
            se = bse.loc[interaction_term, col]
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            or_val = np.exp(coef)
            or_lower = np.exp(ci_lower)
            or_upper = np.exp(ci_upper)
            print("Class " + str(col) + " vs Base Class " + str(base_class) + ":")
            print("  OR: " + str(round(or_val, 4)) + " (95% CI: " + str(round(or_lower, 4)) + " - " + str(round(or_upper, 4)) + ")")
    else:
        print("Interaction term not found in model parameters.")
    print("\n" + "="*50)
    print("--- Mediation Analysis (GSEM Approximation) ---")
    print("="*50)
    print("Testing mediation effect of QKB_Index_Reduced on the relationship between QKB_1_4 (Training) and LCA_Class.")
    print("Control variable: QC (Prior AI Exposure)\n")
    path_a_formula = 'QKB_Index_Reduced ~ QKB_1_4 + QC'
    path_a_model = smf.ols(path_a_formula, data=df_model).fit()
    a_coef = path_a_model.params['QKB_1_4']
    a_pval = path_a_model.pvalues['QKB_1_4']
    print("Path A (Training -> Enablers): coef = " + str(round(a_coef, 4)) + ", p-value = " + str(round(a_pval, 4)))
    path_na_formula = 'Negative_Affect ~ QKB_Index_Reduced + QC'
    path_na_model = smf.ols(path_na_formula, data=df_model).fit()
    na_coef = path_na_model.params['QKB_Index_Reduced']
    na_pval = path_na_model.pvalues['QKB_Index_Reduced']
    print("Path (Enablers -> Negative Affect): coef = " + str(round(na_coef, 4)) + ", p-value = " + str(round(na_pval, 4)))
    path_b_formula = 'LCA_Class ~ QKB_Index_Reduced + QKB_1_4 + QC'
    try:
        path_b_model = smf.mnlogit(path_b_formula, data=df_model).fit(disp=0)
    except Exception:
        path_b_model = smf.mnlogit(path_b_formula, data=df_model).fit(method='bfgs', disp=0)
    print("\nIndirect Effects (Training -> Enablers -> Class) and Bootstrapped 95% CIs (500 iterations):")
    n_iterations = 500
    np.random.seed(42)
    indirect_effects_boot = {col: [] for col in path_b_model.params.columns}
    for i in range(n_iterations):
        sample = df_model.sample(frac=1, replace=True)
        try:
            model_a = smf.ols(path_a_formula, data=sample).fit()
            a = model_a.params['QKB_1_4']
            try:
                model_b = smf.mnlogit(path_b_formula, data=sample).fit(disp=0, maxiter=100)
            except Exception:
                model_b = smf.mnlogit(path_b_formula, data=sample).fit(method='bfgs', disp=0, maxiter=100)
            for col in model_b.params.columns:
                b = model_b.params.loc['QKB_Index_Reduced', col]
                indirect_effects_boot[col].append(a * b)
        except Exception:
            continue
    for col in path_b_model.params.columns:
        b_coef = path_b_model.params.loc['QKB_Index_Reduced', col]
        b_pval = path_b_model.pvalues.loc['QKB_Index_Reduced', col]
        indirect_effect = a_coef * b_coef
        if len(indirect_effects_boot[col]) > 0:
            ci_lower = np.percentile(indirect_effects_boot[col], 2.5)
            ci_upper = np.percentile(indirect_effects_boot[col], 97.5)
        else:
            ci_lower, ci_upper = np.nan, np.nan
        print("\nClass " + str(col) + " vs Base Class " + str(base_class) + ":")
        print("  Path B (Enablers -> Class " + str(col) + "): coef = " + str(round(b_coef, 4)) + ", p-value = " + str(round(b_pval, 4)))
        print("  Indirect Effect: " + str(round(indirect_effect, 4)) + " (95% CI: " + str(round(ci_lower, 4)) + " - " + str(round(ci_upper, 4)) + ")")

if __name__ == '__main__':
    main()