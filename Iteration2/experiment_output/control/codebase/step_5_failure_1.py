# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

def main():
    data_dir = "data/"
    file_path = os.path.join(data_dir, "processed_data.csv")
    print("Loading processed dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    exclude_cols = ['QEA_2', 'QEB_2', 'QGP_1', 'QGP_2', 'QGP_3', 'QGU_1', 'QGU_2', 'QGU_3', 'QGH1_1', 'QGH1_2', 'QGH2_1', 'QGH2_2', 'QF_3', 'QGR', 'Job_Security_Current', 'Job_Security_Future', 'Nature_of_Work_Change', 'Future_Nature_of_Work_Change', 'LCA_Class', 'QEA_2_Not_Sure', 'QEB_2_Not_Sure']
    exact_matches = {'QDC', 'QDE_Year', 'Income_Rank', 'QC', 'QGI', 'QGS', 'QGN_Numeric', 'QKD_Numeric', 'QA_1_1', 'Positive_Affect', 'Negative_Affect', 'AI_Sentiment', 'QKB_Index', 'QKB_Index_Reduced', 'QGO_Index'}
    prefix_matches = {'QHD_', 'QGM_', 'QEA_', 'QEB_', 'QKB_1_', 'QKB_2_', 'QF_', 'QKC_', 'QGO_', 'QED_1_'}
    valid_predictors = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.endswith('_Other'):
            continue
        if col in exact_matches or any(col.startswith(p) for p in prefix_matches):
            if pd.api.types.is_numeric_dtype(df[col]):
                valid_predictors.append(col)
    valid_predictors = sorted(list(set(valid_predictors)))
    print("Identified " + str(len(valid_predictors)) + " valid predictors.")
    outcomes = ['Job_Security_Current', 'Job_Security_Future', 'Nature_of_Work_Change', 'Future_Nature_of_Work_Change']
    print("\n--- Bivariate Analysis (Spearman Rank Correlations) ---")
    results = []
    for outcome in outcomes:
        if outcome not in df.columns:
            print("Warning: Outcome " + outcome + " not found in dataset.")
            continue
        y = df[outcome]
        for predictor in valid_predictors:
            x = df[predictor]
            valid_idx = ~(x.isna() | y.isna())
            if valid_idx.sum() < 10:
                continue
            rho, pval = spearmanr(x[valid_idx], y[valid_idx])
            results.append({'Outcome': outcome, 'Predictor': predictor, 'rho': rho, 'pval': pval})
    res_df = pd.DataFrame(results)
    for outcome in outcomes:
        mask = res_df['Outcome'] == outcome
        if mask.sum() == 0:
            continue
        pvals = res_df.loc[mask, 'pval']
        pvals_clean = pvals.fillna(1.0)
        reject, pvals_corrected, _, _ = multipletests(pvals_clean, alpha=0.05, method='fdr_bh')
        res_df.loc[mask, 'pval_adj'] = pvals_corrected
    for outcome in outcomes:
        print("\nTop 20 predictors for " + outcome + " (Spearman correlation):")
        df_out = res_df[res_df['Outcome'] == outcome].copy()
        df_out['abs_rho'] = df_out['rho'].abs()
        df_out = df_out.sort_values('abs_rho', ascending=False).head(20)
        for _, row in df_out.iterrows():
            print("  " + str(row['Predictor']) + ": rho = " + str(round(row['rho'], 3)) + ", p-adj = " + "{:.3e}".format(row['pval_adj']))
    biv_path = os.path.join(data_dir, "bivariate_correlations.csv")
    res_df.to_csv(biv_path, index=False)
    print("\nBivariate correlations saved to " + biv_path)
    print("\n--- Random Forest Feature Importance (5-fold CV Permutation) ---")
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        print("\nTraining Random Forest for " + outcome + "...")
        y = df[outcome]
        valid_idx = ~y.isna()
        X_valid = df.loc[valid_idx, valid_predictors]
        y_valid = y[valid_idx]
        if len(y_valid) < 50:
            print("Not enough valid samples for " + outcome + ".")
            continue
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X_valid)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        importances = []
        for train_idx, test_idx in kf.split(X_imp):
            X_train, X_test = X_imp[train_idx], X_imp[test_idx]
            y_train, y_test = y_valid.iloc[train_idx], y_valid.iloc[test_idx]
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pi = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            importances.append(pi.importances)
        all_importances = np.concatenate(importances, axis=1)
        mean_imp = all_importances.mean(axis=1)
        std_imp = all_importances.std(axis=1)
        imp_df = pd.DataFrame({'Predictor': valid_predictors, 'Importance_Mean': mean_imp, 'Importance_Std': std_imp})
        imp_df = imp_df.sort_values('Importance_Mean', ascending=False).head(20)
        print("Top 20 Random Forest features for " + outcome + ":")
        for _, row in imp_df.iterrows():
            print("  " + str(row['Predictor']) + ": mean = " + str(round(row['Importance_Mean'], 4)) + ", std = " + str(round(row['Importance_Std'], 4)))
        rf_path = os.path.join(data_dir, "rf_importance_" + outcome + ".csv")
        imp_df.to_csv(rf_path, index=False)
        print("Random Forest importances saved to " + rf_path)

if __name__ == '__main__':
    main()