# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import re
import os

def cronbach_alpha(df_items):
    df_items = df_items.dropna()
    k = df_items.shape[1]
    if k <= 1:
        return np.nan
    var_items = df_items.var(axis=0, ddof=1).sum()
    var_total = df_items.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    return (k / (k - 1)) * (1 - var_items / var_total)

def main():
    file_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    print('Loading dataset...')
    df = pd.read_csv(file_path, sep='\t', low_memory=False, na_values=['.', ' '])
    print('Dataset loaded. Shape: ' + str(df.shape))
    if 'QEA_2' in df.columns:
        df['QEA_2_Not_Sure'] = df['QEA_2'].apply(lambda x: 1 if str(x).strip().lower() == 'not sure' else 0)
    if 'QEB_2' in df.columns:
        df['QEB_2_Not_Sure'] = df['QEB_2'].apply(lambda x: 1 if str(x).strip().lower() == 'not sure' else 0)
    qea_qeb_map = {'significantly negative': -2, 'slightly negative': -1, 'no impact': 0, 'slightly positive': 1, 'significantly positive': 2, 'not sure': np.nan}
    qgp_qgu_map = {'large decrease': -2, 'moderate decrease': -1, 'no major change': 0, 'moderate increase': 1, 'large increase': 2}
    qgr_map = {'not at all confident': 1, 'slightly confident': 2, 'somewhat confident': 3, 'very confident': 4, 'not sure': np.nan}
    qgo_map = {'strongly disagree': 1, 'somewhat disagree': 2, 'neither agree nor disagree': 3, 'somewhat agree': 4, 'strongly agree': 5}
    qkb_map = {'not at all': 1, 'slightly': 2, 'moderately': 3, 'very': 4, 'extremely': 5}
    qc_map = {'once a month or fewer': 1, 'a few times a month': 2, 'once a week': 3, 'a few times a week': 4, 'once a day': 5, 'a few times a day': 6, 'many times a day': 7}
    qgi_map = {'increases time': -1, 'none': 0, '<1h': 1, 'less than 1 hour': 1, '1-3h': 2, '1–3h': 2, '1 to 3 hours': 2, '3-5h': 3, '3–5h': 3, '3 to 5 hours': 3, '>5h': 4, 'more than 5 hours': 4}
    qgs_map = {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3, 'constantly': 4}
    def map_likert(val, mapping):
        if pd.isna(val): return np.nan
        val_str = str(val).strip().lower()
        try:
            num = float(val_str)
            if num in mapping.values(): return num
        except ValueError:
            pass
        for k, v in mapping.items():
            if k in val_str: return v
        return np.nan
    def map_qgn(val):
        if pd.isna(val): return np.nan
        val_str = str(val).lower()
        try:
            num = float(val_str)
            if num in [1, 2, 3, 4, 5]: return num
        except ValueError:
            pass
        if "doesn't use" in val_str or "does not use" in val_str: return 1
        if "explor" in val_str or "experiment" in val_str or "pilot" in val_str: return 2
        if "some" in val_str or "limited" in val_str or "few" in val_str: return 3
        if "many" in val_str or "widely" in val_str or "multiple" in val_str: return 4
        if "transform" in val_str or "most" in val_str or "all" in val_str or "core" in val_str: return 5
        return np.nan
    print('Applying ordinal mappings...')
    for c in df.columns:
        if c.startswith('QEA_') or c.startswith('QEB_'): df[c] = df[c].apply(lambda x: map_likert(x, qea_qeb_map))
        elif c.startswith('QGP_') or c.startswith('QGU_'): df[c] = df[c].apply(lambda x: map_likert(x, qgp_qgu_map))
        elif c == 'QGR': df[c] = df[c].apply(lambda x: map_likert(x, qgr_map))
        elif c.startswith('QGO_'): df[c] = df[c].apply(lambda x: map_likert(x, qgo_map))
        elif c.startswith('QKB_1_'): df[c] = df[c].apply(lambda x: map_likert(x, qkb_map))
        elif c == 'QC': df[c] = df[c].apply(lambda x: map_likert(x, qc_map))
        elif c == 'QGI': df[c] = df[c].apply(lambda x: map_likert(x, qgi_map))
        elif c == 'QGS': df[c] = df[c].apply(lambda x: map_likert(x, qgs_map))
    if 'QGN' in df.columns: df['QGN_Numeric'] = df['QGN'].apply(map_qgn)
    if 'QKD' in df.columns: df['QKD_Numeric'] = df['QKD'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else (0 if str(x).strip().lower() == 'no' else np.nan))
    print('Applying binary mappings...')
    binary_prefixes = ['QHD_', 'QGM_', 'QA_1_', 'QGL_', 'QF_', 'QED_1_', 'QKC_', 'QGH1_', 'QGH2_']
    binary_cols = [c for c in df.columns if any(c.startswith(p) for p in binary_prefixes)]
    def map_binary(x):
        if pd.isna(x): return np.nan
        val = str(x).strip().lower()
        if val in ['yes', '1', '1.0', 'true']: return 1
        return 0
    for c in binary_cols: df[c] = df[c].apply(map_binary)
    if 'QDC' in df.columns: df['QDC'] = pd.to_numeric(df['QDC'], errors='coerce')
    if 'QDE_Year' in df.columns: df['QDE_Year'] = pd.to_numeric(df['QDE_Year'], errors='coerce')
    print('Calculating Income Rank...')
    qdf_cols = [c for c in df.columns if c.startswith('QDF')]
    if qdf_cols and 'QDA' in df.columns:
        df['Income_Band_Raw'] = df[qdf_cols].bfill(axis=1).iloc[:, 0]
        def extract_min_income(s):
            if pd.isna(s): return -1
            s = str(s).replace(',', '').replace('.', '')
            numbers = re.findall(r'\d+', s)
            if numbers: return int(numbers[0])
            return -1
        df['Income_Min'] = df['Income_Band_Raw'].apply(extract_min_income)
        df['Income_Min'] = df['Income_Min'].replace(-1, np.nan)
        df['Income_Rank'] = df.groupby('QDA')['Income_Min'].rank(method='dense', ascending=True)
    print('Constructing composite indices...')
    if 'QEA_2' in df.columns: df['Job_Security_Current'] = df['QEA_2']
    if 'QEB_2' in df.columns: df['Job_Security_Future'] = df['QEB_2']
    qgp_cols = ['QGP_1', 'QGP_2', 'QGP_3']
    if all(c in df.columns for c in qgp_cols): df['Nature_of_Work_Change'] = df[qgp_cols].mean(axis=1)
    qgu_cols = ['QGU_1', 'QGU_2', 'QGU_3']
    if all(c in df.columns for c in qgu_cols): df['Future_Nature_of_Work_Change'] = df[qgu_cols].mean(axis=1)
    qgo_cols = [c for c in df.columns if c.startswith('QGO_') and c != 'QGO_Index']
    if qgo_cols: df['QGO_Index'] = df[qgo_cols].mean(axis=1)
    qkb_cols = [c for c in df.columns if c.startswith('QKB_1_') and c not in ['QKB_Index', 'QKB_Index_Reduced']]
    if qkb_cols: df['QKB_Index'] = df[qkb_cols].mean(axis=1)
    qkb_reduced_cols = [c for c in qkb_cols if c not in ['QKB_1_4', 'QKB_1_11']]
    if qkb_reduced_cols: df['QKB_Index_Reduced'] = df[qkb_reduced_cols].mean(axis=1)
    print('\n--- Reliability Analysis ---')
    if qgo_cols: print('Cronbach\'s alpha for QGO_Index (items=' + str(len(qgo_cols)) + '): ' + str(round(cronbach_alpha(df[qgo_cols]), 3)))
    if qkb_cols: print('Cronbach\'s alpha for QKB_Index (items=' + str(len(qkb_cols)) + '): ' + str(round(cronbach_alpha(df[qkb_cols]), 3)))
    if qkb_reduced_cols: print('Cronbach\'s alpha for QKB_Index_Reduced (items=' + str(len(qkb_reduced_cols)) + '): ' + str(round(cronbach_alpha(df[qkb_reduced_cols]), 3)))
    print('----------------------------\n')
    out_path = 'data/processed_data.csv'
    df.to_csv(out_path, index=False)
    print('\nProcessed dataset saved to ' + out_path)

if __name__ == '__main__':
    main()