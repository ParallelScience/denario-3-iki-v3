# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import re

if __name__ == '__main__':
    data_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    df = pd.read_csv(data_path, sep='\t', low_memory=False)
    df.replace('.', np.nan, inplace=True)
    qhd_cols = [c for c in df.columns if c.startswith('QHD_')]
    qgm_cols = [c for c in df.columns if c.startswith('QGM_')]
    qkb1_cols = [c for c in df.columns if c.startswith('QKB_1_')]
    qkb2_cols = [c for c in df.columns if c.startswith('QKB_2_')]
    qgo_cols = [c for c in df.columns if c.startswith('QGO_')]
    print('QHD columns:', qhd_cols)
    print('QGM columns:', qgm_cols)
    print('QKB_1 columns:', qkb1_cols)
    print('QKB_2 columns:', qkb2_cols)
    print('QGO columns:', qgo_cols)
    binary_prefixes = ['QHD_', 'QGM_', 'QA_1_', 'QGL_', 'QF_', 'QED_1_', 'QKC_', 'QGH1_', 'QGH2_']
    binary_cols = [c for c in df.columns if any(c.startswith(p) for p in binary_prefixes)]
    for c in binary_cols:
        df[c] = df[c].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    qea_qeb_map = {'significantly negative': -2, 'slightly negative': -1, 'no impact': 0, 'slightly positive': 1, 'significantly positive': 2, 'not sure': np.nan}
    qgp_qgu_map = {'large decrease': -2, 'moderate decrease': -1, 'no major change': 0, 'moderate increase': 1, 'large increase': 2}
    qgo_map = {'strongly disagree': 1, 'somewhat disagree': 2, 'neither agree nor disagree': 3, 'somewhat agree': 4, 'strongly agree': 5}
    qc_map = {'once a month or fewer': 1, 'a few times a month': 2, 'once a week': 3, 'a few times a week': 4, 'once a day': 5, 'a few times a day': 6, 'many times a day': 7}
    qgi_map = {'ai increases the time i spend on tasks': -1, 'none': 0, 'less than 1 hour': 1, '1 to less than 3 hours': 2, '3 to less than 5 hours': 3, '5 to less than 10 hours': 4, '10 hours or more': 5}
    qgs_map = {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3, 'constantly': 4}
    qgr_map = {'not at all confident': 1, 'slightly confident': 2, 'somewhat confident': 3, 'very confident': 4, 'not sure': np.nan}
    likert_map = {'strongly disagree': 1, 'somewhat disagree': 2, 'neither agree nor disagree': 3, 'somewhat agree': 4, 'strongly agree': 5, 'not at all important': 1, 'slightly important': 2, 'moderately important': 3, 'very important': 4, 'extremely important': 5, 'not at all': 1, 'to a small extent': 2, 'to a moderate extent': 3, 'to a great extent': 4, 'to a very great extent': 5, 'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4, 'always': 5}
    def clean_map(val, mapping):
        if pd.isna(val): return np.nan
        v = str(val).strip().lower()
        return mapping.get(v, np.nan)
    def map_qkb(val):
        if pd.isna(val): return np.nan
        v = str(val).strip().lower()
        if v in likert_map: return likert_map[v]
        if 'not at all' in v or 'strongly disagree' in v: return 1
        if 'slightly' in v or 'somewhat disagree' in v: return 2
        if 'moderately' in v or 'neither' in v: return 3
        if 'very' in v or 'somewhat agree' in v: return 4
        if 'extremely' in v or 'strongly agree' in v: return 5
        return np.nan
    for c in df.columns:
        if c.startswith('QEA_') or c.startswith('QEB_'): df[c] = df[c].apply(lambda x: clean_map(x, qea_qeb_map))
        elif c.startswith('QGP_') or c.startswith('QGU_'): df[c] = df[c].apply(lambda x: clean_map(x, qgp_qgu_map))
        elif c.startswith('QGO_'): df[c] = df[c].apply(lambda x: clean_map(x, qgo_map))
        elif c.startswith('QKB_1_') or c.startswith('QKB_2_'): df[c] = df[c].apply(map_qkb)
    if 'QC' in df.columns: df['QC_encoded'] = df['QC'].apply(lambda x: clean_map(x, qc_map))
    if 'QGI' in df.columns: df['QGI_encoded'] = df['QGI'].apply(lambda x: clean_map(x, qgi_map))
    if 'QGS' in df.columns: df['QGS_encoded'] = df['QGS'].apply(lambda x: clean_map(x, qgs_map))
    if 'QGR' in df.columns: df['QGR_encoded'] = df['QGR'].apply(lambda x: clean_map(x, qgr_map))
    qea_2_col = next((c for c in df.columns if c.startswith('QEA_2')), None)
    qeb_2_col = next((c for c in df.columns if c.startswith('QEB_2')), None)
    if qea_2_col: df['Job_Security_Current'] = df[qea_2_col]
    if qeb_2_col: df['Job_Security_Future'] = df[qeb_2_col]
    qgp_cols_exact = [c for c in df.columns if c.startswith('QGP_1') or c.startswith('QGP_2') or c.startswith('QGP_3')]
    if len(qgp_cols_exact) == 3: df['Nature_of_Work_Change_Current'] = df[qgp_cols_exact].mean(axis=1)
    qgu_cols_exact = [c for c in df.columns if c.startswith('QGU_1') or c.startswith('QGU_2') or c.startswith('QGU_3')]
    if len(qgu_cols_exact) == 3: df['Nature_of_Work_Change_Future'] = df[qgu_cols_exact].mean(axis=1)
    df['QGM_Composite'] = df[qgm_cols].sum(axis=1)
    qa_1_1_col = next((c for c in df.columns if c.startswith('QA_1_1')), None)
    if qa_1_1_col: df['Usage_State'] = df[qa_1_1_col]
    def map_qgn(val):
        if pd.isna(val): return np.nan
        v = str(val).lower()
        if 'does not' in v or "doesn't" in v: return 1
        if 'exploring' in v or 'experimenting' in v: return 2
        if 'few' in v or 'specific' in v: return 3
        if 'widely' in v or 'multiple' in v: return 4
        if 'transforming' in v or 'most' in v: return 5
        return np.nan
    if 'QGN' in df.columns: df['QGN_encoded'] = df['QGN'].apply(map_qgn)
    if 'QKD' in df.columns:
        df['QKD_Yes'] = (df['QKD'].str.strip().str.lower() == 'yes').astype(int)
        df['QKD_No'] = (df['QKD'].str.strip().str.lower() == 'no').astype(int)
        df['QKD_NotSure'] = (df['QKD'].str.strip().str.lower() == 'not sure').astype(int)
    qdf_cols = [c for c in df.columns if c.startswith('QDF_')]
    if qdf_cols:
        df['Income_Raw'] = df[qdf_cols].bfill(axis=1).iloc[:, 0]
        def extract_income_val(val):
            if pd.isna(val): return np.nan
            s = str(val).replace(',', '').replace('.', '')
            nums = re.findall(r'\d+', s)
            return int(nums[0]) if nums else np.nan
        df['Income_Num'] = df['Income_Raw'].apply(extract_income_val)
        if 'QDA' in df.columns: df['Income_Rank'] = df.groupby('QDA')['Income_Num'].rank(method='dense', ascending=True)
    output_path = os.path.join('data', 'processed_data_step1.csv')
    df.to_csv(output_path, index=False)
    print('Processed data saved to ' + output_path)
    if 'Job_Security_Current' in df.columns:
        print('\nSummary of Job Security Current:')
        print(df['Job_Security_Current'].value_counts(dropna=False))
    if 'Job_Security_Future' in df.columns:
        print('\nSummary of Job Security Future:')
        print(df['Job_Security_Future'].value_counts(dropna=False))
    if 'Nature_of_Work_Change_Current' in df.columns:
        print('\nSummary of Nature of Work Change Current:')
        print(df['Nature_of_Work_Change_Current'].describe())
    print('\nSummary of QGM Composite:')
    print(df['QGM_Composite'].describe())
    if 'Usage_State' in df.columns:
        print('\nSummary of Usage State:')
        print(df['Usage_State'].value_counts(dropna=False))
    if 'Income_Rank' in df.columns:
        print('\nSummary of Income Rank:')
        print(df['Income_Rank'].describe())