# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os
import re

def extract_min_income(val):
    if pd.isna(val): return np.nan
    s = str(val).lower().replace(',', '')
    if 'under' in s or 'less than' in s or 'below' in s:
        return 0.0
    numbers = re.findall(r'\d+', s)
    if numbers:
        return float(numbers[0])
    return np.nan

def map_likert(val, scale_type):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s == 'not sure': return 99
    if scale_type == 'impact':
        if 'significantly negative' in s: return -2
        if 'slightly negative' in s: return -1
        if 'no impact' in s: return 0
        if 'slightly positive' in s: return 1
        if 'significantly positive' in s: return 2
    elif scale_type == 'change':
        if 'large decrease' in s: return -2
        if 'moderate decrease' in s: return -1
        if 'no major change' in s: return 0
        if 'moderate increase' in s: return 1
        if 'large increase' in s: return 2
    elif scale_type == 'importance':
        if 'not at all' in s: return -2
        if 'slightly' in s: return -1
        if 'moderately' in s or 'somewhat' in s: return 0
        if 'very' in s: return 1
        if 'extremely' in s: return 2
    elif scale_type == 'agreement':
        if 'strongly disagree' in s: return -2
        if 'somewhat disagree' in s: return -1
        if 'neither' in s or 'neutral' in s: return 0
        if 'somewhat agree' in s: return 1
        if 'strongly agree' in s: return 2
    try:
        v = float(s)
        if -2 <= v <= 2 or v == 99: return v
    except ValueError:
        pass
    return np.nan

def map_qc(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'once a month or fewer' in s or 'less than once a month' in s: return 1
    if 'a few times a month' in s or 'several times a month' in s: return 2
    if 'once a week' in s: return 3
    if 'a few times a week' in s or 'several times a week' in s: return 4
    if 'once a day' in s: return 5
    if 'a few times a day' in s or 'several times a day' in s: return 6
    if 'many times a day' in s or 'multiple times a day' in s: return 7
    try:
        v = float(s)
        if 1 <= v <= 7: return v
    except ValueError:
        pass
    return np.nan

def map_qgi(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'increases time' in s or 'more time' in s: return -1
    if 'none' in s or '0' in s: return 0
    if '<1' in s or 'less than 1' in s: return 1
    if '1-3' in s or '1 to 3' in s or '1–3' in s: return 2
    if '3-5' in s or '3 to 5' in s or '3–5' in s: return 3
    if '>5' in s or 'more than 5' in s: return 4
    try:
        v = float(s)
        if -1 <= v <= 4: return v
    except ValueError:
        pass
    return np.nan

def map_qgs(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'never' in s: return 0
    if 'rarely' in s: return 1
    if 'sometimes' in s or 'occasionally' in s: return 2
    if 'often' in s or 'frequently' in s: return 3
    if 'constantly' in s or 'always' in s: return 4
    try:
        v = float(s)
        if 0 <= v <= 4: return v
    except ValueError:
        pass
    return np.nan

def map_qgr(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'not sure' in s: return 99
    if 'not at all' in s: return 1
    if 'slightly' in s: return 2
    if 'somewhat' in s: return 3
    if 'very' in s or 'extremely' in s: return 4
    try:
        v = float(s)
        if 1 <= v <= 4 or v == 99: return v
    except ValueError:
        pass
    return np.nan

def map_qdg(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'intern' in s: return 1
    if 'entry' in s or 'junior' in s: return 2
    if 'associate' in s or 'analyst' in s: return 3
    if 'specialist' in s or 'professional' in s: return 4
    if 'senior' in s and ('contributor' in s or 'professional' in s or 'staff' in s): return 5
    if 'senior manager' in s: return 7
    if 'manager' in s: return 6
    if 'senior director' in s: return 9
    if 'director' in s: return 8
    if 'vice president' in s or 'vp' in s: return 10
    if 'c-suite' in s or 'c-level' in s or 'chief' in s or 'executive' in s or 'partner' in s: return 11
    try:
        v = float(s)
        if 1 <= v <= 11: return v
    except ValueError:
        pass
    return np.nan

def map_qgn(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if "doesn't use" in s or "does not use" in s or "no ai" in s: return 1
    if "exploring" in s or "evaluating" in s or "initial" in s: return 2
    if "using in some" in s or "limited" in s or "pockets" in s: return 3
    if "widely" in s or "scaling" in s or "multiple" in s: return 4
    if "transforming" in s or "core" in s or "everywhere" in s: return 5
    try:
        v = float(s)
        if 1 <= v <= 5: return v
    except ValueError:
        pass
    return np.nan

if __name__ == '__main__':
    data_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    df = pd.read_csv(data_path, sep='\t', low_memory=False, na_values='.')
    binary_prefixes = ('QHD', 'QGM', 'QA_1', 'QF', 'QED_1', 'QKC', 'QGH')
    binary_cols = [col for col in df.columns if col.startswith(binary_prefixes)]
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' or str(x).strip() in ['1', '1.0'] else 0)
    for col in df.columns:
        if col.startswith('QEA') or col.startswith('QEB'):
            df[col] = df[col].apply(lambda x: map_likert(x, 'impact'))
        elif col.startswith('QGP') or col.startswith('QGU'):
            df[col] = df[col].apply(lambda x: map_likert(x, 'change'))
        elif col.startswith('QKB_1'):
            df[col] = df[col].apply(lambda x: map_likert(x, 'importance'))
        elif col.startswith('QGO'):
            df[col] = df[col].apply(lambda x: map_likert(x, 'agreement'))
    if 'QC' in df.columns: df['QC'] = df['QC'].apply(map_qc)
    if 'QGI' in df.columns: df['QGI'] = df['QGI'].apply(map_qgi)
    if 'QGS' in df.columns: df['QGS'] = df['QGS'].apply(map_qgs)
    if 'QGR' in df.columns: df['QGR'] = df['QGR'].apply(map_qgr)
    if 'QDG' in df.columns: df['QDG'] = df['QDG'].apply(map_qdg)
    if 'QGN' in df.columns: df['QGN'] = df['QGN'].apply(map_qgn)
    if 'QEA_2' in df.columns and 'QEB_2' in df.columns:
        df['Baseline_Uncertainty'] = ((df['QEA_2'] == 99) | (df['QEB_2'] == 99)).astype(int)
    if 'QEA_2' in df.columns:
        df['Job_Security_Current'] = df['QEA_2'].replace(99, np.nan)
    if 'QEB_2' in df.columns:
        df['Job_Security_Future'] = df['QEB_2'].replace(99, np.nan)
    qgp_cols = ['QGP_1', 'QGP_2', 'QGP_3']
    qgp_cols = [c for c in qgp_cols if c in df.columns]
    if qgp_cols:
        df['Nature_of_Work_Change_Current'] = df[qgp_cols].replace(99, np.nan).mean(axis=1)
    qgu_cols = ['QGU_1', 'QGU_2', 'QGU_3']
    qgu_cols = [c for c in qgu_cols if c in df.columns]
    if qgu_cols:
        df['Nature_of_Work_Change_Future'] = df[qgu_cols].replace(99, np.nan).mean(axis=1)
    qkb_cols = [col for col in df.columns if col.startswith('QKB_1_')]
    if qkb_cols:
        df['QKB_Enablers_Index'] = df[qkb_cols].replace(99, np.nan).mean(axis=1)
    qgo_cols = [col for col in df.columns if col.startswith('QGO_')]
    if qgo_cols:
        df['QGO_Culture_Index'] = df[qgo_cols].replace(99, np.nan).mean(axis=1)
    qdf_cols = [col for col in df.columns if col.startswith('QDF_')]
    if qdf_cols and 'QDA' in df.columns:
        df['Income_Band_Raw'] = df[qdf_cols].bfill(axis=1).iloc[:, 0]
        df['Income_Min'] = df['Income_Band_Raw'].apply(extract_min_income)
        df['Income_Rank'] = df.groupby('QDA')['Income_Min'].rank(method='dense', ascending=True)
    print('Dataset shape:', df.shape)
    print('\nKey Variables Missing Value Counts:')
    key_vars = ['QEA_2', 'QEB_2', 'Job_Security_Current', 'Job_Security_Future', 'Nature_of_Work_Change_Current', 'Nature_of_Work_Change_Future', 'QKB_Enablers_Index', 'QGO_Culture_Index', 'Baseline_Uncertainty', 'Income_Rank']
    for var in key_vars:
        if var in df.columns:
            print(var + ': ' + str(df[var].isna().sum()))
    print('\nColumn Types Summary:')
    existing_key_vars = [var for var in key_vars if var in df.columns]
    print(df[existing_key_vars].dtypes)
    output_path = os.path.join('data', 'IKI_Cleaned_Features.csv')
    df.to_csv(output_path, index=False)
    print('\nCleaned dataset saved to ' + output_path)