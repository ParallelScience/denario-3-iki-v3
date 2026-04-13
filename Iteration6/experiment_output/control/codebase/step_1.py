# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler

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
    if 'strongly disagree' in s: return -2
    if 'somewhat disagree' in s: return -1
    if 'neither agree' in s: return 0
    if 'somewhat agree' in s: return 1
    if 'strongly agree' in s: return 2
    if 'not at all important' in s: return -2
    if 'slightly important' in s: return -1
    if 'moderately important' in s: return 0
    if 'very important' in s: return 1
    if 'extremely important' in s: return 2
    if 'never' in s: return -2
    if 'rarely' in s: return -1
    if 'sometimes' in s: return 0
    if 'often' in s: return 1
    if 'always' in s: return 2
    if 'not at all' in s: return -2
    if 'small extent' in s: return -1
    if 'moderate extent' in s: return 0
    if 'great extent' in s: return 1
    if 'very great extent' in s: return 2
    if 'very dissatisfied' in s: return -2
    if 'somewhat dissatisfied' in s: return -1
    if 'neither satisfied' in s: return 0
    if 'somewhat satisfied' in s: return 1
    if 'very satisfied' in s: return 2
    if 'poor' in s: return -2
    if 'fair' in s: return -1
    if 'good' in s: return 0
    if 'very good' in s: return 1
    if 'excellent' in s: return 2
    try:
        v = float(s)
        if 1 <= v <= 5:
            return int(v) - 3
    except ValueError:
        pass
    return np.nan

def map_qc(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'once a month or fewer' in s: return 1
    if 'a few times a month' in s: return 2
    if 'once a week' in s: return 3
    if 'a few times a week' in s: return 4
    if 'once a day' in s: return 5
    if 'a few times a day' in s: return 6
    if 'many times a day' in s: return 7
    try:
        v = float(s)
        if 1 <= v <= 7:
            return int(v)
    except ValueError:
        pass
    return np.nan

def map_qgi(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'increases time' in s: return -1
    if 'none' in s: return 0
    if 'less than 1' in s or '<1' in s: return 1
    if '1-3' in s or '1 to 3' in s: return 2
    if '3-5' in s or '3 to 5' in s: return 3
    if 'more than 5' in s or '>5' in s: return 4
    try:
        v = float(s)
        if -1 <= v <= 4:
            return int(v)
    except ValueError:
        pass
    return np.nan

def map_qgs(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'never' in s: return 0
    if 'rarely' in s: return 1
    if 'sometimes' in s: return 2
    if 'often' in s: return 3
    if 'constantly' in s: return 4
    try:
        v = float(s)
        if 0 <= v <= 4:
            return int(v)
    except ValueError:
        pass
    return np.nan

def map_qgr(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'not sure' in s: return np.nan
    if 'not at all confident' in s: return 1
    if 'slightly confident' in s: return 2
    if 'somewhat confident' in s: return 3
    if 'very confident' in s: return 4
    try:
        v = float(s)
        if 1 <= v <= 4:
            return int(v)
    except ValueError:
        pass
    return np.nan

def map_binary(x):
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    if s in ['yes', '1', 'true', 'selected', 'checked']: return 1
    return 0

def extract_income_val(s):
    if pd.isna(s): return np.nan
    s_lower = str(s).lower()
    if 'prefer not' in s_lower or "don't know" in s_lower or 'not sure' in s_lower:
        return np.nan
    matches = re.findall(r'\d+', s_lower.replace(',', '').replace('.', ''))
    if not matches:
        return np.nan
    val = int(matches[0])
    if 'under' in s_lower or 'less than' in s_lower or '<' in s_lower:
        val -= 1
    return val

if __name__ == '__main__':
    file_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    df = pd.read_csv(file_path, sep='\t', low_memory=False, na_values=['.'])
    binary_prefixes = ['QHD', 'QGM', 'QF', 'QED_1', 'QKC', 'QGH', 'QA_1', 'QGL']
    binary_cols = [c for c in df.columns if any(c.startswith(p) for p in binary_prefixes)]
    for c in binary_cols:
        df[c] = df[c].apply(map_binary)
    likert_prefixes = ['QEA', 'QEB', 'QGP', 'QGU', 'QKB_1', 'QKB_2', 'QGO']
    likert_cols = [c for c in df.columns if any(c.startswith(p) for p in likert_prefixes)]
    for c in likert_cols:
        df[c] = df[c].apply(map_likert)
    if 'QC' in df.columns: df['QC'] = df['QC'].apply(map_qc)
    if 'QGI' in df.columns: df['QGI'] = df['QGI'].apply(map_qgi)
    if 'QGS' in df.columns: df['QGS'] = df['QGS'].apply(map_qgs)
    if 'QGR' in df.columns: df['QGR'] = df['QGR'].apply(map_qgr)
    if 'QKD' in df.columns:
        df['QKD'] = df['QKD'].apply(lambda x: 1 if 'yes' in str(x).lower() else (0 if 'no' in str(x).lower() else np.nan))
    missing_rates = df.isnull().mean() * 100
    qdf_cols = [c for c in df.columns if c.startswith('QDF')]
    def get_income_str(row):
        for c in qdf_cols:
            if pd.notna(row[c]) and str(row[c]).strip() != '':
                return str(row[c]).strip()
        return np.nan
    df['Income_String'] = df.apply(get_income_str, axis=1)
    df['Income_Val'] = df['Income_String'].apply(extract_income_val)
    if 'QDA' in df.columns:
        df['Income_Rank'] = df.groupby('QDA')['Income_Val'].rank(method='dense', ascending=True)
    else:
        df['Income_Rank'] = df['Income_Val'].rank(method='dense', ascending=True)
    if 'QEA_2' in df.columns: df['Job_Security_Index_Current'] = df['QEA_2']
    if 'QEB_2' in df.columns: df['Job_Security_Index_Future'] = df['QEB_2']
    qgp_cols = [c for c in ['QGP_1', 'QGP_2', 'QGP_3'] if c in df.columns]
    if qgp_cols: df['Nature_of_Work_Change_Index'] = df[qgp_cols].mean(axis=1)
    qgu_cols = [c for c in ['QGU_1', 'QGU_2', 'QGU_3'] if c in df.columns]
    if qgu_cols: df['Future_Nature_of_Work_Change_Index'] = df[qgu_cols].mean(axis=1)
    if 'QGP_1' in df.columns and 'QGU_1' in df.columns: df['Task_Transformation_Repetitive'] = df[['QGP_1', 'QGU_1']].mean(axis=1)
    if 'QGP_2' in df.columns and 'QGU_2' in df.columns: df['Task_Transformation_Creative'] = df[['QGP_2', 'QGU_2']].mean(axis=1)
    if 'QGP_3' in df.columns and 'QGU_3' in df.columns: df['Task_Transformation_Complex'] = df[['QGP_3', 'QGU_3']].mean(axis=1)
    scaler = StandardScaler()
    for col in ['QDC', 'QDE_Year', 'QC']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            valid_idx = df[col].notna()
            if valid_idx.sum() > 0:
                df.loc[valid_idx, col + '_std'] = scaler.fit_transform(df.loc[valid_idx, [col]]).flatten()
    output_path = os.path.join('data', 'cleaned_dataset.csv')
    df.to_csv(output_path, index=False)
    print('Cleaned dataset saved to ' + output_path)