# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import re
import os

def extract_num(s):
    s_str = str(s).lower().replace(',', '').replace('.', '')
    is_under = 'under' in s_str or '<' in s_str
    m = re.search(r'\d+', s_str)
    if m:
        val = float(m.group())
        if is_under:
            val -= 0.1
        return val
    return 0

def robust_map(series, mapping_dict):
    s_lower = series.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.nan, index=series.index)
    sorted_keys = sorted(mapping_dict.keys(), key=len, reverse=True)
    for k in sorted_keys:
        v = mapping_dict[k]
        mask = s_lower.str.contains(k, regex=False, na=False) & mapped.isna()
        mapped.loc[mask] = v
    mapped.loc[series.isna()] = np.nan
    return mapped

if __name__ == '__main__':
    file_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    print('Loading dataset from: ' + file_path)
    df = pd.read_csv(file_path, sep='\t', low_memory=False, na_values=['.'])
    qhd_cols = [c for c in df.columns if c.startswith('QHD')]
    qkb_cols = [c for c in df.columns if c.startswith('QKB')]
    qdf_cols = [c for c in df.columns if c.startswith('QDF')]
    print('\n--- Exact Column Names ---')
    print('QHD Columns: ' + str(qhd_cols))
    print('QKB Columns: ' + str(qkb_cols))
    print('QDF Columns: ' + str(qdf_cols))
    print('--------------------------\n')
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].astype(str).str.strip().str.lower().str.contains('not sure').any():
                df[col + '_Not_Sure'] = df[col].astype(str).str.strip().str.lower().str.contains('not sure').astype(int)
    qea_qeb_map = {'significantly negative': -2, 'slightly negative': -1, 'no impact': 0, 'slightly positive': 1, 'significantly positive': 2, 'not sure': np.nan}
    qgp_qgu_map = {'large decrease': -2, 'moderate decrease': -1, 'no major change': 0, 'moderate increase': 1, 'large increase': 2, 'not sure': np.nan}
    qkb_map = {'not at all': -2, 'slightly': -1, 'moderately': 0, 'very': 1, 'extremely': 2, 'strongly disagree': -2, 'somewhat disagree': -1, 'neither agree nor disagree': 0, 'somewhat agree': 1, 'strongly agree': 2, 'never': -2, 'rarely': -1, 'sometimes': 0, 'often': 1, 'always': 2, 'not sure': np.nan}
    qgo_map = {'strongly disagree': -2, 'somewhat disagree': -1, 'neither agree nor disagree': 0, 'somewhat agree': 1, 'strongly agree': 2, 'not sure': np.nan}
    qc_map = {'once a month or fewer': 1, 'a few times a month': 2, 'once a week': 3, 'a few times a week': 4, 'once a day': 5, 'a few times a day': 6, 'many times a day': 7, 'not sure': np.nan}
    qgi_map = {'increases time': -1, 'none': 0, 'less than 1 hour': 1, '<1h': 1, '1 to 3 hours': 2, '1-3h': 2, '3 to 5 hours': 3, '3-5h': 3, 'more than 5 hours': 4, '>5h': 4, 'not sure': np.nan}
    qgs_map = {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3, 'constantly': 4, 'not sure': np.nan}
    qgr_map = {'not at all confident': 1, 'slightly confident': 2, 'somewhat confident': 3, 'very confident': 4, 'not sure': np.nan}
    qgn_map = {"doesn't use": 1, "does not use": 1, "exploring": 2, "implementing": 3, "scaling": 4, "transforming": 5, 'not sure': np.nan}
    binary_prefixes = ['QHD', 'QGM', 'QA_1', 'QGL', 'QF', 'QED_1', 'QKC', 'QGH']
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in binary_prefixes) and not col.endswith('_Not_Sure'):
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    for col in df.columns:
        if col.startswith('QEA_') or col.startswith('QEB_'):
            df[col] = robust_map(df[col], qea_qeb_map)
        elif col.startswith('QGP_') or col.startswith('QGU_'):
            df[col] = robust_map(df[col], qgp_qgu_map)
        elif col.startswith('QKB_1_') or col.startswith('QKB_2_'):
            df[col] = robust_map(df[col], qkb_map)
        elif col.startswith('QGO_'):
            df[col] = robust_map(df[col], qgo_map)
    if 'QC' in df.columns:
        df['QC'] = robust_map(df['QC'], qc_map)
    if 'QGI' in df.columns:
        df['QGI'] = robust_map(df['QGI'], qgi_map)
    if 'QGS' in df.columns:
        df['QGS'] = robust_map(df['QGS'], qgs_map)
    if 'QGR' in df.columns:
        df['QGR'] = robust_map(df['QGR'], qgr_map)
    if 'QGN' in df.columns:
        df['QGN'] = robust_map(df['QGN'], qgn_map)
    df['Income_Raw'] = np.nan
    for col in qdf_cols:
        df['Income_Raw'] = df['Income_Raw'].fillna(df[col])
    df['Income_Rank'] = np.nan
    if 'QDA' in df.columns:
        for country in df['QDA'].dropna().unique():
            mask = df['QDA'] == country
            unique_incomes = df.loc[mask, 'Income_Raw'].dropna().unique()
            sorted_incomes = sorted(unique_incomes, key=extract_num)
            income_to_rank = {inc: rank for rank, inc in enumerate(sorted_incomes, 1)}
            df.loc[mask, 'Income_Rank'] = df.loc[mask, 'Income_Raw'].map(income_to_rank)
    if 'QEA_2' in df.columns:
        df['Job_Security_Index_Current'] = df['QEA_2']
    if 'QEB_2' in df.columns:
        df['Job_Security_Index_Future'] = df['QEB_2']
    qgp_cols_exist = [c for c in ['QGP_1', 'QGP_2', 'QGP_3'] if c in df.columns]
    if qgp_cols_exist:
        df['Nature_of_Work_Change_Index'] = df[qgp_cols_exist].mean(axis=1)
    qgu_cols_exist = [c for c in ['QGU_1', 'QGU_2', 'QGU_3'] if c in df.columns]
    if qgu_cols_exist:
        df['Future_Nature_of_Work_Change_Index'] = df[qgu_cols_exist].mean(axis=1)
    qkb1_cols_exist = [c for c in df.columns if c.startswith('QKB_1_') and not c.endswith('_Not_Sure')]
    if qkb1_cols_exist:
        df['Org_Enablers_Index'] = df[qkb1_cols_exist].mean(axis=1)
    qgo_cols_exist = [c for c in df.columns if c.startswith('QGO_') and not c.endswith('_Not_Sure')]
    if qgo_cols_exist:
        df['Company_Culture_Index'] = df[qgo_cols_exist].mean(axis=1)
    key_vars = ['QEA_2', 'QEB_2', 'Nature_of_Work_Change_Index', 'Future_Nature_of_Work_Change_Index', 'Org_Enablers_Index', 'Company_Culture_Index', 'Income_Rank', 'QGR', 'QGN', 'QC', 'QGI', 'QGS']
    print('--- Missing Data Rates per Key Variable ---')
    for var in key_vars:
        if var in df.columns:
            missing_rate = df[var].isna().mean() * 100
            print(var + ': ' + str(round(missing_rate, 2)) + '%')
    print('-------------------------------------------\n')
    composite_indices = ['Job_Security_Index_Current', 'Job_Security_Index_Future', 'Nature_of_Work_Change_Index', 'Future_Nature_of_Work_Change_Index', 'Org_Enablers_Index', 'Company_Culture_Index']
    print('--- Distributions of Composite Indices ---')
    for idx in composite_indices:
        if idx in df.columns:
            print('\n' + idx + ' Summary:')
            print(df[idx].describe().to_string())
    print('------------------------------------------\n')
    raw_encoded_path = 'data/data_raw_encoded.csv'
    df.to_csv(raw_encoded_path, index=False)
    print('Raw-encoded data saved to ' + raw_encoded_path)
    nominal_vars = ['QDA', 'QDB', 'QDH', 'HIDDG']
    nominal_vars_exist = [v for v in nominal_vars if v in df.columns]
    df_dummy = pd.get_dummies(df, columns=nominal_vars_exist, drop_first=False)
    dummy_encoded_path = 'data/data_dummy_encoded.csv'
    df_dummy.to_csv(dummy_encoded_path, index=False)
    print('Dummy-encoded data saved to ' + dummy_encoded_path)