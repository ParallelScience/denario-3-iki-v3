# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os
import re

def extract_income_lower_bound(s):
    if pd.isna(s): return np.nan
    s_str = str(s).lower()
    if 'prefer not to say' in s_str or 'don\'t know' in s_str:
        return np.nan
    s_clean = s_str.replace(',', '').replace('.', '')
    numbers = re.findall(r'\d+', s_clean)
    if numbers:
        return int(numbers[0])
    return np.nan

def apply_mapping(series, mapping):
    s_lower = series.dropna().astype(str).str.strip().str.lower()
    return s_lower.map(mapping).reindex(series.index)

def map_qkb(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().lower()
    if val_str.isdigit():
        return int(val_str)
    if 'not at all' in val_str: return 1
    if 'slightly' in val_str: return 2
    if 'moderately' in val_str or 'somewhat' in val_str: return 3
    if 'very' in val_str: return 4
    if 'extremely' in val_str: return 5
    if 'never' in val_str or 'strongly disagree' in val_str: return 1
    if 'rarely' in val_str or 'somewhat disagree' in val_str: return 2
    if 'sometimes' in val_str or 'neither' in val_str: return 3
    if 'often' in val_str or 'somewhat agree' in val_str: return 4
    if 'always' in val_str or 'strongly agree' in val_str: return 5
    return np.nan

def map_qc(val):
    if pd.isna(val): return np.nan
    x = str(val).strip().lower()
    if x.isdigit(): return int(x)
    if 'once a month' in x: return 1
    if 'few times a month' in x or 'couple of times a month' in x: return 2
    if 'once a week' in x: return 3
    if 'few times a week' in x or 'several times a week' in x: return 4
    if 'once a day' in x or 'daily' in x: return 5
    if 'few times a day' in x or 'several times a day' in x: return 6
    if 'many times a day' in x or 'multiple times a day' in x: return 7
    return np.nan

def map_qgi(val):
    if pd.isna(val): return np.nan
    x = str(val).strip().lower()
    if x.lstrip('-').isdigit(): return int(x)
    if 'increases' in x or 'more time' in x: return -1
    if 'none' in x or '0' in x: return 0
    if '<1' in x or 'less than 1' in x: return 1
    if '1-3' in x or '1 to 3' in x or '1–3' in x: return 2
    if '3-5' in x or '3 to 5' in x or '3–5' in x: return 3
    if '>5' in x or 'more than 5' in x: return 4
    return np.nan

def map_qgs(val):
    if pd.isna(val): return np.nan
    x = str(val).strip().lower()
    if x.isdigit(): return int(x)
    if 'never' in x: return 0
    if 'rarely' in x: return 1
    if 'sometimes' in x or 'occasionally' in x: return 2
    if 'often' in x or 'frequently' in x: return 3
    if 'constantly' in x or 'always' in x: return 4
    return np.nan

def map_qgn(val):
    if pd.isna(val): return np.nan
    x = str(val).strip().lower()
    if x.isdigit(): return int(x)
    if "doesn't use" in x or "does not use" in x: return 1
    if "exploring" in x or "evaluating" in x: return 2
    if "using in some" in x or "limited" in x: return 3
    if "using in many" in x or "broadly" in x: return 4
    if "transforming" in x or "core" in x: return 5
    return np.nan

if __name__ == '__main__':
    file_path = '/home/node/work/projects/iki_v2/IKI-Data-Raw.csv'
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df.replace('.', np.nan, inplace=True)
    df['QEA_2_raw'] = df['QEA_2']
    df['QEB_2_raw'] = df['QEB_2']
    print("Frequency table for QEA_2 (raw):")
    print(df['QEA_2_raw'].value_counts(dropna=False).to_string())
    print("\nFrequency table for QEB_2 (raw):")
    print(df['QEB_2_raw'].value_counts(dropna=False).to_string())
    binary_prefixes = ['QHD', 'QGM', 'QA_1', 'QGL', 'QF', 'QED_1', 'QKC', 'QGH']
    binary_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in binary_prefixes)]
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() == 'yes' else 0)
    if 'QKD' in df.columns:
        df['QKD'] = df['QKD'].apply(lambda x: 1 if pd.notna(x) and str(x).strip().lower() == 'yes' else (0 if pd.notna(x) and str(x).strip().lower() == 'no' else np.nan))
    qea_qeb_map = {'significantly negative': -2, 'slightly negative': -1, 'no impact': 0, 'slightly positive': 1, 'significantly positive': 2, 'not sure': np.nan}
    qea_cols = [col for col in df.columns if col.startswith('QEA_') and col != 'QEA_2_raw']
    qeb_cols = [col for col in df.columns if col.startswith('QEB_') and col != 'QEB_2_raw']
    for col in qea_cols + qeb_cols:
        df[col] = apply_mapping(df[col], qea_qeb_map)
    qgp_qgu_map = {'large decrease': -2, 'moderate decrease': -1, 'no major change': 0, 'moderate increase': 1, 'large increase': 2}
    qgp_cols = [col for col in df.columns if col.startswith('QGP_')]
    qgu_cols = [col for col in df.columns if col.startswith('QGU_')]
    for col in qgp_cols + qgu_cols:
        df[col] = apply_mapping(df[col], qgp_qgu_map)
    qgo_map = {'strongly disagree': 1, 'somewhat disagree': 2, 'neither agree nor disagree': 3, 'somewhat agree': 4, 'strongly agree': 5}
    qgo_cols = [col for col in df.columns if col.startswith('QGO_')]
    for col in qgo_cols:
        df[col] = apply_mapping(df[col], qgo_map)
    qgr_map = {'not at all confident': 1, 'slightly confident': 2, 'somewhat confident': 3, 'very confident': 4, 'not sure': np.nan}
    if 'QGR' in df.columns:
        df['QGR'] = apply_mapping(df['QGR'], qgr_map)
    qkb_cols = [col for col in df.columns if col.startswith('QKB_')]
    for col in qkb_cols:
        df[col] = df[col].apply(map_qkb)
    if 'QC' in df.columns:
        df['QC'] = df['QC'].apply(map_qc)
    if 'QGI' in df.columns:
        df['QGI'] = df['QGI'].apply(map_qgi)
    if 'QGS' in df.columns:
        df['QGS'] = df['QGS'].apply(map_qgs)
    if 'QGN' in df.columns:
        df['QGN'] = df['QGN'].apply(map_qgn)
    qdf_cols = [col for col in df.columns if col.startswith('QDF_')]
    if qdf_cols:
        df['Income_String'] = df[qdf_cols].bfill(axis=1).iloc[:, 0]
        df['QDF_Rank'] = np.nan
        if 'QDA' in df.columns:
            for country in df['QDA'].dropna().unique():
                mask = df['QDA'] == country
                unique_incomes = df.loc[mask, 'Income_String'].dropna().unique()
                valid_incomes = [inc for inc in unique_incomes if pd.notna(extract_income_lower_bound(inc))]
                sorted_incomes = sorted(valid_incomes, key=extract_income_lower_bound)
                inc_map = {inc: rank + 1 for rank, inc in enumerate(sorted_incomes)}
                df.loc[mask, 'QDF_Rank'] = df.loc[mask, 'Income_String'].map(inc_map)
    df['Nature_of_Work_Change_Index'] = df[['QGP_1', 'QGP_2', 'QGP_3']].mean(axis=1)
    df['Future_Nature_of_Work_Change_Index'] = df[['QGU_1', 'QGU_2', 'QGU_3']].mean(axis=1)
    continuous_cols = ['QDC', 'QDE_Year', 'QC']
    for col in continuous_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col + '_std'] = (df[col] - mean) / std
            else:
                df[col + '_std'] = 0.0
    print("\nFrequency table for QEA_2 (encoded):")
    print(df['QEA_2'].value_counts(dropna=False).to_string())
    print("\nFrequency table for QEB_2 (encoded):")
    print(df['QEB_2'].value_counts(dropna=False).to_string())
    print("\nSummary of Composite Indices:")
    print(df[['Nature_of_Work_Change_Index', 'Future_Nature_of_Work_Change_Index']].describe().to_string())
    print("\nSummary of Standardized Predictors:")
    std_cols = [col + '_std' for col in continuous_cols if col + '_std' in df.columns]
    print(df[std_cols].describe().to_string())
    print("\nIncome Rank Summary by Country:")
    if 'QDA' in df.columns and 'QDF_Rank' in df.columns:
        print(df.groupby('QDA')['QDF_Rank'].describe().to_string())
    output_path = os.path.join('data', 'processed_data.csv')
    df.to_csv(output_path, index=False)
    print("\nProcessed dataset saved to " + output_path)