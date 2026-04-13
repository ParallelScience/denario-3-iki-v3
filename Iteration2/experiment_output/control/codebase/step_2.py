# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def cronbach_alpha(df_items):
    """
    Calculate Cronbach's alpha for a dataframe of items to measure internal consistency.
    
    Parameters:
    df_items (pd.DataFrame): DataFrame containing the items (columns) for the index.
    
    Returns:
    float: Cronbach's alpha value.
    """
    df_items = df_items.dropna()
    k = df_items.shape[1]
    if k <= 1:
        return np.nan
    var_items = df_items.var(axis=0, ddof=1).sum()
    var_total = df_items.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    return (k / (k - 1)) * (1 - var_items / var_total)

if __name__ == '__main__':
    data_dir = "data/"
    file_path = os.path.join(data_dir, "processed_data.csv")
    
    print("Loading processed dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    
    qgo_cols = [c for c in df.columns if c.startswith('QGO_') and c != 'QGO_Index']
    qkb_cols = [c for c in df.columns if c.startswith('QKB_1_') and c not in ['QKB_Index', 'QKB_Index_Reduced']]
    qkb_reduced_cols = [c for c in qkb_cols if c not in ['QKB_1_4', 'QKB_1_11']]
    
    results = []
    results.append("--- Index Validation Results ---")
    
    if qgo_cols:
        alpha_qgo = cronbach_alpha(df[qgo_cols])
        res_str = "Cronbach's alpha for QGO_Index (items=" + str(len(qgo_cols)) + "): " + str(round(alpha_qgo, 3))
        print(res_str)
        results.append(res_str)
        
    if qkb_cols:
        alpha_qkb = cronbach_alpha(df[qkb_cols])
        res_str = "Cronbach's alpha for QKB_Index (items=" + str(len(qkb_cols)) + "): " + str(round(alpha_qkb, 3))
        print(res_str)
        results.append(res_str)
        
    if qkb_reduced_cols:
        alpha_qkb_red = cronbach_alpha(df[qkb_reduced_cols])
        res_str = "Cronbach's alpha for QKB_Index_Reduced (items=" + str(len(qkb_reduced_cols)) + "): " + str(round(alpha_qkb_red, 3))
        print(res_str)
        results.append(res_str)
        
    results.append("--------------------------------")
    
    out_file = os.path.join(data_dir, "index_validation_results.txt")
    with open(out_file, "w") as f:
        f.write("\n".join(results) + "\n")
        
    print("Validation results saved to " + out_file)