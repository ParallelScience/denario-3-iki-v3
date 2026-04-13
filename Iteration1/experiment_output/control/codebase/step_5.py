# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = False

def main():
    data_dir = 'data/'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    lca_comp_path = os.path.join(data_dir, 'LCA_Model_Comparison.csv')
    if os.path.exists(lca_comp_path):
        lca_comp = pd.read_csv(lca_comp_path)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        metrics = ['BIC', 'AIC', 'Entropy', 'Log-Likelihood']
        for i, metric in enumerate(metrics):
            axes[i].plot(lca_comp['Classes'], lca_comp[metric], marker='o', linestyle='-', linewidth=2, color='#1f77b4')
            axes[i].set_title('LCA Model Comparison: ' + metric, fontsize=14)
            axes[i].set_xlabel('Number of Classes', fontsize=12)
            axes[i].set_ylabel(metric, fontsize=12)
            axes[i].set_xticks(lca_comp['Classes'])
            axes[i].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'LCA_Model_Comparison_1_' + timestamp + '.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print('Plot saved to ' + plot_path)
        print('\n--- LCA Model Comparison ---')
        print(lca_comp.to_string(index=False))

    lca_data_path = os.path.join(data_dir, 'IKI_Cleaned_Features_LCA.csv')
    if os.path.exists(lca_data_path):
        df_lca = pd.read_csv(lca_data_path, low_memory=False)
        qea_col = next((col for col in df_lca.columns if col.startswith('QEA_2')), None)
        qeb_col = next((col for col in df_lca.columns if col.startswith('QEB_2')), None)
        if qea_col and qeb_col and 'LCA_Class' in df_lca.columns:
            df_valid = df_lca.dropna(subset=['LCA_Class', qea_col, qeb_col])
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            val_map = {-2.0: 'Sig. Neg', -1.0: 'Slight. Neg', 0.0: 'No Impact', 1.0: 'Slight. Pos', 2.0: 'Sig. Pos', 99.0: 'Not Sure'}
            colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
            class_names_plot = {1: 'Class 1\n(Resiliently Optimistic)', 2: 'Class 2\n(Stagnant Neutral)', 3: 'Class 3\n(Anxiously Declining)'}
            class_names_print = {1: 'Class 1 (Resiliently Optimistic)', 2: 'Class 2 (Stagnant Neutral)', 3: 'Class 3 (Anxiously Declining)'}
            print('\n--- Conditional Response Probabilities (%) ---')
            for i, (col, title) in enumerate([(qea_col, 'Current Impact'), (qeb_col, 'Future Impact (3-year)')]):
                xtab = pd.crosstab(df_valid['LCA_Class'], df_valid[col], normalize='index') * 100
                for v in val_map.keys():
                    if v not in xtab.columns:
                        xtab[v] = 0
                xtab = xtab[[-2.0, -1.0, 0.0, 1.0, 2.0, 99.0]]
                xtab.rename(columns=val_map, inplace=True)
                xtab_print = xtab.copy()
                xtab_print.index = xtab_print.index.map(class_names_print)
                print('\n' + title + ':')
                print(xtab_print.round(1).to_string())
                xtab_plot = xtab.copy()
                xtab_plot.index = xtab_plot.index.map(class_names_plot)
                xtab_plot.plot(kind='bar', stacked=True, ax=axes[i], color=colors, edgecolor='black', width=0.7)
                axes[i].set_title('Conditional Response Probabilities: ' + title, fontsize=14)
                axes[i].set_xlabel('Latent Class', fontsize=12)
                axes[i].set_ylabel('Probability (%)', fontsize=12)
                axes[i].tick_params(axis='x', rotation=0)
                if i == 1:
                    axes[i].legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                else:
                    axes[i].get_legend().remove()
            plt.tight_layout()
            plot_path = os.path.join(data_dir, 'LCA_Conditional_Probs_2_' + timestamp + '.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print('\nPlot saved to ' + plot_path)

    mnlogit_path = os.path.join(data_dir, 'MNLogit_Results.csv')
    if os.path.exists(mnlogit_path):
        mnlogit_res = pd.read_csv(mnlogit_path)
        print('\n--- MNLogit Regression Results (Significant Features, FDR < 0.05) ---')
        sig_df = mnlogit_res[mnlogit_res['FDR_P_Value'] < 0.05]
        if not sig_df.empty:
            print(sig_df[['Class', 'Feature', 'Odds_Ratio', 'CI_Lower', 'CI_Upper', 'FDR_P_Value']].to_string(index=False))
        else:
            print('No features with FDR < 0.05 found.')
        print('\n--- MNLogit Regression Results (All Features) ---')
        print(mnlogit_res[['Class', 'Feature', 'Odds_Ratio', 'FDR_P_Value']].to_string(index=False))
        exclude_features = ['const']
        mnlogit_res_plot = mnlogit_res[~mnlogit_res['Feature'].isin(exclude_features)]
        classes = mnlogit_res_plot['Class'].unique()
        fig, axes = plt.subplots(1, len(classes), figsize=(16, 10), sharey=False)
        if len(classes) == 1:
            axes = [axes]
        for i, cls in enumerate(classes):
            df_cls = mnlogit_res_plot[mnlogit_res_plot['Class'] == cls].copy()
            df_cls = df_cls.sort_values('Odds_Ratio', ascending=True)
            y_pos = np.arange(len(df_cls))
            colors = ['red' if p < 0.05 else 'black' for p in df_cls['FDR_P_Value']]
            axes[i].errorbar(df_cls['Odds_Ratio'], y_pos, xerr=[df_cls['Odds_Ratio'] - df_cls['CI_Lower'], df_cls['CI_Upper'] - df_cls['Odds_Ratio']], fmt='o', color='black', ecolor='gray', capsize=4, elinewidth=1.5, markersize=6)
            for j, color in enumerate(colors):
                if color == 'red':
                    axes[i].plot(df_cls['Odds_Ratio'].iloc[j], y_pos[j], 'ro', markersize=6)
            axes[i].axvline(x=1, color='blue', linestyle='--', linewidth=1.5)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(df_cls['Feature'], fontsize=10)
            axes[i].set_title('Odds Ratios: ' + cls + ' vs Stagnant Neutral', fontsize=14)
            axes[i].set_xlabel('Odds Ratio (log scale)', fontsize=12)
            axes[i].set_xscale('log')
            axes[i].grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(data_dir, 'MNLogit_Forest_Plot_3_' + timestamp + '.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print('\nPlot saved to ' + plot_path)

    if os.path.exists(lca_data_path):
        df = pd.read_csv(lca_data_path, low_memory=False)
        df = df.dropna(subset=['LCA_Class'])
        class_mapping = {2: 0, 1: 1, 3: 2}
        df['Target'] = df['LCA_Class'].map(class_mapping)
        qkb_1_4_col = next((c for c in df.columns if c.startswith('QKB_1_4')), None)
        qkb_1_11_col = next((c for c in df.columns if c.startswith('QKB_1_11')), None)
        main_cols = ['QKB_Enablers_Index', 'QGO_Culture_Index', 'Positive_Affect', 'Negative_Affect']
        numeric_cols = ['QKB_Enablers_Index', 'QGO_Culture_Index', 'Positive_Affect', 'Negative_Affect']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if qkb_1_4_col:
            df[qkb_1_4_col] = pd.to_numeric(df[qkb_1_4_col], errors='coerce')
            main_cols.append(qkb_1_4_col)
        if qkb_1_11_col:
            df[qkb_1_11_col] = pd.to_numeric(df[qkb_1_11_col], errors='coerce')
            main_cols.append(qkb_1_11_col)
        placebo_cols = ['Global Employee Size', 'Market Capitalization']
        placebo_cols = [c for c in placebo_cols if c in df.columns]
        include_scale = False
        if os.path.exists(mnlogit_path):
            mnlogit_res = pd.read_csv(mnlogit_path)
            if any('Global Employee Size' in f for f in mnlogit_res['Feature']):
                include_scale = True
        if include_scale:
            main_cols.extend(placebo_cols)
        main_cols = [c for c in main_cols if c in df.columns]
        model_df = df[['Target'] + main_cols].dropna()
        if not model_df.empty:
            X_main = pd.DataFrame(index=model_df.index)
            for c in numeric_cols:
                if c in model_df.columns:
                    X_main[c] = model_df[c]
            X_main['Training'] = model_df[qkb_1_4_col] if qkb_1_4_col and qkb_1_4_col in model_df.columns else 0
            X_main['Involvement'] = model_df[qkb_1_11_col] if qkb_1_11_col and qkb_1_11_col in model_df.columns else 0
            if 'Training' in X_main.columns and 'Involvement' in X_main.columns:
                X_main['Training_x_Involvement'] = X_main['Training'] * X_main['Involvement']
            cat_cols = []
            if include_scale:
                cat_cols.extend(placebo_cols)
            for col in cat_cols:
                if col in model_df.columns:
                    dummies = pd.get_dummies(model_df[col], prefix=col, drop_first=True)
                    X_main = pd.concat([X_main, dummies], axis=1)
            X_main = sm.add_constant(X_main.astype(float))
            y_main = model_df['Target']
            non_const_cols = [c for c in X_main.columns if c == 'const' or X_main[c].nunique() > 1]
            X_main = X_main[non_const_cols]
            variances = X_main.var()
            low_var_cols = variances[variances < 0.0001].index.tolist()
            if 'const' in low_var_cols:
                low_var_cols.remove('const')
            if low_var_cols:
                X_main = X_main.drop(columns=low_var_cols)
            corr_matrix = X_main.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            if to_drop:
                X_main = X_main.drop(columns=to_drop)
            try:
                mnlogit_main = sm.MNLogit(y_main, X_main).fit(disp=0, maxiter=1000, method='lbfgs')
                key_preds = ['QKB_Enablers_Index', 'Positive_Affect', 'Negative_Affect', 'Training']
                key_preds = [p for p in key_preds if p in X_main.columns]
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()
                class_names_dict = {0: 'Stagnant Neutral', 1: 'Resiliently Optimistic', 2: 'Anxiously Declining'}
                colors_dict = {0: '#7f7f7f', 1: '#2ca02c', 2: '#d62728'}
                print('\n--- Marginal Effects (Predicted Probabilities at Min, Mean, Max) ---')
                for i, pred in enumerate(key_preds):
                    if i >= 4: break
                    X_synth = pd.DataFrame(np.tile(X_main.mean().values, (100, 1)), columns=X_main.columns)
                    pred_min, pred_max = X_main[pred].min(), X_main[pred].max()
                    pred_vals = np.linspace(pred_min, pred_max, 100)
                    X_synth[pred] = pred_vals
                    if pred == 'Training' and 'Training_x_Involvement' in X_synth.columns:
                        X_synth['Training_x_Involvement'] = X_synth['Training'] * X_synth['Involvement']
                    probs = mnlogit_main.predict(X_synth)
                    for c_idx in range(3):
                        axes[i].plot(pred_vals, probs[c_idx], label=class_names_dict[c_idx], color=colors_dict[c_idx], linewidth=2.5)
                    axes[i].set_title('Marginal Effect of ' + pred, fontsize=14)
                    axes[i].set_xlabel(pred, fontsize=12)
                    axes[i].set_ylabel('Predicted Probability', fontsize=12)
                    axes[i].grid(True, linestyle='--', alpha=0.7)
                    if i == 0:
                        axes[i].legend(fontsize=10)
                    X_synth_print = pd.DataFrame(np.tile(X_main.mean().values, (3, 1)), columns=X_main.columns)
                    pred_mean = X_main[pred].mean()
                    X_synth_print[pred] = [pred_min, pred_mean, pred_max]
                    if pred == 'Training' and 'Training_x_Involvement' in X_synth_print.columns:
                        X_synth_print['Training_x_Involvement'] = X_synth_print['Training'] * X_synth_print['Involvement']
                    probs_print = mnlogit_main.predict(X_synth_print)
                    print('\nPredictor: ' + pred)
                    print(str('Value').ljust(10) + ' | ' + str('Stagnant Neutral').ljust(20) + ' | ' + str('Resiliently Optimistic').ljust(25) + ' | ' + str('Anxiously Declining').ljust(20))
                    print('-' * 85)
                    labels = ['Min', 'Mean', 'Max']
                    for j in range(3):
                        p0 = probs_print.iloc[j, 0]
                        p1 = probs_print.iloc[j, 1]
                        p2 = probs_print.iloc[j, 2]
                        print(str(labels[j]).ljust(10) + ' | ' + str(round(p0, 4)).ljust(20) + ' | ' + str(round(p1, 4)).ljust(25) + ' | ' + str(round(p2, 4)).ljust(20))
                plt.tight_layout()
                plot_path = os.path.join(data_dir, 'Marginal_Effects_4_' + timestamp + '.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print('\nPlot saved to ' + plot_path)
            except Exception as e:
                print('Could not fit model for marginal effects: ' + str(e))

if __name__ == '__main__':
    main()