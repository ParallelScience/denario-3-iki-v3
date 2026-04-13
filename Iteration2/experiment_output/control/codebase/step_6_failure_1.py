# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def main():
    data_dir = 'data/'
    df_path = os.path.join(data_dir, 'processed_data.csv')
    rf_path = os.path.join(data_dir, 'rf_feature_importance.csv')
    print('Loading datasets...')
    df = pd.read_csv(df_path, low_memory=False)
    rf_df = pd.read_csv(rf_path)
    df['LCA_Class'] = df['LCA_Class'].astype(int)
    class_names = {1: 'Class 1 (Slightly Positive)', 2: 'Class 2 (No Impact)', 3: 'Class 3 (Negative)', 4: 'Class 4 (Sig. Positive)'}
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.rcParams['text.usetex'] = False
    print('Generating Subplot (a)...')
    ax_a = axes[0, 0]
    df_a = df[['LCA_Class', 'QKB_Index']].dropna()
    df_a['QKB_Index'] = df_a['QKB_Index'].astype(float)
    try:
        mnlogit_a = smf.mnlogit('LCA_Class ~ QKB_Index', data=df_a).fit(disp=0)
    except Exception:
        mnlogit_a = smf.mnlogit('LCA_Class ~ QKB_Index', data=df_a).fit(method='bfgs', disp=0)
    try:
        qkb_range = np.linspace(df_a['QKB_Index'].min(), df_a['QKB_Index'].max(), 100)
        pred_df = pd.DataFrame({'QKB_Index': qkb_range})
        preds = mnlogit_a.predict(pred_df)
        classes = sorted(df_a['LCA_Class'].unique())
        for i, col in enumerate(preds.columns):
            class_label = classes[i]
            ax_a.plot(qkb_range, preds[col], label=class_names.get(class_label, 'Class ' + str(class_label)), lw=2.5)
    except Exception as e:
        print('Error in Subplot (a) prediction: ' + str(e))
        ax_a.text(0.5, 0.5, 'Model prediction failed', ha='center', va='center')
    ax_a.set_title('(a) Predicted Probability of Latent Classes by Organizational Enablers', fontsize=14)
    ax_a.set_xlabel('Organizational Enablers Index (QKB_Index) [1-5 Scale]', fontsize=12)
    ax_a.set_ylabel('Predicted Probability [0-1]', fontsize=12)
    ax_a.legend(title='Latent Class', fontsize=10)
    ax_a.grid(True, alpha=0.3)
    print('Generating Subplot (b)...')
    ax_b = axes[0, 1]
    df_b = df[['LCA_Class', 'QKB_1_11', 'QKB_1_4']].dropna()
    df_b['QKB_1_11'] = df_b['QKB_1_11'].astype(float)
    df_b['QKB_1_4'] = df_b['QKB_1_4'].astype(float)
    df_b['Is_Class_4'] = (df_b['LCA_Class'] == 4).astype(int)
    try:
        logit_b = smf.logit('Is_Class_4 ~ QKB_1_11 * QKB_1_4', data=df_b).fit(disp=0)
    except Exception:
        logit_b = smf.logit('Is_Class_4 ~ QKB_1_11 * QKB_1_4', data=df_b).fit(method='bfgs', disp=0)
    try:
        qkb_11_range = np.linspace(1, 5, 50)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for idx, qkb_4_val in enumerate([1, 3, 5]):
            pred_df_b = pd.DataFrame({'QKB_1_11': qkb_11_range, 'QKB_1_4': qkb_4_val})
            preds_b = logit_b.predict(pred_df_b)
            label = 'Low (1)' if qkb_4_val == 1 else ('Medium (3)' if qkb_4_val == 3 else 'High (5)')
            ax_b.plot(qkb_11_range, preds_b, label='Regular Training = ' + label, lw=2.5, color=colors[idx])
    except Exception as e:
        print('Error in Subplot (b) prediction: ' + str(e))
        ax_b.text(0.5, 0.5, 'Model prediction failed', ha='center', va='center')
    ax_b.set_title('(b) Interaction: Employee Involvement x Regular Training\non Probability of "Sig. Positive" Class', fontsize=14)
    ax_b.set_xlabel('Employee Involvement in AI Development (QKB_1_11) [1-5 Scale]', fontsize=12)
    ax_b.set_ylabel('Probability of Class 4 (Sig. Positive) [0-1]', fontsize=12)
    ax_b.legend(title='Regular Training (QKB_1_4)', fontsize=10)
    ax_b.grid(True, alpha=0.3)
    print('Generating Subplot (c)...')
    ax_c = axes[1, 0]
    try:
        rf_pivot = rf_df.pivot(index='Predictor', columns='Outcome', values='Importance_Mean')
        rf_pivot['Average'] = rf_pivot.mean(axis=1)
        rf_top20 = rf_pivot.sort_values('Average', ascending=False).head(20)
        rf_top20 = rf_top20.drop(columns=['Average'])
        sns.heatmap(rf_top20, annot=True, cmap='viridis', fmt='.3f', ax=ax_c, cbar_kws={'label': 'Permutation Importance [R² drop]'})
        ax_c.set_title('(c) Top 20 Predictors: Random Forest Feature Importance', fontsize=14)
        ax_c.set_xlabel('Outcome Variable', fontsize=12)
        ax_c.set_ylabel('Predictor', fontsize=12)
        ax_c.tick_params(axis='y', rotation=0)
    except Exception as e:
        print('Error in Subplot (c): ' + str(e))
        ax_c.text(0.5, 0.5, 'Heatmap generation failed', ha='center', va='center')
    print('Generating Subplot (d)...')
    ax_d = axes[1, 1]
    hiddg_col = 'HIDDG'
    if hiddg_col in df.columns:
        def shorten_profile(x):
            x_str = str(x).lower()
            if 'manager' in x_str: return 'Managerial'
            if 'senior' in x_str or 'director' in x_str or 'c-suite' in x_str or 'cxo' in x_str or 'vp' in x_str: return 'Senior Leadership'
            if 'individual' in x_str: return 'Individual Contributor'
            return 'Other'
        df['Job_Profile_Short'] = df[hiddg_col].apply(shorten_profile)
        df_valid = df[df['Job_Profile_Short'] != 'Other']
        if not df_valid.empty:
            props = df_valid.groupby('Job_Profile_Short')['LCA_Class'].value_counts(normalize=True).unstack()
            for c in [1, 2, 3, 4]:
                if c not in props.columns: props[c] = 0.0
            props = props[[1, 2, 3, 4]]
            props.columns = [class_names.get(c, 'Class ' + str(c)) for c in props.columns]
            order = ['Individual Contributor', 'Managerial', 'Senior Leadership']
            props = props.reindex([x for x in order if x in props.index])
            props.plot(kind='bar', stacked=True, ax=ax_d, colormap='Set2', edgecolor='black')
            ax_d.set_title('(d) Latent Class Distribution by Job Profile', fontsize=14)
            ax_d.set_xlabel('Job Profile', fontsize=12)
            ax_d.set_ylabel('Proportion of Respondents [0-1]', fontsize=12)
            ax_d.legend(title='Latent Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax_d.tick_params(axis='x', rotation=0)
            ax_d.grid(axis='y', alpha=0.3)
        else:
            ax_d.text(0.5, 0.5, 'No valid Job Profile data', ha='center', va='center')
    else:
        ax_d.text(0.5, 0.5, 'Job Profile column not found', ha='center', va='center')
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(data_dir, 'results_visualization_1_' + timestamp + '.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print('\nPlot saved to ' + out_path)
    print('\n' + '='*50)
    print('--- Visualization Summary Statistics ---')
    print('='*50)
    print('Subplot (a) - Predicted Probabilities at QKB_Index extremes:')
    try:
        pred_min = mnlogit_a.predict(pd.DataFrame({'QKB_Index': [1.0]}))
        pred_max = mnlogit_a.predict(pd.DataFrame({'QKB_Index': [5.0]}))
        for i, col in enumerate(pred_min.columns):
            c_name = class_names.get(classes[i], 'Class ' + str(classes[i]))
            val_min = round(pred_min.iloc[0, i], 3)
            val_max = round(pred_max.iloc[0, i], 3)
            print('  ' + c_name + ': QKB=1 -> ' + str(val_min) + ', QKB=5 -> ' + str(val_max))
    except Exception as e:
        print('  Could not compute Subplot (a) stats: ' + str(e))
    print('\nSubplot (b) - Interaction Probabilities for Class 4:')
    try:
        for qkb_4 in [1, 3, 5]:
            p_min = logit_b.predict(pd.DataFrame({'QKB_1_11': [1.0], 'QKB_1_4': [qkb_4]})).iloc[0]
            p_max = logit_b.predict(pd.DataFrame({'QKB_1_11': [5.0], 'QKB_1_4': [qkb_4]})).iloc[0]
            print('  Regular Training=' + str(qkb_4) + ': Emp.Inv=1 -> ' + str(round(p_min, 3)) + ', Emp.Inv=5 -> ' + str(round(p_max, 3)))
    except Exception as e:
        print('  Could not compute Subplot (b) stats: ' + str(e))
    print('\nSubplot (c) - Top 5 Predictors by Average Importance:')
    try:
        print(rf_top20.head(5).to_string())
    except Exception as e:
        print('  Could not compute Subplot (c) stats: ' + str(e))
    print('\nSubplot (d) - Class Proportions by Job Profile:')
    try:
        print(props.to_string())
    except Exception as e:
        print('  Could not compute Subplot (d) stats: ' + str(e))

if __name__ == '__main__':
    main()