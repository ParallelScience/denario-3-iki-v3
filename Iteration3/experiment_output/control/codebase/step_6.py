# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix

def truncate_label(label, max_len=35):
    if len(label) > max_len:
        return label[:max_len] + '...'
    return label

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data'
    timestamp = int(time.time())
    print('Generating LCA class-conditional probabilities heatmap...')
    df3 = pd.read_csv(os.path.join(data_dir, 'processed_data_step3.csv'))
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, col in enumerate(['Job_Security_Current', 'Job_Security_Future']):
        prop_df = df3.groupby(['LCA_Class_Name', col]).size().unstack(fill_value=0)
        prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)
        sns.heatmap(prop_df, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes1[i], vmin=0, vmax=1)
        axes1[i].set_title('Item Probabilities: ' + col)
        axes1[i].set_xlabel('Response Category (-2 to +2)')
        if i == 0:
            axes1[i].set_ylabel('LCA Class')
        else:
            axes1[i].set_ylabel('')
    plt.tight_layout()
    p1_path = os.path.join(data_dir, 'lca_class_conditional_probs_1_' + str(timestamp) + '.png')
    plt.savefig(p1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print('LCA class-conditional probabilities heatmap saved to ' + p1_path)
    print('Generating confusion matrix plot...')
    df5 = pd.read_csv(os.path.join(data_dir, 'processed_data_step5.csv'))
    exog_cols = ['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'HIDDG', 'QDB', 'QDA']
    exog_df = pd.get_dummies(df5[exog_cols], columns=['HIDDG', 'QDB', 'QDA'], drop_first=True, dtype=float)
    exog_df.insert(0, 'Intercept', 1.0)
    logit_model = sm.Logit(df5['Target'], exog_df)
    logit_result = logit_model.fit(disp=0)
    preds_prob = logit_result.predict(exog_df)
    preds_class = (preds_prob > 0.5).astype(int)
    cm = confusion_matrix(df5['Target'], preds_class)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=['Anxiously Declining', 'Resiliently Optimistic'], yticklabels=['Anxiously Declining', 'Resiliently Optimistic'])
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('Actual Class')
    ax2.set_title('Confusion Matrix: Logistic Regression Classifier')
    plt.tight_layout()
    p2_path = os.path.join(data_dir, 'confusion_matrix_2_' + str(timestamp) + '.png')
    plt.savefig(p2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print('Confusion matrix plot saved to ' + p2_path)
    print('Generating predicted probabilities contour plot...')
    df5['QKB_1_4_x_QKB_1_11'] = df5['QKB_1_4'] * df5['QKB_1_11']
    exog_cols_dual = ['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11', 'QKB_1_4_x_QKB_1_11', 'HIDDG', 'QDB']
    exog_df_dual = pd.get_dummies(df5[exog_cols_dual], columns=['HIDDG', 'QDB'], drop_first=True, dtype=float)
    exog_df_dual.insert(0, 'Intercept', 1.0)
    logit_dual = sm.Logit(df5['Target'], exog_df_dual)
    result_dual = logit_dual.fit(disp=0)
    qkb_range = np.linspace(1, 5, 50)
    X_grid, Y_grid = np.meshgrid(qkb_range, qkb_range)
    grid_df = pd.DataFrame({'QKB_1_4': X_grid.flatten(), 'QKB_1_11': Y_grid.flatten()})
    grid_df['QKB_1_4_x_QKB_1_11'] = grid_df['QKB_1_4'] * grid_df['QKB_1_11']
    grid_df['Positive_Affect'] = df5['Positive_Affect'].mean()
    grid_df['Negative_Affect'] = df5['Negative_Affect'].mean()
    for col in exog_df_dual.columns:
        if col not in grid_df.columns:
            grid_df[col] = 0.0
    grid_df = grid_df[exog_df_dual.columns]
    probs = result_dual.predict(grid_df).values.reshape(X_grid.shape)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    contour = ax3.contourf(X_grid, Y_grid, probs, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label('Probability of "Resiliently Optimistic"')
    ax3.set_xlabel('Regular Training (QKB_1_4)')
    ax3.set_ylabel('Employee Involvement (QKB_1_11)')
    ax3.set_title('Predicted Probability of Resiliently Optimistic Class')
    plt.tight_layout()
    p3_path = os.path.join(data_dir, 'predicted_probabilities_3_' + str(timestamp) + '.png')
    plt.savefig(p3_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print('Predicted probabilities plot saved to ' + p3_path)
    print('Generating simple slopes plot...')
    df5['QGM_Centered'] = df5['QGM_Composite'] - df5['QGM_Composite'].mean()
    df5['QKB_1_11_Centered'] = df5['QKB_1_11'] - df5['QKB_1_11'].mean()
    df5['QKB_1_11_x_QGM'] = df5['QKB_1_11_Centered'] * df5['QGM_Centered']
    exog_cols_mod = ['Positive_Affect', 'Negative_Affect', 'QKB_1_4', 'QKB_1_11_Centered', 'QGM_Centered', 'QKB_1_11_x_QGM', 'HIDDG', 'QDB']
    exog_df_mod = pd.get_dummies(df5[exog_cols_mod], columns=['HIDDG', 'QDB'], drop_first=True, dtype=float)
    exog_df_mod.insert(0, 'Intercept', 1.0)
    logit_mod = sm.Logit(df5['Target'], exog_df_mod)
    result_mod = logit_mod.fit(disp=0)
    qgm_sd = df5['QGM_Composite'].std()
    qgm_levels = {'Low (-1 SD)': -qgm_sd, 'Mean': 0, 'High (+1 SD)': qgm_sd}
    qkb_vals = np.linspace(df5['QKB_1_11_Centered'].min(), df5['QKB_1_11_Centered'].max(), 100)
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for label, qgm_val in qgm_levels.items():
        slopes_df = pd.DataFrame({'QKB_1_11_Centered': qkb_vals})
        slopes_df['QGM_Centered'] = qgm_val
        slopes_df['QKB_1_11_x_QGM'] = slopes_df['QKB_1_11_Centered'] * slopes_df['QGM_Centered']
        slopes_df['Positive_Affect'] = df5['Positive_Affect'].mean()
        slopes_df['Negative_Affect'] = df5['Negative_Affect'].mean()
        slopes_df['QKB_1_4'] = df5['QKB_1_4'].mean()
        for col in exog_df_mod.columns:
            if col not in slopes_df.columns:
                slopes_df[col] = 0.0
        slopes_df = slopes_df[exog_df_mod.columns]
        log_odds = np.dot(slopes_df, result_mod.params)
        probs_mod = 1 / (1 + np.exp(-log_odds))
        ax4.plot(qkb_vals + df5['QKB_1_11'].mean(), probs_mod, label='QGM: ' + label, linewidth=2)
    ax4.set_xlabel('Employee Involvement (QKB_1_11)')
    ax4.set_ylabel('Probability of "Resiliently Optimistic"')
    ax4.set_title('Simple Slopes: Moderation by AI Task Comfort (QGM)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    p4_path = os.path.join(data_dir, 'simple_slopes_4_' + str(timestamp) + '.png')
    plt.savefig(p4_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print('Simple slopes plot saved to ' + p4_path)
    print('Generating odds ratios plot...')
    or_df = pd.read_csv(os.path.join(data_dir, 'logistic_regression_results.csv'), index_col=0)
    or_df = or_df[~or_df.index.isin(['Intercept', 'VC_1'])]
    or_df = or_df.sort_values('Odds Ratio')
    or_df.index = [truncate_label(idx) for idx in or_df.index]
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(or_df))
    xerr_lower = or_df['Odds Ratio'] - or_df['OR CI Lower']
    xerr_upper = or_df['OR CI Upper'] - or_df['Odds Ratio']
    xerr_lower = np.maximum(0, xerr_lower)
    xerr_upper = np.maximum(0, xerr_upper)
    ax5.errorbar(or_df['Odds Ratio'], y_pos, xerr=[xerr_lower, xerr_upper], fmt='o', color='black', ecolor='gray', capsize=3)
    ax5.axvline(x=1, color='red', linestyle='--')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(or_df.index)
    ax5.set_xlabel('Odds Ratio (95% CI)')
    ax5.set_title('Predictors of "Resiliently Optimistic" Class Membership')
    ax5.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    p5_path = os.path.join(data_dir, 'odds_ratios_5_' + str(timestamp) + '.png')
    plt.savefig(p5_path, dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print('Odds ratios plot saved to ' + p5_path)