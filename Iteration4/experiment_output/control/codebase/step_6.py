# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def main():
    data_dir = 'data/'
    df_proc = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'), low_memory=False)
    df_policy = pd.read_csv(os.path.join(data_dir, 'policy_lift_probabilities.csv'), index_col=0)
    df_ame = pd.read_csv(os.path.join(data_dir, 'average_marginal_effects.csv'))
    df_ame['Class'] = df_ame['Class'].replace({'Resiliently Optimistic': 'Anxiously Declining'})
    df_ame['Predictor'] = df_ame['Predictor'].str.replace('QKB_1_4 ', '').str.replace('QKB_1_11 ', '')
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax = axes[0]
    class_counts = df_proc['LCA_Class'].dropna().value_counts(normalize=True) * 100
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, palette='viridis', hue=class_counts.index, legend=False)
    ax.set_title('A. LCA Class Distribution', fontsize=15, pad=15)
    ax.set_ylabel('Percentage of Respondents (%)', fontsize=13)
    ax.set_xlabel('Latent Class', fontsize=13)
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=0, fontsize=11)
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 1, str(round(v, 1)) + '%', ha='center', va='bottom', fontsize=12)
    ax.set_ylim(0, max(class_counts.values) * 1.15)
    ax = axes[1]
    sns.heatmap(df_policy, annot=True, cmap='Blues', fmt='.4f', ax=ax, cbar_kws={'label': 'Predicted Probability'}, annot_kws={'size': 12})
    ax.set_title('B. Dual-Pillar Interaction Heatmap', fontsize=15, pad=15)
    ax.set_ylabel('Enabler Level Combination\n(Training, Involvement)', fontsize=13)
    ax.set_xlabel('Latent Class', fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax = axes[2]
    sns.stripplot(data=df_ame, y='Predictor', x='AME', hue='Class', ax=ax, size=12, jitter=False, dodge=True, palette='Set1')
    ax.set_title('C. Average Marginal Effects (AME)', fontsize=15, pad=15)
    ax.set_xlabel('Average Marginal Effect (Δ Probability)', fontsize=13)
    ax.set_ylabel('Organizational Enabler', fontsize=13)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.legend(title='Latent Class', loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
    plt.tight_layout(pad=3.0, w_pad=4.0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'lca_results_summary_3_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print('--- Data Summary for Visualization ---')
    print('\nLCA Class Distribution (%):')
    print(class_counts.round(2).to_string())
    print('\nPolicy-Lift Predicted Probabilities:')
    print(df_policy.round(4).to_string())
    print('\nAverage Marginal Effects:')
    print(df_ame.round(5).to_string())
    print('\nPlot saved to ' + plot_filepath)

if __name__ == '__main__':
    main()