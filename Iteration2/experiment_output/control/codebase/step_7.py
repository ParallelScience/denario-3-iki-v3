# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

def find_col(df, prefix):
    for c in df.columns:
        if c.startswith(prefix):
            return c
    return None

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    data_dir = "data/"
    timestamp = int(time.time())
    
    file_path = os.path.join(data_dir, "processed_data.csv")
    print("Loading processed dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    
    qea2_col = find_col(df, 'QEA_2:')
    qeb2_col = find_col(df, 'QEB_2:')
    
    print("\n--- 1. LCA Metrics ---")
    X_lca = df[[qea2_col, qeb2_col]].dropna()
    bics = []
    aics = []
    entropies = []
    k_range = range(2, 6)
    
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_lca)
        bics.append(gmm.bic(X_lca))
        aics.append(gmm.aic(X_lca))
        probs = gmm.predict_proba(X_lca)
        entropy = -np.sum(probs * np.log(probs + 1e-10)) / len(X_lca)
        entropies.append(entropy)
        print("k=" + str(k) + ": AIC=" + str(round(aics[-1], 2)) + ", BIC=" + str(round(bics[-1], 2)) + ", Entropy=" + str(round(entropies[-1], 4)))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(list(k_range), bics, marker='o', color='blue')
    axes[0].set_title('BIC')
    axes[0].set_xlabel('Number of Classes')
    axes[0].set_ylabel('BIC Score')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].plot(list(k_range), aics, marker='o', color='green')
    axes[1].set_title('AIC')
    axes[1].set_xlabel('Number of Classes')
    axes[1].set_ylabel('AIC Score')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    axes[2].plot(list(k_range), entropies, marker='o', color='red')
    axes[2].set_title('Entropy')
    axes[2].set_xlabel('Number of Classes')
    axes[2].set_ylabel('Entropy')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=2.0)
    lca_metrics_path = os.path.join(data_dir, "lca_metrics_1_" + str(timestamp) + ".png")
    plt.savefig(lca_metrics_path, dpi=300)
    plt.close()
    print("LCA metrics plot saved to " + lca_metrics_path)
    
    print("\n--- 2. Class-Conditional Item-Response Probabilities ---")
    if 'LCA_Class' in df.columns and not df['LCA_Class'].isna().all():
        categories = [-2.0, -1.0, 0.0, 1.0, 2.0]
        cat_labels = ['Sig. Neg', 'Slight Neg', 'No Impact', 'Slight Pos', 'Sig. Pos']
        heatmap_data = []
        row_labels = []
        unique_classes = sorted(df['LCA_Class'].dropna().unique())
        for c in unique_classes:
            class_df = df[df['LCA_Class'] == c]
            row = []
            for item in [qea2_col, qeb2_col]:
                counts = class_df[item].value_counts(normalize=True)
                for cat in categories:
                    row.append(counts.get(cat, 0.0))
            heatmap_data.append(row)
            row_labels.append("Class " + str(int(c)))
        col_labels = ["QEA_2: " + l for l in cat_labels] + ["QEB_2: " + l for l in cat_labels]
        for i, row in enumerate(heatmap_data):
            print("Class " + str(int(unique_classes[i])) + ":")
            for j, val in enumerate(row):
                print("  " + col_labels[j] + ": " + str(round(val, 4)))
        plt.figure(figsize=(14, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='Blues', xticklabels=col_labels, yticklabels=row_labels, fmt=".2f")
        plt.title('Class-Conditional Item-Response Probabilities')
        plt.xticks(rotation=90, ha='center', fontsize=9)
        plt.tight_layout()
        heatmap_path = os.path.join(data_dir, "lca_heatmap_2_" + str(timestamp) + ".png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print("Heatmap saved to " + heatmap_path)
    else:
        print("LCA_Class not found or empty. Skipping heatmap.")
        
    print("\n--- 3. Predicted Probability Curves ---")
    if 'LCA_Class' in df.columns and 'QKB_Index' in df.columns and 'QGO_Index' in df.columns:
        df_pred = df[['LCA_Class', 'QKB_Index', 'QGO_Index']].dropna()
        if len(df_pred) > 0:
            X_pred = df_pred[['QKB_Index', 'QGO_Index']]
            y_pred = df_pred['LCA_Class']
            clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            clf.fit(X_pred, y_pred)
            qkb_range = np.linspace(X_pred['QKB_Index'].min(), X_pred['QKB_Index'].max(), 100)
            qgo_mean = X_pred['QGO_Index'].mean()
            X_sim_qkb = pd.DataFrame({'QKB_Index': qkb_range, 'QGO_Index': qgo_mean})
            probs_qkb = clf.predict_proba(X_sim_qkb)
            qgo_range = np.linspace(X_pred['QGO_Index'].min(), X_pred['QGO_Index'].max(), 100)
            qkb_mean = X_pred['QKB_Index'].mean()
            X_sim_qgo = pd.DataFrame({'QKB_Index': qkb_mean, 'QGO_Index': qgo_range})
            probs_qgo = clf.predict_proba(X_sim_qgo)
            print("Predicted Probabilities across QKB_Index (at mean QGO_Index=" + str(round(qgo_mean, 2)) + "):")
            for i, c in enumerate(clf.classes_):
                print("  Class " + str(c) + ":")
                for qkb_val in [X_pred['QKB_Index'].min(), X_pred['QKB_Index'].median(), X_pred['QKB_Index'].max()]:
                    X_sim_single = pd.DataFrame({'QKB_Index': [qkb_val], 'QGO_Index': [qgo_mean]})
                    prob = clf.predict_proba(X_sim_single)[0, i]
                    print("    QKB_Index = " + str(round(qkb_val, 2)) + " -> Probability = " + str(round(prob, 4)))
            print("Predicted Probabilities across QGO_Index (at mean QKB_Index=" + str(round(qkb_mean, 2)) + "):")
            for i, c in enumerate(clf.classes_):
                print("  Class " + str(c) + ":")
                for qgo_val in [X_pred['QGO_Index'].min(), X_pred['QGO_Index'].median(), X_pred['QGO_Index'].max()]:
                    X_sim_single = pd.DataFrame({'QKB_Index': [qkb_mean], 'QGO_Index': [qgo_val]})
                    prob = clf.predict_proba(X_sim_single)[0, i]
                    print("    QGO_Index = " + str(round(qgo_val, 2)) + " -> Probability = " + str(round(prob, 4)))
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for i, c in enumerate(clf.classes_):
                axes[0].plot(qkb_range, probs_qkb[:, i], label="Class " + str(c), linewidth=2)
            axes[0].set_title('Predicted Probabilities across QKB_Index\n(QGO_Index at mean)')
            axes[0].set_xlabel('QKB_Index (Organizational Enablers)')
            axes[0].set_ylabel('Probability')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.7)
            for i, c in enumerate(clf.classes_):
                axes[1].plot(qgo_range, probs_qgo[:, i], label="Class " + str(c), linewidth=2)
            axes[1].set_title('Predicted Probabilities across QGO_Index\n(QKB_Index at mean)')
            axes[1].set_xlabel('QGO_Index (Company Culture)')
            axes[1].set_ylabel('Probability')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            prob_curves_path = os.path.join(data_dir, "predicted_probabilities_3_" + str(timestamp) + ".png")
            plt.savefig(prob_curves_path, dpi=300)
            plt.close()
            print("Predicted probability curves saved to " + prob_curves_path)
            
    print("\n--- 4. Interaction Plot ---")
    qkb4_col = find_col(df, 'QKB_1_4:')
    qkb11_col = find_col(df, 'QKB_1_11:')
    if 'LCA_Class' in df.columns and qkb4_col and qkb11_col:
        df['Security_Sum'] = df[qea2_col] + df[qeb2_col]
        class_sums = df.groupby('LCA_Class')['Security_Sum'].mean()
        optimistic_class = class_sums.idxmax()
        print("Identified 'Resiliently Optimistic' class as Class " + str(optimistic_class) + " (highest mean security sum).")
        df_int = df[['LCA_Class', qkb4_col, qkb11_col]].dropna()
        df_int['Is_Optimistic'] = (df_int['LCA_Class'] == optimistic_class).astype(int)
        X_int = df_int[[qkb4_col, qkb11_col]].copy()
        X_int['Interaction'] = X_int[qkb4_col] * X_int[qkb11_col]
        y_int = df_int['Is_Optimistic']
        clf_int = LogisticRegression(solver='lbfgs')
        clf_int.fit(X_int, y_int)
        qkb11_range = np.linspace(X_int[qkb11_col].min(), X_int[qkb11_col].max(), 100)
        
        unique_levels = []
        labels = []
        for lvl, name in zip([X_int[qkb4_col].min(), X_int[qkb4_col].median(), X_int[qkb4_col].max()], ['Minimum', 'Median', 'Maximum']):
            if lvl not in unique_levels:
                unique_levels.append(lvl)
                labels.append(name + " (" + str(lvl) + ")")
                
        print("Interaction Plot Data (Probability of Resiliently Optimistic Class):")
        for level, label in zip(unique_levels, labels):
            print("  Regular Training (QKB_1_4) = " + label + ":")
            for qkb11_val in [X_int[qkb11_col].min(), X_int[qkb11_col].median(), X_int[qkb11_col].max()]:
                X_sim_single = pd.DataFrame({qkb4_col: [level], qkb11_col: [qkb11_val], 'Interaction': [level * qkb11_val]})
                prob = clf_int.predict_proba(X_sim_single)[0, 1]
                print("    Employee Involvement (QKB_1_11) = " + str(round(qkb11_val, 2)) + " -> Probability = " + str(round(prob, 4)))
        plt.figure(figsize=(8, 6))
        for level, label in zip(unique_levels, labels):
            X_sim = pd.DataFrame({qkb4_col: level, qkb11_col: qkb11_range, 'Interaction': level * qkb11_range})
            probs = clf_int.predict_proba(X_sim)[:, 1]
            plt.plot(qkb11_range, probs, label="Regular Training: " + label, linewidth=2)
        plt.title('Interaction of Regular Training and Employee Involvement\non Probability of "Resiliently Optimistic" Class')
        plt.xlabel('Employee Involvement in Development (QKB_1_11)')
        plt.ylabel('Probability of Resiliently Optimistic Class')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        interaction_path = os.path.join(data_dir, "interaction_plot_4_" + str(timestamp) + ".png")
        plt.savefig(interaction_path, dpi=300)
        plt.close()
        print("Interaction plot saved to " + interaction_path)
        
    print("\n--- 5. Random Forest Permutation Importances ---")
    rf_file = os.path.join(data_dir, "rf_importance_Job_Security_Future.csv")
    if os.path.exists(rf_file):
        df_rf = pd.read_csv(rf_file)
        print("Top 20 Random Forest Permutation Importances (Job Security Future):")
        print(df_rf[['Predictor', 'Importance_Mean', 'Importance_Std']].to_string(index=False))
        df_rf = df_rf.sort_values('Importance_Mean', ascending=True)
        
        df_rf['Predictor_Trunc'] = df_rf['Predictor'].apply(lambda x: str(x)[:57] + '...' if len(str(x)) > 60 else str(x))
        
        plt.figure(figsize=(12, 10))
        plt.barh(df_rf['Predictor_Trunc'], df_rf['Importance_Mean'], xerr=df_rf['Importance_Std'], capsize=4, color='skyblue', edgecolor='black')
        plt.title('Top 20 Random Forest Permutation Importances\n(Target: Job Security Future)')
        plt.xlabel('Mean Permutation Importance')
        plt.ylabel('Predictor')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        rf_plot_path = os.path.join(data_dir, "rf_importance_5_" + str(timestamp) + ".png")
        plt.savefig(rf_plot_path, dpi=300)
        plt.close()
        print("Random Forest importance plot saved to " + rf_plot_path)
    else:
        print("File " + rf_file + " not found. Skipping RF plot.")

if __name__ == '__main__':
    main()