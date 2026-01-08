import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def evaluate_performance(csv_path):
    print(f"Evaluating: {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return 0, 0, 0

    df = pd.read_csv(csv_path)
    
    # Cleaning
    cols_to_drop = [c for c in df.columns if "diagnostics" in c]
    df = df.drop(columns=cols_to_drop)
    df = df.dropna(subset=["cdr"])
    
    y = (df["cdr"] > 0).astype(int)
    drop_cols = ["id", "cdr", "mmse"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('selector_var', VarianceThreshold()), 
        ('scaler', StandardScaler()),                 
        ('selector_best', SelectKBest(f_classif, k=15)),   
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) 
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    return scores.mean() * 100, scores.std() * 100, len(df)

def plot_comparison():
    # Dynamic calculation
    wb_acc, wb_std, _ = evaluate_performance("outputs/features/feature_matrix.csv")
    hippo_acc, hippo_std, total_subjects = evaluate_performance("outputs/features/feature_matrix_hippo.csv")
    
    results = {
        'Method': ['Whole Brain', 'Hippocampus ROI'],
        'Accuracy': [wb_acc, hippo_acc],
        'Std Dev': [wb_std, hippo_std]
    }
    
    plt.figure(figsize=(8, 6))
    
    # Bar Chart
    bars = plt.bar(results['Method'], results['Accuracy'], 
                   yerr=results['Std Dev'], # Error bars (Standard Deviation)
                   capsize=10, 
                   color=['#95a5a6', '#2ecc71'], # Gray (Old), Green (New)
                   alpha=0.9, width=0.6)
    
    # Label values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, 
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f"Impact of Hippocampus Focusing (N={total_subjects})", fontsize=14, fontweight='bold')
    plt.ylabel("Model Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotation Box
    improvement = results['Accuracy'][1] - results['Accuracy'][0]
    color_improvement = 'green' if improvement >= 0 else 'red'
    sign = '+' if improvement >= 0 else ''
    
    plt.text(0.5, 90, f"Performance Boost: {sign}{improvement:.1f}%", 
             ha='center', fontsize=12, color=color_improvement, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor=color_improvement, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    output_file = "outputs/figures/final_method_comparison.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Comparison Chart Saved: {output_file}")

if __name__ == "__main__":
    plot_comparison()