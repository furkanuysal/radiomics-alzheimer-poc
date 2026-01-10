import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

INPUT_PATH = "outputs/features/feature_matrix_hippo.csv"

def run_cross_validation():
    print("--- Advanced Radiomics Model (Cross-Validation) ---")
    
    # Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: File not found -> {INPUT_PATH}")
        print("Please run 'src/feature_extraction_hippo.py' first.")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # Cleanup: Remove metadata columns
    cols_to_drop = [c for c in df.columns if "diagnostics" in c]
    df = df.drop(columns=cols_to_drop)
    
    # Drop rows without CDR value (Training is impossible without labels)
    df = df.dropna(subset=["cdr"])
    
    # Labeling (CDR > 0 -> Patient (1), CDR == 0 -> Healthy (0))
    y = (df["cdr"] > 0).astype(int)
    
    # Input data (Drop columns like ID, CDR, MMSE)
    drop_cols = ["id", "cdr", "mmse"]
    # Check to avoid errors if a column does not exist in the file
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)
    
    print(f"Total Patient Count: {len(X)}")
    print(f"Total Radiomic Feature Count: {X.shape[1]}")
    print("-" * 30)
    print(f"Class Distribution:\n{y.value_counts().rename({0: 'Healthy', 1: 'Alzheimer'})}")
    print("-" * 30)

    # Create Pipeline
    # Step 1: SimpleImputer -> Fill possible NaN values with mean (Safety measure)
    # Step 2: VarianceThreshold -> Drop features that are the same (constant) for all patients (Prevents warnings)
    # Step 3: StandardScaler -> Normalize data
    # Step 4: SelectKBest -> Select best 15 features (Increased from 10 to 15 as data increased)
    # Step 5: RandomForest -> Classification
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('selector_var', VarianceThreshold()), 
        ('scaler', StandardScaler()),                 
        ('selector_best', SelectKBest(f_classif, k=15)),   
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) 
    ])

    # Cross-Validation (5-Fold)
    # Split data into 5 parts, test sequentially.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        # Results
        print("\n--- 🎯 RESULTS ---")
        print(f"Success Per Fold: {scores}")
        print(f"Mean Accuracy: %{scores.mean()*100:.2f}")
        print(f"Standard Deviation: +/- %{scores.std()*100:.2f}")
        
        # Report Best Features
        # Fit pipeline to all data and see which features it selected
        pipeline.fit(X, y)
        
        # We need to find the feature names remaining after VarianceThreshold
        # This part is a bit technical: Names are lost in Pandas -> Numpy conversion, we trace back.
        
        # Which columns did VarianceThreshold keep?
        feature_names = X.columns
        support_var = pipeline.named_steps['selector_var'].get_support()
        feature_names_after_var = feature_names[support_var]
        
        # Which of these did SelectKBest select?
        support_best = pipeline.named_steps['selector_best'].get_support()
        selected_features = feature_names_after_var[support_best]
        
        print("\n🏆 Top 15 Critical Features Selected by Model:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
            
        # If Shape features are present, celebrate!
        shape_count = sum(1 for f in selected_features if "shape" in f)
        if shape_count > 0:
            print(f"\n✨ GREAT! There are {shape_count} 'Shape' (Shape/Volume) features in the list.")
            print("This is the biggest proof that the Hippocampus mask is working correctly.")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        # Give details in case of error
        if len(X) < 5:
            print("Warning: 5-Fold CV cannot run because patient count is less than 5.")

if __name__ == "__main__":
    run_cross_validation()