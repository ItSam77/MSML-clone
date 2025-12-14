import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
import os
warnings.filterwarnings("ignore")

# Set tracking URI ke folder kriteria2 agar mlruns dan mlartifacts ada di sini
current_dir = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"file:///{current_dir}/mlruns")
mlflow.set_experiment("Hyperparameter Tuning Random Forest")

# Buat folder artifacts di dalam kriteria2
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

df = pd.read_csv("preprocess.csv")

x = df.drop("quality_label", axis=1)
y = df["quality_label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

## Biasanya disini dilakukan sampling/scalling untuk data train

# Define hyperparameter grid untuk Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

with mlflow.start_run(run_name="Hyperparameter Tuning Random Forest"):
    # Initialize base model
    rf_base = RandomForestClassifier(random_state=42)
    
    # GridSearchCV untuk mencari hyperparameter terbaik
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,  # 5-fold cross validation
        n_jobs=-1,  # menggunakan semua core CPU
        verbose=2,
        scoring='accuracy'
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(x_train, y_train)
    
    # Best parameters
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    print(f"\nBest Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Manual log best parameters ke MLflow
    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)
    
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Use best model untuk prediksi
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    
    # Evaluation metrics
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT:")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print("PERFORMANCE METRICS:")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*50)
    
    # Manual log metrics ke MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    # ========== ARTEFAK TAMBAHAN 1: CONFUSION MATRIX ==========
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix - Random Forest (Tuned)', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(confusion_matrix_path)
    plt.close()
    print(f"✓ Confusion Matrix saved and logged")
    
    # ========== ARTEFAK TAMBAHAN 2: FEATURE IMPORTANCE ==========
    print("\nGenerating Feature Importance Plot...")
    feature_importance = pd.DataFrame({
        'feature': x.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance - Random Forest (Tuned)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    feature_importance_path = os.path.join(artifacts_dir, "feature_importance.png")
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(feature_importance_path)
    plt.close()
    print(f"✓ Feature Importance Plot saved and logged")
    
    # ========== ARTEFAK TAMBAHAN 3: CV RESULTS ==========
    print("\nSaving Cross-Validation Results...")
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_path = os.path.join(artifacts_dir, "cv_results.csv")
    cv_results_df.to_csv(cv_results_path, index=False)
    mlflow.log_artifact(cv_results_path)
    print(f"✓ CV Results saved and logged")
    
    # ========== ARTEFAK TAMBAHAN 4: HYPERPARAMETER COMPARISON PLOT ==========
    print("\nGenerating Hyperparameter Comparison Plot...")
    # Ambil top 10 kombinasi hyperparameter
    top_10_results = cv_results_df.nlargest(10, 'mean_test_score')
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_10_results)), top_10_results['mean_test_score'])
    plt.yticks(range(len(top_10_results)), 
               [f"Config {i+1}" for i in range(len(top_10_results))])
    plt.xlabel('Mean CV Score', fontsize=12)
    plt.ylabel('Configuration', fontsize=12)
    plt.title('Top 10 Hyperparameter Configurations', fontsize=14, fontweight='bold')
    plt.xlim([top_10_results['mean_test_score'].min() - 0.01, 
              top_10_results['mean_test_score'].max() + 0.01])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    hyperparameter_comparison_path = os.path.join(artifacts_dir, "hyperparameter_comparison.png")
    plt.savefig(hyperparameter_comparison_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(hyperparameter_comparison_path)
    plt.close()
    print(f"✓ Hyperparameter Comparison Plot saved and logged")
    
    # Log feature importance sebagai metric tambahan (top 3 features)
    for idx, row in feature_importance.head(3).iterrows():
        # Clean feature name untuk MLflow (replace invalid characters)
        clean_feature_name = str(row['feature']).replace(':', '_').replace(' ', '_')
        mlflow.log_metric(f"importance_{clean_feature_name}", row['importance'])
    
    # Manual log model ke MLflow
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    print(f"\n✓ Model saved and logged")
    print(f"📁 All artifacts saved in: {os.path.abspath(artifacts_dir)}")
    print(f"📁 MLflow tracking data in: {os.path.abspath('mlruns')}")

print("\n" + "="*50)
print("Hyperparameter tuning completed!")
print("="*50)

