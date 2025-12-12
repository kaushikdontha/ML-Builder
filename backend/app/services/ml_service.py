import os
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from app.models.schemas import PipelineRequest, PipelineResult, PipelineStep

UPLOAD_DIR = "uploads"
MODELS_DIR = "generated_models"

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

STEP_TEMPLATES = {
  "standard_scaler": { "label": "Standard Scaler" },
  "min_max_scaler": { "label": "Min Max Scaler" },
  "drop_nulls": { "label": "Drop Missing Values" },
  "logistic_regression": { "label": "Logistic Regression" },
  "decision_tree_classifier": { "label": "Decision Tree Classifier" },
  "train_test_split": {"label": "Train/Test Split"} 
}

def process_pipeline(pipeline: PipelineRequest) -> PipelineResult:
    logs = []
    try:
        start_time = time.time()
        dataset_path = os.path.join(UPLOAD_DIR, pipeline.datasetName)
        
        if not os.path.exists(dataset_path):
             return PipelineResult(
                success=False, 
                message="Dataset not found", 
                metrics={}, 
                logs=[f"Error: File '{pipeline.datasetName}' not found in uploads."]
            )
            
        logs.append(f"Loading dataset '{pipeline.datasetName}'...")
        if dataset_path.lower().endswith('.csv'):
             df = pd.read_csv(dataset_path)
        else:
             df = pd.read_excel(dataset_path)
        
        logs.append(f"Initial shape: {df.shape}")

        test_size = 0.2
        model_artifact = None
        
        # === Preprocessing ===
        for step in pipeline.steps:
            if step.type == "drop_nulls":
                logs.append("Applying Drop Missing Values...")
                shape_before = df.shape
                axis = step.params.get("axis", 0)
                df = df.dropna(axis=axis)
                logs.append(f"Dropped rows/cols. Shape: {shape_before} -> {df.shape}")
                
            elif step.type == "standard_scaler":
                logs.append("Applying Standard Scaler...")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    logs.append(f"Scaled {len(numeric_cols)} numeric columns.")
                else:
                    logs.append("Warning: No numeric columns to scale.")

            elif step.type == "min_max_scaler":
                logs.append("Applying MinMax Scaler...")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    feature_range = step.params.get("feature_range", "0,1")
                    try:
                        fr = tuple(map(int, feature_range.split(',')))
                        scaler = MinMaxScaler(feature_range=fr)
                        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                        logs.append(f"Scaled {len(numeric_cols)} numeric columns to range {fr}.")
                    except:
                         logs.append("Error parsing feature_range, using default (0, 1).")
                         scaler = MinMaxScaler()
                         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                else:
                    logs.append("Warning: No numeric columns to scale.")
            
            elif step.type == "train_test_split":
                ts = step.params.get("test_size", 0.2)
                logs.append(f"Configuration: Train/Test Split set to {ts}")
                test_size = float(ts)

        # === Model Training ===
        model_step = next((s for s in pipeline.steps if s.type in ["logistic_regression", "decision_tree_classifier"]), None)
        
        metrics = {}
        saved_model_filename = None

        if model_step:
            logs.append(f"Preparing Model: {STEP_TEMPLATES[model_step.type]['label']}...")
            
            # Target Selection (Last Column)
            target_col_name = df.columns[-1]
            y = df.iloc[:, -1]
            X_df = df.iloc[:, :-1]
            
            # Feature Encoding (One-Hot for Categorical Features) - Simple implementation
            # Identify object columns in features
            X_categorical = X_df.select_dtypes(include=['object', 'category']).columns
            X_numeric = X_df.select_dtypes(include=[np.number]).columns
            
            if len(X_categorical) > 0:
                logs.append(f"One-Hot Encoding categorical features: {list(X_categorical)}")
                X = pd.get_dummies(X_df, columns=X_categorical, drop_first=True)
            else:
                X = X_df.select_dtypes(include=[np.number])

            logs.append(f"Target Column: {target_col_name}")
            logs.append(f"Feature Columns (Post-Encoding): {list(X.columns)}")
            
            if X.shape[1] == 0:
                 logs.append("Error: No numeric feature columns found/generated.")
                 return PipelineResult(success=False, message="Training failed", metrics={}, logs=logs)

            # Target Encoding
            if y.dtype == 'object' or y.dtype.name == 'category' or isinstance(df.iloc[0, -1], str):
                 le = LabelEncoder()
                 y = le.fit_transform(y.astype(str)) 
                 logs.append(f"Encoded target '{target_col_name}' to integer classes: {list(le.classes_)}")

            logs.append(f"Splitting data with test_size={test_size}...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            logs.append(f"Train Set: {X_train.shape[0]} samples")
            logs.append(f"Test Set: {X_test.shape[0]} samples")

            model = None
            if model_step.type == "logistic_regression":
                C_val = model_step.params.get("C", 1.0)
                model = LogisticRegression(C=float(C_val), random_state=42, max_iter=1000)
            elif model_step.type == "decision_tree_classifier":
                 max_depth_val = model_step.params.get("max_depth", None)
                 max_depth = int(max_depth_val) if max_depth_val else None
                 model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            
            logs.append("Training model...")
            model.fit(X_train, y_train)
            
            logs.append("Evaluating...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                "Accuracy": f"{round(accuracy * 100, 2)}%",
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            logs.append(f"Training completed. Accuracy: {accuracy:.4f}")

             # === 1. Confusion Matrix ===
            try:
                cm = confusion_matrix(y_test, y_pred)
                metrics["confusion_matrix"] = cm.tolist() # [[TP, FP], [FN, TN]] etc
                logs.append(f"Confusion Matrix calculated. Shape: {cm.shape}")
            except Exception as e:
                logs.append(f"Warning: Failed to calc confusion matrix: {str(e)}")

            # === 2. Feature Importance ===
            try:
                feature_names = X.columns.tolist()
                raw_importance = []
                
                if hasattr(model, "feature_importances_"):
                    # Decision Tree
                    raw_importance = model.feature_importances_
                elif hasattr(model, "coef_"):
                    # Logistic Regression (Coef shape: [n_classes, n_features])
                    # We take mean absolute importance across classes
                    raw_importance = np.abs(model.coef_).mean(axis=0)
                
                if len(raw_importance) > 0:
                    # Zip and sort
                    feat_imp = [
                        {"feature": str(name), "importance": float(score)} 
                        for name, score in zip(feature_names, raw_importance)
                    ]
                    # Sort descending
                    feat_imp.sort(key=lambda x: x["importance"], reverse=True)
                    metrics["feature_importance"] = feat_imp
                    logs.append("Feature importance calculated.")
            except Exception as e:
                logs.append(f"Warning: Failed to calc feature importance: {str(e)}")
            
            # Save Model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_model_filename = f"model_{pipeline.datasetName}_{timestamp}.pkl"
            save_path = os.path.join(MODELS_DIR, saved_model_filename)
            joblib.dump(model, save_path)
            logs.append(f"Model saved to {saved_model_filename}")
            
        else:
            logs.append("No model step found. Preprocessing only.")
            metrics = {"rows_processed": df.shape[0], "columns_processed": df.shape[1]}

        duration = time.time() - start_time
        metrics["duration_seconds"] = round(duration, 2)

        return PipelineResult(
            success=True,
            message="Pipeline run successfully",
            metrics=metrics,
            logs=logs,
            model_path=saved_model_filename
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logs.append(f"Critical Error: {str(e)}")
        return PipelineResult(
            success=False,
            message="Pipeline failed",
            metrics={},
            logs=logs
        )
