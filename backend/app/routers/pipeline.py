from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.models.schemas import PipelineRequest, PipelineResult, DatasetResponse
from app.services.ml_service import process_pipeline
import shutil
import os
import pandas as pd
import numpy as np

router = APIRouter()

UPLOAD_DIR = "uploads"
MODELS_DIR = "generated_models"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/api/upload", response_model=DatasetResponse)
def upload_file(file: UploadFile = File(...)):
    print(f"[{os.getpid()}] Starting upload for: {file.filename}")
    
    # Validate file extension
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload CSV or Excel.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save file to disk (Stream to avoid memory spike)
        print(f"[{os.getpid()}] Saving file to disk...")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[{os.getpid()}] File saved. Size: {os.path.getsize(file_path)} bytes")
            
        # Parse with Pandas (Memory Optimization: Load only necessary rows for preview first if large)
        print(f"[{os.getpid()}] Reading dataframe...")
        if file.filename.lower().endswith('.csv'):
            # Check file size, if huge, warn or handle
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"[{os.getpid()}] Dataframe loaded. Shape: {df.shape}")

        # Extract Metadata
        df_display = df.where(pd.notnull(df), None)
        preview_data = df_display.head(5).to_dict(orient='records')
        
        # --- Visualizations Calculation (Limit memory usage) ---
        distributions = {}
        correlations = None
        scatter_data = None
        
        if df.shape[1] > 0:
            target_col = df.columns[-1]
            try:
                # Optimized Value Counts (Top 10)
                val_counts = df[target_col].value_counts().head(10).to_dict()
                val_counts = {str(k): int(v) for k, v in val_counts.items()}
                distributions = {"target_column": target_col, "data": val_counts}
            except Exception as e:
                print(f"Dist calc error: {e}")
            
            # Select Numeric Columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            # 2. Correlations (Limit to first 1000 rows for speed/memory if large)
            if numeric_df.shape[1] >= 2:
                try:
                    # Sampling for correlation if dataset is large to save memory
                    if len(numeric_df) > 5000:
                        corr_df = numeric_df.sample(n=5000, random_state=42)
                    else:
                        corr_df = numeric_df
                    
                    corr_matrix = corr_df.corr().round(2)
                    correlations = {
                        "labels": corr_matrix.columns.tolist(),
                        "data": corr_matrix.values.tolist()
                    }
                except Exception as e:
                    print(f"Corr calc error: {e}")
            
            # 3. Scatter Plot
            if numeric_df.shape[1] >= 2:
                try:
                    col_x = numeric_df.columns[0]
                    col_y = numeric_df.columns[1]
                    
                    # Sample max 300 points for performance
                    sample_df = df.sample(n=min(300, len(df)), random_state=42)
                    
                    scatter_points = []
                    for _, row in sample_df.iterrows():
                        scatter_points.append({
                            "x": float(row[col_x]) if pd.notnull(row[col_x]) else 0,
                            "y": float(row[col_y]) if pd.notnull(row[col_y]) else 0,
                            "class": str(row[target_col])
                        })
                        
                    scatter_data = scatter_points
                except Exception as e:
                    print(f"Scatter calc error: {e}")

        print(f"[{os.getpid()}] Calculations done. Returning response.")
        return DatasetResponse(
            filename=file.filename,
            rows=df.shape[0],
            columns=df.shape[1],
            column_names=df.columns.tolist(),
            preview=preview_data,
            distributions=distributions,
            correlations=correlations,
            scatter_data=scatter_data
        )
        
    except Exception as e:
        print(f"CRITICAL UPLOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

@router.post("/api/pipeline/run", response_model=PipelineResult)
def run_pipeline_endpoint(pipeline: PipelineRequest):
    return process_pipeline(pipeline)

@router.get("/api/download/{filename}")
def download_model(filename: str):
    file_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=filename)
    raise HTTPException(status_code=404, detail="Model file not found")
