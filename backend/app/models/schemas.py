from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PipelineStep(BaseModel):
    id: str
    type: str
    params: Dict[str, Any] = {}

class PipelineRequest(BaseModel):
    datasetName: str
    steps: List[PipelineStep]

class PipelineResult(BaseModel):
    success: bool
    message: str
    metrics: Dict[str, Any]
    logs: List[str]
    model_path: Optional[str] = None

class DatasetResponse(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    preview: List[Dict[str, Any]]
    distributions: Optional[Dict[str, Any]] = None 
    correlations: Optional[Dict[str, Any]] = None # New: {x_labels: [], y_labels: [], data: [[val, val], ...]}
    scatter_data: Optional[List[Dict[str, Any]]] = None # New: [{x: 1.2, y: 3.4, class: "Iris-setosa"}, ...]
