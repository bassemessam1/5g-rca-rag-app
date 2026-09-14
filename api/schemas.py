from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    question: str

class TelemetryRequest(BaseModel):
    engineering_parameters: str
    drive_test_data: str

class DiagnosisResponse(BaseModel):
    root_cause_code: str
    description: str
    confidence: float
    timestamp: str
    citations: List[Dict[str, Any]]
    input_summary: Dict[str, Any]
