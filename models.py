# app/models.py

from pydantic import BaseModel
from typing import List, Dict

class CriterionResult(BaseModel):
    reasoning: str
    sentence_ids: List[int]
    eligibility_label: str

class InclusionExclusionResults(BaseModel):
    inclusion: Dict[str, CriterionResult]
    exclusion: Dict[str, CriterionResult]

class AggregationResult(BaseModel):
    relevance_explanation: str
    relevance_score_R: float
    eligibility_explanation: str
    eligibility_score_E: float

class RankScore(BaseModel):
    trial_id: str
    score: float

class FinalScore(BaseModel):
    trial_id: str
    score: float

class RetrieveRequest(BaseModel):
    patient_id: str
    text: str
    N: int

class RetrieveResponse(BaseModel):
    patient_id: str
    nctids: List[str]
    rank_scores: List[RankScore]
    final_scores: List[FinalScore]
    matching_results: Dict[str, InclusionExclusionResults]
    aggregation_results: Dict[str, AggregationResult]
