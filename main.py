# app/main.py

from fastapi import FastAPI, HTTPException, Depends
from typing import List, Dict, Any
from app.tasks import ClinicalTrialRetrievalPipeline
from app.dependency import get_pipeline
from app.models import (
    RetrieveRequest,
    RetrieveResponse,
    RankScore,
    FinalScore,
    InclusionExclusionResults,
    AggregationResult,
)

app = FastAPI(
    title="Clinical Trial Retrieval API",
    description="API for retrieving and ranking relevant clinical trials based on patient descriptions.",
    version="1.0.0",
    debug=True
)

@app.post("/retrieve", response_model=RetrieveResponse, summary="Retrieve and rank relevant clinical trials.")
def retrieve_trials(
    request: RetrieveRequest,
    pipeline: ClinicalTrialRetrievalPipeline = Depends(get_pipeline)
):
    """
    Retrieve and rank relevant clinical trials based on the patient's description.

    - **patient_id**: Unique identifier for the patient.
    - **text**: Doctor's note describing the patient's condition.
    - **N**: Number of top trials to retrieve.
    """
    try:
        # Execute the pipeline
        results = pipeline.execute_pipeline(
            patient_id=request.patient_id,
            patient_text=request.text,
            N=request.N
        )

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        response = RetrieveResponse(
            patient_id=request.patient_id,
            nctids=results.get("nctids", []),
            rank_scores=[
                RankScore(trial_id=rank['trial_id'], score=rank['score'])
                for rank in results.get("rank_scores", [])
            ],
            final_scores=[
                FinalScore(trial_id=score['trial_id'], score=score['score'])
                for score in results.get("final_scores", [])
            ],
            matching_results={
                trial_id: InclusionExclusionResults(**matching)
                for trial_id, matching in results.get("matching_results", {}).items()
            },
            aggregation_results={
                trial_id: AggregationResult(**aggregation)
                for trial_id, aggregation in results.get("aggregation_results", {}).items()
            }
        )

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
