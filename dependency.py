# app/dependencies.py

from typing import Generator
from app.tasks import ClinicalTrialRetrievalPipeline

def get_pipeline() -> Generator[ClinicalTrialRetrievalPipeline, None, None]:
    """
    Dependency to provide a ClinicalTrialRetrievalPipeline instance.
    Uses singleton pattern within the dependency scope.
    """
    pipeline = ClinicalTrialRetrievalPipeline(model="gpt-4o-mini")
    try:
        yield pipeline
    finally:
        # Perform any necessary cleanup if needed
        pass
