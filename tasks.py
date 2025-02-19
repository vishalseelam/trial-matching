# app/tasks.py

import logging
from typing import Dict, Any

from app.retrieval.hybrid_fusion_retrieval import HybridRetriever
from app.aggregation.aggregator import Aggregator
from app.ranking.ranker import Ranker
from app.matching.run_matching import MatchingRunner
from app.matching.matching import MatchingProcessor
from app.retrieval.keyword_generation import generate_keywords
from app.utils import load_trial_info  # Assuming a utility function to load trial info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalTrialRetrievalPipeline:
    _instance = None  # Class variable for singleton instance

    def __new__(cls, model: str = "gpt-4o-mini"):
        if cls._instance is None:
            cls._instance = super(ClinicalTrialRetrievalPipeline, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, model: str = "gpt-4o-mini"):
        if getattr(self, '__initialized', False):
            return
        self.hybrid_retriever = HybridRetriever()
        self.aggregator = Aggregator(model=model)
        self.ranker = Ranker(model=model)
        self.__initialized = True
        logger.info("ClinicalTrialRetrievalPipeline singleton instance created.")

    def execute_pipeline(self, patient_id: str, patient_text: str, N: int = 20) -> Dict[str, Any]:
        """
        Execute the retrieval pipeline for a single patient.

        :param patient_id: Unique identifier for the patient.
        :param patient_text: Doctor's note describing the patient's condition.
        :param N: Number of top trials to retrieve.
        :return: Dictionary containing ranked trial information.
        """
        try:
            logger.info(f"Starting pipeline for Patient ID: {patient_id}")

            # Step 1: Keyword Generation
            keywords_output = generate_keywords(patient_text)
            if "error" in keywords_output:
                logger.error(f"Keyword generation failed for Patient ID {patient_id}: {keywords_output['error']}")
                return {"error": keywords_output["error"]}

            keywords = keywords_output.get("conditions", [])
            if not keywords:
                logger.warning(f"No keywords generated for Patient ID {patient_id}.")
                return {"error": "No keywords generated from patient notes."}

            logger.info(f"Keywords for Patient ID {patient_id}: {keywords}")

            # Step 2: Hybrid Retrieval
            top_nctids = self.hybrid_retriever.retrieve_nctids(keywords, N=N)
            if not top_nctids:
                logger.warning(f"No NCTIDs retrieved for Patient ID {patient_id}.")
                return {"error": "No clinical trials retrieved for the given keywords."}

            logger.info(f"Retrieved NCTIDs for Patient ID {patient_id}: {top_nctids}")

            # Load trial data and patient notes
            qid2nctids = {patient_id: top_nctids}
            qid_to_patient = {patient_id: patient_text}
            nctid_to_trial = load_trial_info(top_nctids)

            # Step 3: Matching
            matching_processor = MatchingProcessor(model="gpt-4o-mini")
            matching_runner = MatchingRunner(
                matching_processor=matching_processor,
                qid2nctids=qid2nctids,
                qid_to_patient=qid_to_patient,
                nctid_to_trial=nctid_to_trial,
            )

            # Ensure matching_runner is defined before using it
            matching_results_raw = matching_runner.run_matching()
            matching_results = matching_results_raw.get(patient_id, {})
            if not matching_results:
                logger.warning(f"No matching results for Patient ID {patient_id}.")
                return {"error": "No matching results found for the retrieved clinical trials."}

            logger.info(f"Matching results for Patient ID {patient_id}: {matching_results}")

            # Step 4: Aggregation
            trial_info = nctid_to_trial  # Already loaded earlier
            aggregation_results_raw = self.aggregator.aggregate(
                matching_results={patient_id: matching_results},
                trial_info=trial_info,
                patient_queries={patient_id: patient_text},
                model=self.aggregator.model,
            )
            aggregation_results = aggregation_results_raw.get(patient_id, {})
            if not aggregation_results:
                logger.warning(f"Aggregation failed for Patient ID {patient_id}.")
                return {"error": "Aggregation of matching results failed."}

            logger.info(f"Aggregation results for Patient ID {patient_id}: {aggregation_results}")

            # Step 5: Calculate Final Scores
            final_scores_raw = self.ranker.calculate_final_score(
                matching_results={patient_id: matching_results},
                aggregation_results={patient_id: aggregation_results}
            )
            final_scores = {
                trial_id: score
                for trial_id, score in final_scores_raw.items()
            }
            if not final_scores:
                logger.warning(f"Final score calculation failed for Patient ID {patient_id}.")
                return {"error": "Final score calculation failed."}

            logger.info(f"Final scores for Patient ID {patient_id}: {final_scores}")

            # Step 6: Ranking
            rank_scores = self.ranker.rank_trials(final_scores)
            if not rank_scores:
                logger.warning(f"Ranking failed for Patient ID {patient_id}.")
                return {"error": "Ranking of trials failed."}

            logger.info(f"Rank scores for Patient ID {patient_id}: {rank_scores}")
            # Format rank_scores for API response
            rank_scores_formatted = [{"trial_id": trial_id, "score": score} for trial_id, score in rank_scores]
            logger.info(f"Formatted Rank scores for Patient ID {patient_id}: {rank_scores_formatted}")

            # Prepare final_scores for response
            final_scores_formatted = [{"trial_id": trial_id, "score": score} for trial_id, score in final_scores.items()]

            # Compile final output
            return {
                "patient_id": patient_id,
                "nctids": top_nctids,
                "rank_scores": rank_scores_formatted,  # Properly formatted for the response
                "final_scores": final_scores_formatted,
                "matching_results": matching_results,
                "aggregation_results": aggregation_results,
            }

        except Exception as e:
            logger.exception(f"An unexpected error occurred during pipeline execution for Patient ID {patient_id}: {e}")
            return {"error": "An internal error occurred. Please try again later."}
