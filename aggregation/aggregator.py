# clinical_trial_api/app/aggregation/aggregator.py

__author__ = "vishal"

"""
Aggregator for Clinical Trials.
"""

import json
import logging
from typing import Dict, Any, List

from app.ranking.ranker import Ranker  # Updated import path
from app.utils import load_json_file  # Assuming a utility function for loading JSON

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Aggregator:
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the Aggregator with the specified OpenAI model.
        
        :param model: OpenAI model to use for aggregation.
        """
        self.model = model
        logger.info(f"Aggregator initialized with model: {self.model}")

    def match_trials(self, patient_id: str, nctids: List[str]) -> Dict[str, Any]:
        """
        Match the patient with the retrieved trials based on inclusion and exclusion criteria.

        :param patient_id: Unique identifier for the patient.
        :param nctids: List of retrieved NCTIDs.
        :return: Dictionary of matching results.
        """
        matching_results = {}
        try:
            for nctid in nctids:
                # Implement matching logic, possibly using OpenAI's API
                # Placeholder matching result
                # Replace this with actual matching logic
                matching_results[nctid] = {
                    "inclusion": {"criteria_met": True},   # Example: patient meets inclusion criteria
                    "exclusion": {"criteria_met": False}   # Example: patient does not meet exclusion criteria
                }
            logger.info(f"Matching results for Patient ID {patient_id}: {matching_results}")
            return matching_results
        except Exception as e:
            logger.exception(f"Error during matching: {e}")
            return {}

    def aggregate(
        self,
        matching_results: Dict[str, Any],
        trial_info: Dict[str, Any],
        patient_queries: Dict[str, str],
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Aggregate matching results to compute relevance and eligibility scores.
        
        :param matching_results: Dictionary containing matching results for each patient and trial.
        :param trial_info: Dictionary mapping trial IDs to their detailed information.
        :param patient_queries: Dictionary mapping patient IDs to their notes.
        :param model: OpenAI model to use for aggregation.
        :return: Aggregated results dictionary.
        """
        aggregated_results = {}
        
        for patient_id, trials in matching_results.items():
            patient_note = patient_queries.get(patient_id, "")
            if not patient_note:
                logger.warning(f"No patient note found for patient ID {patient_id}. Skipping aggregation.")
                continue

            formatted_patient = self.format_patient_note(patient_note)
            aggregated_results[patient_id] = {}

            for trial_id, trial_results in trials.items():
                if not isinstance(trial_results, dict):
                    logger.error(f"Invalid trial results format for Trial ID {trial_id} and Patient ID {patient_id}.")
                    aggregated_results[patient_id][trial_id] = {"error": "Invalid trial results format."}
                    continue

                trial_details = trial_info.get(trial_id)
                if not trial_details:
                    logger.error(f"Trial ID {trial_id} not found in trial_info. Skipping aggregation.")
                    aggregated_results[patient_id][trial_id] = {"error": "Missing trial information."}
                    continue

                try:
                    # Perform aggregation using Ranker
                    logger.info(f"Aggregating Trial ID {trial_id} for Patient ID {patient_id}.")
                    ranker = Ranker(model=model)
                    aggregation_result = ranker.aggregate_trial_scores(
                        patient=formatted_patient,
                        trial_results=trial_results,
                        trial_info=trial_details
                    )
                    aggregated_results[patient_id][trial_id] = aggregation_result
                    logger.info(f"Successfully aggregated Trial ID {trial_id} for Patient ID {patient_id}.")
                except Exception as e:
                    logger.exception(f"Error during aggregation for Trial ID {trial_id} and Patient ID {patient_id}: {e}")
                    aggregated_results[patient_id][trial_id] = {"error": str(e)}
        
        logger.info("Aggregation process completed.")
        return aggregated_results

    @staticmethod
    def format_patient_note(patient_text: str) -> str:
        """
        Formats the patient note by tokenizing sentences and adding sentence IDs.
        
        :param patient_text: Raw patient note.
        :return: Formatted patient note with sentence IDs.
        """
        from nltk.tokenize import sent_tokenize  # Import here to avoid global dependency

        sents = sent_tokenize(patient_text)
        sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
        formatted_sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
        formatted_patient = "\n".join(formatted_sents)
        return formatted_patient
