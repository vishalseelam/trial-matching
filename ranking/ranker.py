# clinical_trial_api/app/ranking/ranker.py

__author__ = "vishal"

"""
Ranker for Clinical Trials.
"""

import json
import logging
import os
from typing import Dict, Any, List, Tuple

import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type  # For implementing retries
from openai import RateLimitError, OpenAIError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key from environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Ranker:
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the Ranker with the specified OpenAI model.
        
        :param model: OpenAI model to use for ranking.
        """
        self.model = model
        logger.info(f"Ranker initialized with model: {self.model}")

    @staticmethod
    def get_matching_score(matching: dict) -> float:
        """
        Calculate the matching score for a trial based on inclusion and exclusion criteria.
        
        :param matching: Dictionary containing inclusion and exclusion matching results.
        :return: Calculated matching score.
        """
        eps = 1e-9  # Small epsilon to prevent division by zero

        included = 0
        not_inc = 0
        na_inc = 0
        no_info_inc = 0

        excluded = 0
        not_exc = 0
        na_exc = 0
        no_info_exc = 0

        # Count inclusion criteria
        for criteria, info in matching.get("inclusion", {}).items():
            if len(info) != 3:
                continue

            if info[2] == "included":
                included += 1
            elif info[2] == "not included":
                not_inc += 1
            elif info[2] == "not applicable":
                na_inc += 1
            elif info[2] == "not enough information":
                no_info_inc += 1

        # Count exclusion criteria
        for criteria, info in matching.get("exclusion", {}).items():
            if len(info) != 3:
                continue

            if info[2] == "excluded":
                excluded += 1
            elif info[2] == "not excluded":
                not_exc += 1
            elif info[2] == "not applicable":
                na_exc += 1
            elif info[2] == "not enough information":
                no_info_exc += 1

        # Calculate the matching score
        score = included / (included + not_inc + no_info_inc + eps)

        # Penalize for "not included" and "excluded"
        if not_inc > 0:
            score -= 1

        if excluded > 0:
            score -= 1

        return score

    @staticmethod
    def get_agg_score(assessment: Dict[str, Any]) -> float:
        """
        Calculate the aggregation score for a trial based on relevance and eligibility scores.
        
        :param assessment: Dictionary containing relevance and eligibility scores.
        :return: Calculated aggregation score.
        """
        try:
            rel_score = float(assessment.get("relevance_score_R", 0))
            eli_score = float(assessment.get("eligibility_score_E", 0))
        except (ValueError, TypeError):
            rel_score = 0.0
            eli_score = 0.0

        # Normalize scores
        score = (rel_score + eli_score) / 100
        return score

    def calculate_final_score(self, matching_results: Dict[str, Any], aggregation_results: Dict[str, Any]) -> Dict[str, float]:
        final_scores = {}
        logger.debug(f"Matching Results: {json.dumps(matching_results, indent=2)}")
        logger.debug(f"Aggregation Results: {json.dumps(aggregation_results, indent=2)}")
        
        for patient_id, trials in matching_results.items():
            agg_trials = aggregation_results.get(patient_id, {})
            for trial_id, matching in trials.items():
                agg = agg_trials.get(trial_id, {})
                if "error" in agg:
                    logger.warning(f"Skipping Trial ID {trial_id} for Patient ID {patient_id} due to aggregation error.")
                    continue

                try:
                    relevance_score = float(agg.get("relevance_score_R", 0))
                    eligibility_score = float(agg.get("eligibility_score_E", 0))
                    trial_score = relevance_score + eligibility_score
                    final_scores[trial_id] = trial_score
                except (TypeError, ValueError) as e:
                    logger.error(f"Error calculating score for Trial ID {trial_id}: {e}")
                    continue

        logger.info(f"Final Scores: {final_scores}")
        return final_scores






    def rank_trials(self, final_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        logger.debug(f"Final Scores Input to Ranking: {final_scores}")

        # Filter valid numeric scores
        print("Print idhar ayaaaaa....ansdoujandfkm iwjvbiae")
        print(final_scores)
        valid_scores = {k: v for k, v in final_scores.items() if isinstance(v, (int, float))}
        print(valid_scores)
        if not valid_scores:
            logger.error("No valid numeric scores found for ranking.")
            logger.debug(json.dumps(final_scores, indent=2))

            return []  # Return an empty list if no valid scores are found

        ranked_trials = sorted(valid_scores.items(), key=lambda x: -x[1])
        logger.debug(f"Ranked Trials: {ranked_trials}")
        return ranked_trials





    @staticmethod
    def print_rankings(
        ranked_trials: List[Tuple[str, float]],
        patient_id: str
    ):
        """
        Print the ranked trials for a patient.
        
        :param ranked_trials: Sorted list of tuples containing trial IDs and their scores.
        :param patient_id: ID of the patient.
        """
        print(f"\nPatient ID: {patient_id}")
        print("Clinical trial ranking:")
        for trial, score in ranked_trials:
            print(f"{trial}: {score}")
        print("===")


    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def call_openai_api(self, messages: list) -> str:
        """
        Call OpenAI API with retry logic and timeout.
        
        :param messages: List of messages for the ChatCompletion API.
        :return: Response content as string.
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def convert_criteria_pred_to_string(prediction: dict, trial_info: dict) -> str:
        output = ""
        for inc_exc in ["inclusion", "exclusion"]:
            idx2criterion = {}
            criteria = trial_info.get('metadata', {}).get(inc_exc + "_criteria", "").split("\n\n")
            idx = 0
            for criterion in criteria:
                criterion = criterion.strip()
                if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                    continue
                if len(criterion) < 5:
                    continue
                idx2criterion[str(idx)] = criterion
                idx += 1

            for criterion_idx, preds in prediction.get(inc_exc, {}).items():
                if criterion_idx not in idx2criterion:
                    logger.warning(f"Criterion index {criterion_idx} not found in criteria list.")
                    continue

                criterion = idx2criterion[criterion_idx]
                if not all(key in preds for key in ['reasoning', 'sentence_ids', 'eligibility_label']):
                    logger.warning(f"Prediction for criterion {criterion_idx} does not have all required keys.")
                    continue

                output += f"{inc_exc} criterion {criterion_idx}: {criterion}\n"
                output += f"\tPatient relevance: {preds['reasoning']}\n"
                if len(preds['sentence_ids']) > 0:
                    output += f"\tEvident sentences: {preds['sentence_ids']}\n"
                output += f"\tPatient eligibility: {preds['eligibility_label']}\n"
        return output

    def convert_pred_to_prompt(self, patient: str, pred: dict, trial_info: dict) -> Tuple[str, str]:
        title = trial_info.get('title', "No Title Provided")
        brief_summary = trial_info.get('metadata', {}).get('brief_summary') or trial_info.get('text', "No Summary Provided")
        diseases_list = trial_info.get('metadata', {}).get('diseases_list', [])
        drugs_list = trial_info.get('metadata', {}).get('drugs_list', [])

        trial = f"Title: {title}\n"
        trial += f"Target conditions: {', '.join(diseases_list)}\n"
        trial += f"Interventions: {', '.join(drugs_list)}\n"
        trial += f"Summary: {brief_summary}\n"

        pred_str = self.convert_criteria_pred_to_string(pred, trial_info)

        system_prompt = (
            "You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, "
            "and the patient eligibility predictions for each criterion.\n"
            "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clinical trial.\n"
            'Output a JSON dict formatted as Dict{"relevance_explanation": Str, "relevance_score_R": Float, "eligibility_explanation": Str, "eligibility_score_E": Float}.'
        )

        user_prompt = (
            f"Here is the patient note:\n{patient}\n\n"
            f"Here is the clinical trial description:\n{trial}\n\n"
            f"Here are the criterion-level eligibility predictions:\n{pred_str}\n\n"
            "Plain JSON output:"
        )

        return system_prompt, user_prompt

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )

    def aggregate_trial_scores(
        self,
        patient: str,
        trial_results: dict,
        trial_info: dict
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = self.convert_pred_to_prompt(patient, trial_results, trial_info)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response_content = self.call_openai_api(messages)
            response_content = response_content.strip("`").strip("json")
            results = json.loads(response_content)

            # Validate response structure
            required_keys = {"relevance_explanation", "relevance_score_R", "eligibility_explanation", "eligibility_score_E"}
            if not required_keys.issubset(results.keys()):
                logger.error("Missing required keys in the API response.")
                return {"error": "Missing required keys in the API response."}

            logger.info(f"Aggregation scores: {results}")
            return results

        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON from OpenAI API response.")
            return {"error": "Failed to decode JSON from OpenAI API response."}
        except OpenAIError as e:
            logger.exception(f"OpenAI API error during score aggregation: {e}")
            return {"error": f"OpenAI API error: {str(e)}"}
        except Exception as e:
            logger.exception(f"Unexpected error during score aggregation: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
