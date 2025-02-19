# app/matching/matching.py

__author__ = "vishal"

"""
Matching Processor for Clinical Trials.
"""

import json
from nltk.tokenize import sent_tokenize
import logging
import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

import openai
from openai import OpenAIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type  # For implementing retries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key from environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_criteria(criteria: str) -> str:
    """
    Parses the criteria string to format it for prompting.

    :param criteria: Raw criteria string.
    :return: Formatted criteria string.
    """
    output = ""
    criteria_blocks = criteria.split("\n\n")

    idx = 0
    for criterion in criteria_blocks:
        criterion = criterion.strip()

        if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
            continue

        if len(criterion) < 5:
            continue

        output += f"{idx}. {criterion}\n"
        idx += 1

    return output


def format_trial_info(trial_info: dict, inc_exc: str) -> str:
    """
    Formats trial information based on inclusion or exclusion criteria.

    :param trial_info: Dictionary containing trial information.
    :param inc_exc: 'inclusion' or 'exclusion'.
    :return: Formatted trial information string.
    """
    brief_title = trial_info.get('metadata', {}).get('brief_title') or trial_info.get('title', "No Title Provided")
    brief_summary = trial_info.get('metadata', {}).get('brief_summary') or trial_info.get('text', "No Summary Provided")
    diseases_list = trial_info.get('metadata', {}).get('diseases_list', [])
    drugs_list = trial_info.get('metadata', {}).get('drugs_list', [])
    criteria = trial_info.get('metadata', {}).get(f'{inc_exc}_criteria', "")

    trial = f"Title: {brief_title}\n"
    trial += f"Target diseases: {', '.join(diseases_list)}\n"
    trial += f"Interventions: {', '.join(drugs_list)}\n"
    trial += f"Summary: {brief_summary}\n"

    if inc_exc in ["inclusion", "exclusion"]:
        formatted_criteria = parse_criteria(criteria)
        trial += f"{inc_exc.capitalize()} criteria:\n{formatted_criteria}\n"

    return trial


def get_matching_prompt(trial_info: dict, inc_exc: str, patient: str) -> Tuple[str, str]:
    """
    Constructs the system and user prompts for OpenAI's ChatCompletion.

    :param trial_info: Dictionary containing trial information.
    :param inc_exc: 'inclusion' or 'exclusion'.
    :param patient: Formatted patient note.
    :return: Tuple containing system prompt and user prompt.
    """
    system_prompt = (
        f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note "
        f"and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level."
    )

    if inc_exc == "inclusion":
        system_prompt += (
            " The factors that allow someone to participate in a clinical study are called inclusion criteria. "
            "They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions."
        )
    elif inc_exc == "exclusion":
        system_prompt += (
            " The factors that disqualify someone from participating are called exclusion criteria. "
            "They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions."
        )

    system_prompt += (
        f" You should check the {inc_exc} criteria one-by-one, and output the following three elements for each criterion:\n"
        f"Element 1. For each {inc_exc} criterion, briefly generate your reasoning process: First, judge whether the criterion is not applicable (not very common), "
        "where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. "
        "If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? "
        "If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
        "Element 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.\n"
        "Element 3. Classify the patient eligibility for this specific {inc_exc} criterion: "
    )

    if inc_exc == "inclusion":
        system_prompt += (
            'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. '
            '"not applicable" should only be used for criteria that are not applicable to the patient. '
            '"not enough information" should be used where the patient note does not contain sufficient information for making the classification. '
            'Try to use as few "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. '
            '"included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.\n'
        )
    elif inc_exc == "exclusion":
        system_prompt += (
            'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. '
            '"not applicable" should only be used for criteria that are not applicable to the patient. '
            '"not enough information" should be used where the patient note does not contain sufficient information for making the classification. '
            'Try to use as few "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. '
            '"excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'
        )

    system_prompt += (
        " You should output only a JSON dict exactly formatted as: "
        'dict{str(criterion_number): list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label)]}.'
    )

    user_prompt = (
        "Here is the patient note, each sentence is led by a sentence_id:\n"
        f"{patient}\n\n"
        "Here is the clinical trial:\n"
        f"{format_trial_info(trial_info, inc_exc)}\n\n"
        "Plain JSON output:"
    )

    return system_prompt, user_prompt


class MatchingProcessor:
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the MatchingProcessor with the specified OpenAI model.

        :param model: OpenAI model to use for matching.
        """
        self.model = model
        logger.info(f"MatchingProcessor initialized with model: {self.model}")

    @retry(
        retry=retry_if_exception_type(OpenAIError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call OpenAI API with retry logic.

        :param messages: List of messages for the ChatCompletion API.
        :return: Response content as string.
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def trial_matching(self, trial_info: dict, patient: str) -> Dict[str, Any]:
        """
        Perform matching for both inclusion and exclusion criteria.

        :param trial_info: Dictionary containing trial information.
        :param patient: Formatted patient note.
        :return: Dictionary containing matching results for inclusion and exclusion criteria.
        """
        results = {}

        for inc_exc in ["inclusion", "exclusion"]:
            system_prompt, user_prompt = get_matching_prompt(trial_info, inc_exc, patient)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                response_content = self.call_openai_api(messages)
                response_content = response_content.strip("`").strip("json")

                try:
                    raw_results = json.loads(response_content)
                    # Format the results to match the CriterionResult model
                    formatted_results = {
                        criterion_num: {
                            "reasoning": criterion_data[0],
                            "sentence_ids": criterion_data[1],
                            "eligibility_label": criterion_data[2]
                        }
                        for criterion_num, criterion_data in raw_results.items()
                    }
                    results[inc_exc] = formatted_results
                    logger.info(f"{inc_exc.capitalize()} criteria matching successful.")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON format in response for {inc_exc} criteria.")
                    results[inc_exc] = {"error": "Invalid JSON format in response."}

            except OpenAIError as e:
                logger.error(f"OpenAI API error during {inc_exc} criteria matching: {e}")
                results[inc_exc] = {"error": f"OpenAI API error: {str(e)}"}

            except Exception as e:
                logger.error(f"Unexpected error during {inc_exc} criteria matching: {e}")
                results[inc_exc] = {"error": f"Unexpected error: {str(e)}"}

        return results
