# clinical_trial_api/app/retrieval/keyword_generation.py

__author__ = "vishal"

"""
Generate the search keywords for each patient.
"""

import json
import os
import openai
from openai import OpenAIError
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key from environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_keyword_generation_messages(note: str) -> list:
    """
    Prepare messages for OpenAI's ChatCompletion API.

    :param note: The patient's description or doctor's note.
    :return: List of messages formatted for the API.
    """
    system_content = (
        'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description possibly generated from a genomic profiling. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient include the main detected gene values. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
    )

    user_prompt = f"Here is the patient description: \n{note}\n\nJSON output:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt}
    ]

    return messages


def generate_keywords(patient_text: str) -> Dict[str, Any]:
    """
    Generate a summary and key conditions from the patient's doctor's note.

    :param patient_text: The doctor's note describing the patient's condition.
    :return: A dictionary with 'summary' and 'conditions' or an 'error' key.
    """
    if not patient_text or not isinstance(patient_text, str):
        logger.error("Invalid input: patient_text must be a non-empty string.")
        return {"error": "Invalid input: patient_text must be a non-empty string."}

    messages = get_keyword_generation_messages(patient_text)

    try:
        # Call OpenAI API for chat completion
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=500,
        )

        # Extract and clean the output
        message_content = response.choices[0].message.content.strip()

        # Attempt to parse JSON from the response
        output = json.loads(message_content)

        # Validate the output structure
        if "summary" in output and "conditions" in output:
            logger.info("Keyword generation successful.")
            return output
        else:
            logger.error("Invalid response structure from OpenAI API.")
            return {"error": "Invalid response structure from OpenAI API."}

    except json.JSONDecodeError:
        logger.exception("Failed to decode JSON from OpenAI API response.")
        return {"error": "Failed to decode JSON from OpenAI API response."}
    except OpenAIError as e:
        logger.exception(f"OpenAI API error: {e}")
        return {"error": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


# Example usage (for testing purposes only)
if __name__ == "__main__":
    sample_text = (
        "The patient is a 60-year-old male with a history of hypertension and type 2 diabetes. "
        "He presents with chest pain and shortness of breath. Recent tests indicate elevated cholesterol levels."
    )
    keywords_output = generate_keywords(sample_text)
    print(json.dumps(keywords_output, indent=4))
