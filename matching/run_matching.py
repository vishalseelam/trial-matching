# clinical_trial_api/app/matching/run_matching.py

__author__ = "vishal"

"""
Matching Runner for Clinical Trials.
"""

import json
from nltk.tokenize import sent_tokenize
import logging
import os
from typing import Dict, Any, List

from .matching import MatchingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchingRunner:
    def __init__(
        self,
        matching_processor: MatchingProcessor,
        qid2nctids: Dict[str, List[str]],
        qid_to_patient: Dict[str, str],
        nctid_to_trial: Dict[str, Dict[str, Any]],
    ):
        """
        Initialize the MatchingRunner with necessary data mappings.

        :param matching_processor: Instance of MatchingProcessor.
        :param qid2nctids: Mapping from patient IDs to list of NCTIDs.
        :param qid_to_patient: Mapping from patient IDs to patient notes.
        :param nctid_to_trial: Mapping from NCTIDs to trial information.
        """
        self.matching_processor = matching_processor
        self.qid2nctids = qid2nctids
        self.qid_to_patient = qid_to_patient
        self.nctid_to_trial = nctid_to_trial

    def format_patient_note(self, patient_text: str) -> str:
        """
        Formats the patient note by tokenizing sentences and adding sentence IDs.

        :param patient_text: Raw patient note.
        :return: Formatted patient note with sentence IDs.
        """
        sents = sent_tokenize(patient_text)
        sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
        formatted_sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
        formatted_patient = "\n".join(formatted_sents)
        return formatted_patient

    def run_matching(self) -> Dict[str, Any]:
        """
        Executes the matching process for all patients and their NCTIDs.

        :return: Dictionary containing matching results for all patients.
        """
        output = {}

        for qid, nctids in self.qid2nctids.items():
            logger.info(f"Processing patient ID: {qid}")

            patient_text = self.qid_to_patient.get(qid, "")
            if not patient_text:
                logger.warning(f"No patient note found for patient ID {qid}. Skipping.")
                continue

            formatted_patient = self.format_patient_note(patient_text)

            if qid not in output:
                output[qid] = {}

            for nctid in nctids:
                trial_info = self.nctid_to_trial.get(nctid)
                if not trial_info:
                    logger.error(f"Trial ID {nctid} not found in corpus. Skipping.")
                    continue

                if nctid in output[qid]:
                    logger.info(f"Already processed Trial ID {nctid} for patient ID {qid}. Skipping.")
                    continue

                try:
                    results = self.matching_processor.trial_matching(trial_info, formatted_patient)
                    output[qid][nctid] = results
                    logger.info(f"Successfully matched Trial ID {nctid} for patient ID {qid}.")

                except Exception as e:
                    logger.error(f"Error processing Trial ID {nctid} for patient ID {qid}: {e}")
                    output[qid][nctid] = {"error": str(e)}

        logger.info("Matching process completed.")
        return output


def load_qid2nctids(qid2nctids_path: str) -> Dict[str, List[str]]:
    """
    Load the qid to NCTIDs mapping.

    :param qid2nctids_path: Path to the JSON file containing qid to NCTIDs mapping.
    :return: Dictionary mapping patient IDs to lists of NCTIDs.
    """
    if not os.path.exists(qid2nctids_path):
        logger.error(f"File {qid2nctids_path} does not exist.")
        return {}
    
    with open(qid2nctids_path, "r") as f:
        qid2nctids = json.load(f)
    logger.info(f"Loaded qid2nctids from {qid2nctids_path}.")
    return qid2nctids


def load_qid_to_patient_notes(queries_path: str) -> Dict[str, str]:
    """
    Load patient notes from queries.jsonl.

    :param queries_path: Path to the queries.jsonl file.
    :return: Dictionary mapping patient IDs to patient notes.
    """
    qid_to_patient = {}
    if not os.path.exists(queries_path):
        logger.error(f"File {queries_path} does not exist.")
        return qid_to_patient

    with open(queries_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["_id"]
            patient_text = entry["text"]
            qid_to_patient[qid] = patient_text
    logger.info(f"Loaded patient notes from {queries_path}.")
    return qid_to_patient


def load_corpus(corpus_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the corpus.jsonl into a dictionary mapping NCTID to trial info.

    :param corpus_path: Path to the corpus.jsonl file.
    :return: Dictionary mapping NCTIDs to trial information.
    """
    nctid_to_trial = {}
    if not os.path.exists(corpus_path):
        logger.error(f"File {corpus_path} does not exist.")
        return nctid_to_trial

    with open(corpus_path, "r") as f:
        for line in f:
            trial = json.loads(line)
            nctid = trial["_id"]
            nctid_to_trial[nctid] = trial
    logger.info(f"Loaded corpus data from {corpus_path}.")
    return nctid_to_trial


# Example usage (for testing purposes only)
if __name__ == "__main__":
    # Define paths
    qid2nctids_path = "results/qid2nctids_results_human_summary.json"
    queries_path = "dataset/queries.jsonl"
    corpus_path = "dataset/corpus.jsonl"
    output_path = "results/matching_results.json"

    # Load mappings
    qid2nctids = load_qid2nctids(qid2nctids_path)
    qid_to_patient = load_qid_to_patient_notes(queries_path)
    nctid_to_trial = load_corpus(corpus_path)

    # Initialize MatchingProcessor
    matching_processor = MatchingProcessor(model="gpt-4")

    # Initialize MatchingRunner
    matching_runner = MatchingRunner(
        matching_processor=matching_processor,
        qid2nctids=qid2nctids,
        qid_to_patient=qid_to_patient,
        nctid_to_trial=nctid_to_trial
    )

    # Run matching
    matching_results = matching_runner.run_matching()

    # Save the results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(matching_results, f, indent=4)

    logger.info(f"Matching results saved to {output_path}")
