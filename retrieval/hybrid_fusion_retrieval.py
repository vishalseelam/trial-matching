# clinical_trial_api/app/retrieval/hybrid_fusion_retrieval.py

__author__ = "vishal"

"""
Conduct the first stage retrieval by the hybrid retriever.
"""

import faiss
import json
import os
import logging
from typing import List, Tuple, Dict, Any
import nltk
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        dataset_path: str = "dataset",
        bm25_cache_path: str = "retrieval/bm25_corpus.json",
        medcpt_embeds_path: str = "retrieval/medcpt_embeds.npy",
        medcpt_nctids_path: str = "retrieval/medcpt_nctids.json",
        batch_size: int = 10,
    ):
        """
        Initialize the HybridRetriever by loading or building BM25 and MedCPT indices.

        :param dataset_path: Path to the dataset containing corpus.jsonl.
        :param bm25_cache_path: Path to cache BM25 corpus.
        :param medcpt_embeds_path: Path to cache MedCPT embeddings.
        :param medcpt_nctids_path: Path to cache MedCPT NCTIDs.
        :param batch_size: Batch size for embedding generation.
        """
        self.dataset_path = dataset_path
        self.bm25_cache_path = bm25_cache_path
        self.medcpt_embeds_path = medcpt_embeds_path
        self.medcpt_nctids_path = medcpt_nctids_path
        self.batch_size = batch_size
        self.device = self.get_device()

        # Initialize BM25
        self.bm25, self.bm25_nctids = self.get_bm25_corpus_index()

        # Initialize MedCPT
        self.medcpt_index, self.medcpt_nctids = self.get_medcpt_corpus_index()

        # Load MedCPT Query Encoder
        self.model, self.tokenizer = self.load_medcpt_query_encoder()

    def get_device(self) -> torch.device:
        """Determine the device for PyTorch operations."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        return device

    def get_bm25_corpus_index(self) -> Tuple[BM25Okapi, List[str]]:
        """Load or build BM25 corpus index."""
        corpus_path = self.bm25_cache_path

        if os.path.exists(corpus_path):
            logger.info("Loading cached BM25 corpus...")
            with open(corpus_path, "r") as f:
                corpus_data = json.load(f)
                tokenized_corpus = corpus_data["tokenized_corpus"]
                corpus_nctids = corpus_data["corpus_nctids"]
        else:
            logger.info("Building BM25 corpus...")
            tokenized_corpus = []
            corpus_nctids = []

            corpus_file = os.path.join(self.dataset_path, "corpus.jsonl")
            with open(corpus_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    corpus_nctids.append(entry["_id"])

                    # Weighting: 3 * title, 2 * condition, 1 * text
                    tokens = word_tokenize(entry.get("title", "").lower()) * 3
                    diseases = entry.get("metadata", {}).get("diseases_list", [])
                    for disease in diseases:
                        tokens += word_tokenize(disease.lower()) * 2
                    tokens += word_tokenize(entry.get("text", "").lower())

                    tokenized_corpus.append(tokens)

            corpus_data = {
                "tokenized_corpus": tokenized_corpus,
                "corpus_nctids": corpus_nctids,
            }

            os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
            with open(corpus_path, "w") as f:
                json.dump(corpus_data, f, indent=4)
            logger.info("BM25 corpus built and cached.")

        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, corpus_nctids

    def get_medcpt_corpus_index(self) -> Tuple[faiss.IndexFlatIP, List[str]]:
        """
        Load or build MedCPT corpus index.

        :return: FAISS index and list of NCTIDs.
        """
        corpus_path = self.medcpt_embeds_path
        nctids_path = self.medcpt_nctids_path
        batch_size = self.batch_size

        if os.path.exists(corpus_path) and os.path.exists(nctids_path):
            logger.info("Loading cached MedCPT embeddings and IDs...")
            embeds = np.load(corpus_path)
            with open(nctids_path, "r") as f:
                corpus_nctids = json.load(f)
        else:
            logger.info("Building MedCPT corpus...")
            embeds = []
            corpus_nctids = []

            model = AutoModel.from_pretrained(
                "ncbi/MedCPT-Article-Encoder",
                force_download=True
            ).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

            corpus_file = os.path.join(self.dataset_path, "corpus.jsonl")
            with open(corpus_file, "r") as f:
                entries = f.readlines()
                logger.info("Encoding the corpus in batches...")
                for i in range(0, len(entries), batch_size):
                    batch = entries[i:i + batch_size]
                    logger.info(f"Processing batch {i // batch_size + 1}/{len(entries) // batch_size + 1}")

                    for line in batch:
                        entry = json.loads(line)
                        corpus_nctids.append(entry["_id"])

                        title = entry.get("title", "")
                        text = entry.get("text", "")

                        with torch.no_grad():
                            try:
                                encoded = tokenizer(
                                    [[title, text]],
                                    truncation=True,
                                    padding=True,
                                    return_tensors="pt",
                                    max_length=512,
                                ).to(self.device)

                                embed = model(**encoded).last_hidden_state[:, 0, :]
                                embeds.append(embed[0].cpu().numpy())
                            except Exception as e:
                                logger.error(f"Error during encoding: {e}")
                                continue

            embeds = np.array(embeds)
            os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
            np.save(corpus_path, embeds)
            with open(nctids_path, "w") as f:
                json.dump(corpus_nctids, f, indent=4)
            logger.info("MedCPT corpus built and cached.")

        index = faiss.IndexFlatIP(768)  # Inner product index
        index.add(embeds)
        logger.info("FAISS index built and populated with embeddings.")

        return index, corpus_nctids

    def load_medcpt_query_encoder(self) -> Tuple[AutoTokenizer, AutoModel]:
        """
        Load MedCPT query encoder model and tokenizer.

        :return: Tuple of tokenizer and model.
        """
        logger.info("Loading MedCPT Query Encoder...")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.device)
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        return tokenizer, model

    def encode_queries(self, conditions: List[str]) -> np.ndarray:
        """
        Encode a list of conditions using the MedCPT Query Encoder.

        :param conditions: List of condition strings.
        :return: Numpy array of encoded embeddings.
        """
        tokenizer, model = self.tokenizer, self.model
        if not conditions:
            logger.warning("No conditions provided for encoding.")
            return np.array([])

        try:
            encoded = tokenizer(
                conditions,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=256,
            ).to(self.device)

            with torch.no_grad():
                embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

            logger.info("Conditions encoded successfully.")
            return embeds
        except Exception as e:
            logger.error(f"Error during query encoding: {e}")
            return np.array([])

    def retrieve_nctids(
        self,
        keywords: List[str],
        N: int = 2000,
        k: int = 100,
        bm25_wt: float = 1.0,
        medcpt_wt: float = 1.0
    ) -> List[str]:
        """
        Retrieve top N NCTIDs based on a list of keywords using a hybrid approach.

        :param keywords: List of keywords extracted from the query.
        :param N: Number of top trials to retrieve.
        :param k: Inverse rank scaling factor.
        :param bm25_wt: Weight for BM25 scores.
        :param medcpt_wt: Weight for MedCPT scores.
        :return: List of top NCTIDs.
        """
        if not keywords:
            logger.warning("No keywords provided for retrieval.")
            return []

        nctid2score: Dict[str, float] = {}

        # BM25 Retrieval
        bm25_top_nctids: List[str] = []
        for condition in keywords:
            tokens = word_tokenize(condition.lower())
            top_nctids = self.bm25.get_top_n(tokens, self.bm25_nctids, n=k)
            bm25_top_nctids.extend(top_nctids)

        # Remove duplicates while preserving order
        bm25_top_nctids = list(dict.fromkeys(bm25_top_nctids))
        logger.info(f"BM25 retrieved {len(bm25_top_nctids)} unique NCTIDs.")

        # MedCPT Retrieval
        medcpt_top_nctids: List[str] = []
        embeds = self.encode_queries(keywords)
        if embeds.size > 0:
            try:
                scores, inds = self.medcpt_index.search(embeds, k=k)
                for ind_list in inds:
                    medcpt_top_nctids.extend([self.medcpt_nctids[idx] for idx in ind_list])
                logger.info(f"MedCPT retrieved {len(medcpt_top_nctids)} NCTIDs.")
            except Exception as e:
                logger.error(f"Error during MedCPT search: {e}")

        # Remove duplicates while preserving order
        medcpt_top_nctids = list(dict.fromkeys(medcpt_top_nctids))

        # Combine BM25 and MedCPT scores
        for rank, nctid in enumerate(bm25_top_nctids):
            if nctid not in nctid2score:
                nctid2score[nctid] = 0.0
            nctid2score[nctid] += (bm25_wt * (1 / (rank + k)))

        for rank, nctid in enumerate(medcpt_top_nctids):
            if nctid not in nctid2score:
                nctid2score[nctid] = 0.0
            nctid2score[nctid] += (medcpt_wt * (1 / (rank + k)))

        # Sort and select top N
        sorted_nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
        top_nctids = [nctid for nctid, _ in sorted_nctid2score[:N]]
        logger.info(f"Total of {len(top_nctids)} NCTIDs retrieved after combining BM25 and MedCPT scores.")

        return top_nctids


# Example usage (for testing purposes only)
if __name__ == "__main__":
    # Initialize the HybridRetriever
    retriever = HybridRetriever()

    # Sample keywords
    sample_keywords = ["chest pain", "hypertension", "obesity"]

    # Retrieve top NCTIDs
    top_nctids = retriever.retrieve_nctids(sample_keywords, N=10)
    print("Top NCTIDs:", top_nctids)
