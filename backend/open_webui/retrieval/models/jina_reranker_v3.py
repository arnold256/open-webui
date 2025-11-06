import logging
from typing import List, Tuple, Optional

from open_webui.env import SRC_LOG_LEVELS
from open_webui.retrieval.models.base_reranker import BaseReranker

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class JinaRerankerV3(BaseReranker):
    def __init__(self, model_path: str) -> None:
        """
        Initialize Jina Reranker V3 model.

        This model uses a listwise reranking approach with "last but not late"
        interaction, where the model processes all documents in a single context
        window with causal attention.

        Note: Requires trust_remote_code=True as the model includes custom code
        for prompt template construction and embedding extraction.

        Args:
            model_path: Path to the model (local or HuggingFace model ID)
        """
        log.info(f"JinaRerankerV3: Loading model from {model_path}")

        try:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                model_path,
                dtype="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            log.info("JinaRerankerV3: Model loaded successfully")
        except Exception as e:
            log.error(f"JinaRerankerV3: Failed to load model: {e}")
            raise

    def predict(self, sentences: List[Tuple[str, str]]) -> Optional[List[float]]:
        """
        Rerank documents given a query.

        This method converts the standard CrossEncoder input format (list of tuples)
        to the format expected by jina-reranker-v3's .rerank() method, then converts
        the results back to a simple list of scores.

        The model internally constructs a complex prompt template with system/user/assistant
        roles and special tokens (<|doc_emb|>, <|query_emb|>) as described in the paper.

        Args:
            sentences: List of (query, document) tuples

        Returns:
            List of relevance scores (floats) in the same order as input documents,
            or None if an error occurs
        """
        if not sentences:
            log.warning("JinaRerankerV3: Empty sentences list provided")
            return None

        try:
            # Extract query from first tuple (all tuples have the same query)
            query = sentences[0][0]

            # Extract all documents from the tuples
            documents = [doc for _, doc in sentences]

            log.debug(f"JinaRerankerV3: Reranking {len(documents)} documents")
            log.debug(f"JinaRerankerV3: Query: {query[:100]}...")

            # Call the model's rerank method
            # The model handles prompt construction internally
            results = self.model.rerank(
                query=query,
                documents=documents,
                return_embeddings=False
            )

            # Extract scores and ensure they're in the original document order
            # The results dict contains a 'results' key with a list of dicts
            # Each dict has 'index' and 'relevance_score' keys
            num_docs = len(documents)
            scores = [0.0] * num_docs

            for result in results['results']:
                idx = result['index']
                score = result['relevance_score']
                if 0 <= idx < num_docs:
                    scores[idx] = score
                else:
                    log.warning(f"JinaRerankerV3: Invalid index {idx} in results")

            log.debug(f"JinaRerankerV3: Scores: {scores}")
            return scores

        except Exception as e:
            log.exception(f"JinaRerankerV3: Error during reranking: {e}")
            return None
