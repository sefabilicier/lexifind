"""
BGE-M3 embedder wrapper.
Supports dense (1024-dim), sparse (lexical weights) embeddings.
Used for both ingestion (batch) and query-time (single).
"""

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)


class BGEEmbedder:
    """
    Singleton-friendly wrapper around BAAI/bge-m3.

    Provides:
        - Dense embeddings (1024-dim float vectors)
        - Sparse embeddings (lexical weights dict)
        - Batch encoding for ingestion
        - Single encoding for query
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        settings = get_settings()
        logger.info(
            "embedder.loading",
            model=settings.embedding_model,
            device=settings.embedding_device,
        )

        self.model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=True,  # 2x speedup on CPU/GPU
            device=settings.embedding_device,
        )
        self.batch_size = settings.embedding_batch_size
        self._initialized = True

        logger.info("BGE-M3 model loaded", event="embedder.ready")

    def embed_texts(
        self,
        texts: list[str],
        return_sparse: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[dict]]:
        """
        Encode a list of texts.

        Args:
            texts: List of strings to encode.
            return_sparse: If True, also returns sparse lexical weights.

        Returns:
            Dense embeddings as np.ndarray of shape (N, 1024).
            If return_sparse=True, returns (dense_embeddings, sparse_weights_list).
        """
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        dense = np.array(output["dense_vecs"], dtype=np.float32)

        if return_sparse:
            sparse = output["lexical_weights"]
            return dense, sparse

        return dense

    def embed_query(self, query: str) -> tuple[np.ndarray, dict]:
        """
        Encode a single query — always returns both dense and sparse.
        Used at retrieval time.
        """
        dense, sparse = self.embed_texts([query], return_sparse=True)
        return dense[0], sparse[0]