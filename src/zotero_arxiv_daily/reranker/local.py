from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseReranker, register_reranker


@register_reranker("local")
class LocalReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self._encoder: Optional[SentenceTransformer] = None
        self._encoder_model_name: Optional[str] = None

    def _get_encoder(self) -> SentenceTransformer:
        model_name = self.config.reranker.local.model

        # 如果配置里 model 改了，重新加载
        if self._encoder is None or self._encoder_model_name != model_name:
            if not self.config.executor.debug:
                from transformers.utils import logging as transformers_logging
                from huggingface_hub.utils import logging as hf_logging

                transformers_logging.set_verbosity_error()
                hf_logging.set_verbosity_error()
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
                logging.getLogger("transformers").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", category=FutureWarning)

            self._encoder = SentenceTransformer(model_name, trust_remote_code=True)
            self._encoder_model_name = model_name

        return self._encoder

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        encoder = self._get_encoder()

        encode_kwargs: dict[str, Any] = self.config.reranker.local.encode_kwargs or {}

        s1_feature = encoder.encode(s1, **encode_kwargs)
        s2_feature = encoder.encode(s2, **encode_kwargs)

        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()