"""Embedding abstraction: `EmbeddingProvider` is the interface every backend
implements; `EmbeddingRouter` picks a concrete provider from
`kenzo.config.settings.EmbeddingConfig`, exactly mirroring how
`kenzo.models.router.ModelRouter` picks a chat provider from `RoutingConfig`/
`ProviderConfig` -- so the provider is never hardcoded anywhere that calls
`EmbeddingRouter.embed`/`embed_batch`.

Two concrete providers:

- `LiteLLMEmbeddingProvider` -- routes through `litellm.embedding()`, the
  same library Kenzo already uses for chat completions. This is what covers
  BOTH "local embeddings" (e.g. `provider="ollama"` pointing at a locally
  running Ollama server with an embedding model pulled) and "OpenAI-
  compatible embeddings" (`provider="openai"`, or any other OpenAI-compatible
  endpoint) through the exact same code path -- litellm's "<provider>/<model>"
  addressing convention is the abstraction, not a Kenzo-specific branch per
  provider.
- `HashingEmbeddingProvider` -- a dependency-free, fully offline, deterministic
  fallback (character-trigram feature hashing + L2 normalization -- the
  classic "hashing trick"). This is a REAL local embedding with no network
  call and no model download, but it is lexical/fuzzy-substring similarity,
  not learned semantic/synonym understanding -- e.g. it will NOT on its own
  know "sign out" and "logout" are related the way a trained embedding model
  would. It exists so semantic search has a genuine zero-setup default rather
  than requiring an API key before anything works at all; anyone who wants
  real synonym-level retrieval should configure a real provider via
  `embedding.provider`/`embedding.model`.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from kenzo.config.settings import EmbeddingConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(ABC):
    """Interface every embedding backend implements."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text into a fixed-length vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts at once (batched where the backend supports it)."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The length of every vector this provider produces."""


class HashingEmbeddingProvider(EmbeddingProvider):
    """Dependency-free, offline, deterministic local embedding via character-
    trigram feature hashing. See module docstring for its real, honest
    capability (lexical similarity) versus what it is NOT (learned semantic
    similarity).
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self._dimensions, dtype=np.float64)
        normalized = text.lower()

        for gram in self._char_trigrams(normalized):
            index, sign = self._hash_gram(gram)
            vector[index] += sign

        # Word-level tokens too (in addition to char trigrams) -- gives exact
        # short-identifier matches (e.g. "logout" as a whole word) equal
        # standing alongside the fuzzy substring signal from trigrams.
        for token in _TOKEN_RE.findall(normalized):
            index, sign = self._hash_gram(token)
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def _hash_gram(self, gram: str) -> tuple[int, float]:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big")
        index = value % self._dimensions
        sign = 1.0 if (value >> 1) % 2 == 0 else -1.0
        return index, sign

    @staticmethod
    def _char_trigrams(text: str) -> list[str]:
        padded = f"  {text} "
        if len(padded) < 3:
            return [padded]
        return [padded[i : i + 3] for i in range(len(padded) - 2)]


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Routes embedding calls through `litellm.embedding()` -- the same
    library Kenzo already depends on for chat completions
    (`kenzo.models.router.ModelRouter`). Covers any litellm-supported
    embedding backend (OpenAI, Azure OpenAI, a local Ollama server, HuggingFace
    text-embeddings-inference, etc.) through litellm's own "<provider>/<model>"
    addressing -- this class never special-cases a specific provider.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        api_base: Optional[str] = None,
        dimensions: int = 256,
    ) -> None:
        self._model = self._resolve_model_string(provider, model)
        self._api_key = api_key
        self._api_base = api_base
        self._dimensions = dimensions

    @staticmethod
    def _resolve_model_string(provider: str, model: str) -> str:
        """Mirrors `kenzo.models.providers.resolve_model_string`'s convention:
        a model string that already contains "/" is treated as an already-
        fully-qualified litellm id; otherwise it's prefixed with the provider
        name, e.g. provider="openai", model="text-embedding-3-small" ->
        "openai/text-embedding-3-small".
        """
        if "/" in model:
            return model
        return f"{provider}/{model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Deliberately lazy: litellm is a heavy import (pulls in httpx and
        # every provider SDK it supports), and callers that only ever use
        # the default "local" HashingEmbeddingProvider -- or import this
        # module for its types alone -- should never pay that cost.
        import litellm

        kwargs: dict[str, str] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = litellm.embedding(model=self._model, input=texts, **kwargs)
        return [item["embedding"] for item in response.data]


class EmbeddingRouter:
    """Picks a concrete `EmbeddingProvider` from an `EmbeddingConfig`. The
    ONLY place that decides which embedding backend is in play -- everything
    else (SemanticIndexer, CodeSearchEngine) depends on this class rather
    than instantiating a provider directly, so the provider is never
    hardcoded anywhere that needs an embedding.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._provider = self._build_provider(config)

    @staticmethod
    def _build_provider(config: EmbeddingConfig) -> EmbeddingProvider:
        if config.provider == "local":
            return HashingEmbeddingProvider(dimensions=config.dimensions)
        return LiteLLMEmbeddingProvider(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            dimensions=config.dimensions,
        )

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def embed(self, text: str) -> list[float]:
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed_batch(texts)
