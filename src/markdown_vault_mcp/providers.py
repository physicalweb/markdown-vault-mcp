"""Embedding providers for markdown-vault-mcp.

Provides an :class:`EmbeddingProvider` ABC and three concrete implementations:

- :class:`OllamaProvider` — HTTP client to Ollama REST API.
- :class:`OpenAIProvider` — HTTP client to OpenAI Embeddings API.
- :class:`FastEmbedProvider` — local fastembed/ONNX runtime embeddings.

Use :func:`get_embedding_provider` to auto-detect and return the best
available provider based on a :class:`CollectionConfig` instance.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from markdown_vault_mcp.config import CollectionConfig

logger = logging.getLogger(__name__)

# Maximum texts per ONNX inference call inside FastEmbed.
# BAAI/bge-small-en-v1.5 (512-token context, the default model) has a
# manageable attention footprint even at batch_size=32.  If you switch to a
# long-context model such as nomic-embed-text-v1.5 (8192-token context) you
# may need to reduce this value significantly — see issue #306.
_FASTEMBED_ONNX_BATCH_SIZE = 32

# Default per-request token budget for bulk embedding.  OpenAI's
# ``/v1/embeddings`` refuses requests above 300 000 tokens ("Requested
# 547830 tokens, max 300000 tokens per request", AGO-576); the estimate in
# :func:`estimate_tokens` is coarse, so the default sits well under the cap.
DEFAULT_MAX_TOKENS_PER_REQUEST = 200_000

# A provider error that names a token cap looks like a 4xx with "token" in
# the body (OpenAI: "... 400: Requested N tokens, max M tokens per request").
_TOKEN_CAP_ERROR_RE = re.compile(r"\b4\d\d\b.*token", re.IGNORECASE | re.DOTALL)


def estimate_tokens(text: str) -> int:
    """Cheap upper-leaning token estimate for request budgeting.

    ASCII text tokenises at roughly four characters per token; anything
    outside ASCII (CJK, accented letters, symbols) is counted as one token
    per character, which over-estimates rather than under-estimates.

    Args:
        text: The string to estimate.

    Returns:
        Estimated token count, at least 1 for a non-empty string.
    """
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return max(1, (len(text) - non_ascii) // 4 + non_ascii)


def is_token_cap_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* is a provider refusal naming a token cap.

    Args:
        exc: The exception raised by :meth:`EmbeddingProvider.embed`.
    """
    return _TOKEN_CAP_ERROR_RE.search(str(exc)) is not None


class EmbeddingRequestTooLargeError(RuntimeError):
    """A single text exceeds the provider's per-request token cap on its own.

    Raised by :func:`embed_in_token_batches` once halving has reduced the
    failing request to one text.  Carries the position of that text so the
    caller can name the document it came from.

    Attributes:
        index: Position of the offending text in the ``texts`` list passed
            to :func:`embed_in_token_batches`.
        path: Vault path of the document the text came from, when the
            caller knows it (``None`` otherwise).
    """

    def __init__(
        self, index: int, cause: BaseException, *, path: str | None = None
    ) -> None:
        self.index = index
        self.path = path
        where = f"{path!r}" if path is not None else f"text #{index}"
        super().__init__(
            f"Embedding request for {where} exceeds the provider's per-request "
            f"token cap on its own: {cause}"
        )


def _split_by_token_budget(
    texts: list[str], budget: int
) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(start_index, sub_batch)`` pairs whose estimated tokens fit *budget*.

    A single text above the budget is yielded alone (the estimate may be
    high, and the provider gets to decide).
    """
    start = 0
    current: list[str] = []
    current_tokens = 0
    for i, text in enumerate(texts):
        tokens = estimate_tokens(text)
        if current and current_tokens + tokens > budget:
            yield start, current
            start = i
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens
    if current:
        yield start, current


def _embed_halving(
    provider: EmbeddingProvider, texts: list[str], offset: int
) -> list[list[float]]:
    """Embed *texts*, halving the batch whenever the provider names a token cap."""
    try:
        return provider.embed(texts)
    except Exception as exc:
        if not is_token_cap_error(exc):
            raise
        if len(texts) == 1:
            raise EmbeddingRequestTooLargeError(offset, exc) from exc
        mid = len(texts) // 2
        logger.warning(
            "embed: provider refused %d texts on a token cap — retrying as "
            "%d + %d (%s)",
            len(texts),
            mid,
            len(texts) - mid,
            exc,
        )
        return _embed_halving(provider, texts[:mid], offset) + _embed_halving(
            provider, texts[mid:], offset + mid
        )


def embed_in_token_batches(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    max_tokens_per_request: int | None = None,
) -> list[list[float]]:
    """Embed *texts* in requests that stay under a per-request token budget.

    Splits the input greedily by :func:`estimate_tokens` so no single
    request is expected to exceed the budget, and — because the estimate is
    coarse — halves and retries any request the provider still refuses on
    a token cap.  A text the provider refuses on its own raises
    :class:`EmbeddingRequestTooLargeError` naming its position; every other
    provider error propagates unchanged.

    Args:
        provider: The embedding provider to call.
        texts: Texts to embed, in order.
        max_tokens_per_request: Token budget per request.  Defaults to
            ``provider.max_tokens_per_request``.

    Returns:
        One vector per input text, in input order.

    Raises:
        EmbeddingRequestTooLargeError: If a single text exceeds the cap.
        RuntimeError: Propagated from the provider for any other failure.
    """
    if not texts:
        return []
    budget = (
        max_tokens_per_request
        if max_tokens_per_request is not None
        else provider.max_tokens_per_request
    )
    if budget <= 0:
        return provider.embed(texts)
    batches = list(_split_by_token_budget(texts, budget))
    if len(batches) > 1:
        logger.info(
            "embed: %d texts split into %d requests under a %d-token budget",
            len(texts),
            len(batches),
            budget,
        )
    out: list[list[float]] = []
    for start, batch in batches:
        out.extend(_embed_halving(provider, batch, start))
    return out


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Attributes:
        max_tokens_per_request: Estimated-token budget for a single
            :meth:`embed` request when called through
            :func:`embed_in_token_batches`.  ``0`` disables splitting.
    """

    max_tokens_per_request: int = DEFAULT_MAX_TOKENS_PER_REQUEST

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension size.

        Returns:
            Integer dimension of each embedding vector.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier for index compatibility metadata."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier for index compatibility metadata."""
        ...

    @property
    def recommended_batch_size(self) -> int:
        """Recommended chunk batch size for bulk embedding.

        Defaults to a conservative 4 (matches the FastEmbed/ONNX memory
        constraint that originally motivated the module-level constant).
        API-backed providers (OpenAI, Ollama HTTP) can override with a
        larger value to amortize per-request HTTP overhead.
        """
        return 4


class OllamaProvider(EmbeddingProvider):
    """Embedding provider backed by the Ollama REST API.

    Args:
        host: Base URL of the Ollama server.
        model: Model name to use for embeddings.
        cpu_only: When ``True``, request CPU-only inference (sets
            ``num_gpu=0`` in the Ollama options payload).
    """

    def __init__(self, host: str, model: str, *, cpu_only: bool = False) -> None:
        """Initialise OllamaProvider with explicit parameters.

        Args:
            host: Base URL of the Ollama server.
            model: Model name to use for embeddings.
            cpu_only: When ``True``, request CPU-only inference.

        Raises:
            ImportError: If ``httpx`` is not installed.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "OllamaProvider requires 'httpx'. "
                "Install it with: pip install 'markdown-vault-mcp[embeddings-api]'"
            ) from exc

        self._httpx = httpx
        self._host = host.rstrip("/")
        self._model = model
        self._cpu_only = cpu_only
        self._dimension: int | None = None

        logger.debug(
            "OllamaProvider initialised: host=%s model=%s cpu_only=%s",
            self._host,
            self._model,
            self._cpu_only,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the Ollama REST API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            RuntimeError: If the Ollama API returns an error response.
        """
        payload: dict[str, object] = {"model": self._model, "input": texts}
        if self._cpu_only:
            payload["options"] = {"num_gpu": 0}

        url = f"{self._host}/api/embed"
        logger.debug("POST %s model=%s texts=%d", url, self._model, len(texts))

        with self._httpx.Client() as client:
            response = client.post(url, json=payload, timeout=30.0)

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API error {response.status_code}: {response.text}"
            )

        data = response.json()
        embeddings: list[list[float]] = data["embeddings"]

        # Cache dimension from first successful call.
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings

    @property
    def dimension(self) -> int:
        """Embedding dimension size.

        Embeds a test string on first access to determine the dimension.

        Returns:
            Integer dimension of each embedding vector.
        """
        if self._dimension is None:
            self.embed(["dimension probe"])
        if self._dimension is None:
            raise RuntimeError(
                "OllamaProvider.embed() returned no embeddings; "
                "cannot determine dimension."
            )
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI-compatible Embeddings API.

    Args:
        api_key: OpenAI API key for authentication.
        base_url: Base URL for an OpenAI-compatible API.
        model: Embedding model name.
    """

    _MODEL = "text-embedding-3-small"
    _BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _BASE_URL,
        model: str = _MODEL,
    ) -> None:
        """Initialise OpenAIProvider with an explicit API key.

        Args:
            api_key: OpenAI API key for authentication.
            base_url: Base URL for an OpenAI-compatible API.
            model: Embedding model name.

        Raises:
            ImportError: If ``httpx`` is not installed.
            RuntimeError: If ``api_key`` is empty.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "OpenAIProvider requires 'httpx'. "
                "Install it with: pip install 'markdown-vault-mcp[embeddings-api]'"
            ) from exc

        self._httpx = httpx
        if not api_key:
            raise RuntimeError("OpenAIProvider requires a non-empty api_key.")
        self._api_key = api_key
        self._base_url = (base_url or self._BASE_URL).rstrip("/")
        self._endpoint = f"{self._base_url}/embeddings"
        self._model = model or self._MODEL
        self._dimension: int | None = None

        logger.debug(
            "OpenAIProvider initialised: base_url=%s model=%s",
            self._base_url,
            self._model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the OpenAI Embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors in input order.

        Raises:
            RuntimeError: If the OpenAI API returns an error response.
        """
        payload = {"input": texts, "model": self._model}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "POST %s model=%s texts=%d", self._endpoint, self._model, len(texts)
        )

        with self._httpx.Client() as client:
            response = client.post(
                self._endpoint, json=payload, headers=headers, timeout=30.0
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {response.text}"
            )

        data = response.json()
        # Sort by index to guarantee input order is preserved.
        items: list[dict] = sorted(data["data"], key=lambda d: d["index"])
        embeddings: list[list[float]] = [item["embedding"] for item in items]

        # Cache dimension from first successful call.
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings

    @property
    def dimension(self) -> int:
        """Embedding dimension size.

        Embeds a test string on first access to determine the dimension.

        Returns:
            Integer dimension of each embedding vector.
        """
        if self._dimension is None:
            self.embed(["dimension probe"])
        if self._dimension is None:
            raise RuntimeError(
                "OpenAIProvider.embed() returned no embeddings; "
                "cannot determine dimension."
            )
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def recommended_batch_size(self) -> int:
        """OpenAI accepts up to 2048 inputs per /v1/embeddings call. Using
        a large batch amortizes the per-request HTTP overhead — first-boot
        embedding goes from O(seconds × N/4) to O(seconds × N/256).
        Below 2048 to leave headroom for token limits per request."""
        return 256


class FastEmbedProvider(EmbeddingProvider):
    """Embedding provider backed by the local fastembed library.

    The ``fastembed`` package is imported lazily at instantiation
    time so that it does not need to be installed unless this provider is used.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ) -> None:
        """Initialise FastEmbed model.

        Args:
            model_name: FastEmbed model identifier.
            cache_dir: Optional model cache directory.

        Raises:
            ImportError: If ``fastembed`` is not installed.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "FastEmbedProvider requires 'fastembed'. "
                "Install it with: pip install 'markdown-vault-mcp[embeddings]'"
            ) from exc

        self._model_name = model_name
        self._cache_dir = cache_dir
        kwargs: dict[str, object] = {"model_name": self._model_name}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        self._model = TextEmbedding(**kwargs)
        self._dimension: int | None = None
        logger.debug(
            "FastEmbedProvider initialised: model=%s cache_dir=%s",
            self._model_name,
            self._cache_dir,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the local fastembed model.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        vectors = [
            vector.tolist()
            for vector in self._model.embed(
                texts, batch_size=_FASTEMBED_ONNX_BATCH_SIZE
            )
        ]
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors

    @property
    def dimension(self) -> int:
        """Embedding dimension size from the loaded model.

        Returns:
            Integer dimension of each embedding vector.
        """
        if self._dimension is None:
            self.embed(["dimension probe"])
        if self._dimension is None:
            raise RuntimeError(
                "FastEmbedProvider.embed() returned no embeddings; "
                "cannot determine dimension."
            )
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "fastembed"

    @property
    def model_name(self) -> str:
        return self._model_name


def get_embedding_provider(config: CollectionConfig) -> EmbeddingProvider:
    """Auto-detect and return an embedding provider from config.

    Checks ``config.embedding_provider`` for an explicit selection. When
    that field is ``None``, probes for available providers in this order:

    1. If ``config.openai_api_key`` is set → :class:`OpenAIProvider`.
    2. If Ollama is reachable at ``config.ollama_host`` →
       :class:`OllamaProvider`.
    3. If ``fastembed`` can be imported →
       :class:`FastEmbedProvider`.
    4. Raises :class:`RuntimeError` with installation instructions.

    Args:
        config: Collection configuration containing embedding settings.

    Returns:
        An initialised :class:`EmbeddingProvider` instance.

    Raises:
        RuntimeError: If no provider is available and
            ``config.embedding_provider`` is not set, or if the explicitly
            requested provider cannot be initialised.
        ValueError: If ``config.embedding_provider`` is set to an
            unrecognised value.
    """
    provider = _select_embedding_provider(config)
    budget = getattr(config, "embedding_max_tokens_per_request", None)
    if budget is not None:
        provider.max_tokens_per_request = int(budget)
    return provider


def _select_embedding_provider(config: CollectionConfig) -> EmbeddingProvider:
    """Resolve the provider named (or auto-detected) by *config*."""
    explicit = (config.embedding_provider or "").strip().lower()

    if explicit == "openai":
        logger.info("Using OpenAIProvider (embedding_provider=openai)")
        return OpenAIProvider(
            api_key=config.openai_api_key or "",
            base_url=config.openai_base_url,
            model=config.openai_embedding_model,
        )

    if explicit == "ollama":
        logger.info("Using OllamaProvider (embedding_provider=ollama)")
        return OllamaProvider(
            host=config.ollama_host,
            model=config.ollama_model,
            cpu_only=config.ollama_cpu_only,
        )

    if explicit == "fastembed":
        logger.info(
            "Using FastEmbedProvider (embedding_provider=%s)",
            explicit,
        )
        return FastEmbedProvider(
            model_name=config.fastembed_model,
            cache_dir=config.fastembed_cache_dir,
        )

    if explicit:
        raise ValueError(
            f"Unrecognised embedding_provider value: {explicit!r}. "
            "Valid values: 'openai', 'ollama', 'fastembed'."
        )

    # Auto-detect: OpenAI API key present?
    if config.openai_api_key:
        logger.info("Auto-detected OpenAIProvider (openai_api_key is set)")
        return OpenAIProvider(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.openai_embedding_model,
        )

    # Auto-detect: Ollama reachable?
    host = config.ollama_host.rstrip("/")
    try:
        import httpx

        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{host}/api/tags")
        if response.status_code == 200:
            logger.info("Auto-detected OllamaProvider (Ollama reachable at %s)", host)
            return OllamaProvider(
                host=config.ollama_host,
                model=config.ollama_model,
                cpu_only=config.ollama_cpu_only,
            )
    except Exception:
        logger.debug("Ollama not reachable at %s, skipping", host)

    # Auto-detect: fastembed importable?
    try:
        import fastembed  # noqa: F401

        logger.info("Auto-detected FastEmbedProvider")
        return FastEmbedProvider(
            model_name=config.fastembed_model,
            cache_dir=config.fastembed_cache_dir,
        )
    except ImportError:
        logger.debug("fastembed not available, skipping")

    raise RuntimeError(
        "No embedding provider is available. Install one of:\n"
        "  pip install 'markdown-vault-mcp[embeddings-api]'  # httpx for Ollama or OpenAI\n"
        "  pip install 'markdown-vault-mcp[embeddings]'       # fastembed (local)\n"
        "Or set OPENAI_API_KEY for the OpenAI provider, "
        "or start an Ollama server for the Ollama provider."
    )
