"""Tests for embedding providers in markdown_vault_mcp.providers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markdown_vault_mcp.config import CollectionConfig
from markdown_vault_mcp.providers import (
    DEFAULT_MAX_TOKENS_PER_REQUEST,
    EmbeddingProvider,
    EmbeddingRequestTooLargeError,
    FastEmbedProvider,
    OllamaProvider,
    OpenAIProvider,
    embed_in_token_batches,
    estimate_tokens,
    get_embedding_provider,
    is_token_cap_error,
)


def _make_httpx_mock(
    status_code: int = 200,
    json_body: dict | None = None,
    text: str = "",
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_client, mock_response) with context-manager wiring."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text
    mock_response.json.return_value = json_body or {}

    mock_client = MagicMock()
    mock_client.__enter__ = lambda _: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response
    mock_client.get.return_value = mock_response

    return mock_client, mock_response


def _config(**overrides: object) -> CollectionConfig:
    """Build a minimal CollectionConfig with optional overrides."""
    defaults: dict[str, object] = {"source_dir": Path("/tmp/vault")}
    defaults.update(overrides)
    return CollectionConfig(**defaults)  # type: ignore[arg-type]


class TestOllamaProvider:
    def test_embed_posts_to_correct_url(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[0.1, 0.2, 0.3]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://localhost:11434", model="nomic-embed-text"
            )
            result = provider.embed(["hello"])

        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["model"] == "nomic-embed-text"
        assert result == [[0.1, 0.2, 0.3]]
        assert provider.provider_name == "ollama"
        assert provider.model_name == "nomic-embed-text"

    def test_embed_payload_includes_model_and_input(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[0.5, 0.6]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(host="http://localhost:11434", model="test-model")
            provider.embed(["alpha", "beta"])

        _, call_kwargs = mock_client.post.call_args
        payload = call_kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["input"] == ["alpha", "beta"]

    def test_embed_cpu_only_includes_num_gpu_zero(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[1.0, 2.0]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://localhost:11434",
                model="nomic-embed-text",
                cpu_only=True,
            )
            provider.embed(["test"])

        _, call_kwargs = mock_client.post.call_args
        assert call_kwargs["json"].get("options") == {"num_gpu": 0}

    def test_embed_cpu_only_false_no_options_key(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[1.0, 2.0]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://localhost:11434",
                model="nomic-embed-text",
                cpu_only=False,
            )
            provider.embed(["test"])

        _, call_kwargs = mock_client.post.call_args
        assert "options" not in call_kwargs["json"]

    def test_embed_raises_on_non_200_status(self) -> None:
        mock_client, _ = _make_httpx_mock(status_code=503, text="Service Unavailable")
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://localhost:11434", model="nomic-embed-text"
            )
            with pytest.raises(RuntimeError, match="503"):
                provider.embed(["hello"])

    def test_dimension_triggers_embed_on_first_access(self) -> None:
        mock_client, _ = _make_httpx_mock(
            json_body={"embeddings": [[0.1, 0.2, 0.3, 0.4]]}
        )
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://localhost:11434", model="nomic-embed-text"
            )
            assert provider.dimension == 4

    def test_dimension_cached_after_first_embed(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[0.1, 0.2, 0.3]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://localhost:11434", model="nomic-embed-text"
            )
            provider.embed(["prime"])
            _ = provider.dimension
            _ = provider.dimension

        assert mock_client.post.call_count == 1

    def test_custom_host_changes_base_url(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[0.9]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://remote-host:12345", model="nomic-embed-text"
            )
            provider.embed(["x"])

        call_args = mock_client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1]["url"]
        assert url == "http://remote-host:12345/api/embed"

    def test_host_trailing_slash_stripped(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[0.1]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://myhost:9999/", model="nomic-embed-text"
            )
            provider.embed(["y"])

        call_args = mock_client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1]["url"]
        assert url == "http://myhost:9999/api/embed"

    def test_custom_model(self) -> None:
        mock_client, _ = _make_httpx_mock(json_body={"embeddings": [[0.3, 0.4]]})
        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaProvider(
                host="http://localhost:11434", model="my-custom-model"
            )
            provider.embed(["test"])

        _, call_kwargs = mock_client.post.call_args
        assert call_kwargs["json"]["model"] == "my-custom-model"
        assert provider.model_name == "my-custom-model"

    def test_missing_httpx_raises_import_error(self) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("No module named 'httpx'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(ImportError, match="httpx"),
        ):
            OllamaProvider(host="http://localhost:11434", model="nomic-embed-text")


class TestOpenAIProvider:
    def test_init_raises_without_api_key(self) -> None:
        with (
            patch("httpx.Client"),
            pytest.raises(RuntimeError, match="non-empty api_key"),
        ):
            OpenAIProvider(api_key="")

    def test_embed_sends_bearer_token(self) -> None:
        mock_client, _ = _make_httpx_mock(
            json_body={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
        )
        with patch("httpx.Client", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test-key-123")
            provider.embed(["hello"])

        _, call_kwargs = mock_client.post.call_args
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key-123"
        assert provider.provider_name == "openai"
        assert provider.model_name == "text-embedding-3-small"

    def test_embed_sorts_by_index(self) -> None:
        mock_client, _ = _make_httpx_mock(
            json_body={
                "data": [
                    {"index": 1, "embedding": [9.0, 8.0]},
                    {"index": 0, "embedding": [1.0, 2.0]},
                ]
            }
        )
        with patch("httpx.Client", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            result = provider.embed(["first", "second"])
        assert result == [[1.0, 2.0], [9.0, 8.0]]

    def test_embed_raises_on_non_200_status(self) -> None:
        mock_client, _ = _make_httpx_mock(status_code=401, text="Unauthorized")
        with patch("httpx.Client", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            with pytest.raises(RuntimeError, match="401"):
                provider.embed(["secret"])

    def test_dimension_caches_after_first_embed(self) -> None:
        mock_client, _ = _make_httpx_mock(
            json_body={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
        )
        with patch("httpx.Client", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            provider.embed(["probe"])
            dim1 = provider.dimension
            dim2 = provider.dimension

        assert dim1 == 3
        assert dim2 == 3
        assert mock_client.post.call_count == 1

    def test_dimension_triggers_embed_when_uncached(self) -> None:
        mock_client, _ = _make_httpx_mock(
            json_body={"data": [{"index": 0, "embedding": [0.5, 0.6, 0.7, 0.8]}]}
        )
        with patch("httpx.Client", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            dim = provider.dimension

        assert dim == 4
        mock_client.post.assert_called_once()

    def test_embed_posts_to_openai_endpoint(self) -> None:
        mock_client, _ = _make_httpx_mock(
            json_body={"data": [{"index": 0, "embedding": [0.1]}]}
        )
        with patch("httpx.Client", return_value=mock_client):
            provider = OpenAIProvider(api_key="sk-test")
            provider.embed(["hello"])

        call_args = mock_client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1]["url"]
        assert url == "https://api.openai.com/v1/embeddings"

    def test_custom_base_url_and_model(self) -> None:
        mock_client, _ = _make_httpx_mock(
            json_body={"data": [{"index": 0, "embedding": [0.1]}]}
        )
        with patch("httpx.Client", return_value=mock_client):
            provider = OpenAIProvider(
                api_key="sk-test",
                base_url="https://api.siliconflow.cn/v1/",
                model="BAAI/bge-m3",
            )
            provider.embed(["hello"])

        call_args = mock_client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1]["url"]
        assert url == "https://api.siliconflow.cn/v1/embeddings"
        _, call_kwargs = mock_client.post.call_args
        assert call_kwargs["json"]["model"] == "BAAI/bge-m3"
        assert provider.model_name == "BAAI/bge-m3"


class TestFastEmbedProvider:
    def test_embed_uses_fastembed_model_and_cache(self) -> None:
        vec = MagicMock()
        vec.tolist.return_value = [0.1, 0.2, 0.3]
        model_instance = MagicMock()
        model_instance.embed.return_value = [vec]
        module = MagicMock()
        module.TextEmbedding.return_value = model_instance

        with patch.dict("sys.modules", {"fastembed": module}):
            provider = FastEmbedProvider(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                cache_dir="/tmp/fastembed-cache",
            )
            result = provider.embed(["hello"])

        module.TextEmbedding.assert_called_once_with(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            cache_dir="/tmp/fastembed-cache",
        )
        model_instance.embed.assert_called_once_with(["hello"], batch_size=32)
        assert result == [[0.1, 0.2, 0.3]]
        assert provider.dimension == 3
        assert provider.provider_name == "fastembed"
        assert provider.model_name == "nomic-ai/nomic-embed-text-v1.5"

    def test_missing_fastembed_raises(self) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(ImportError, match="fastembed"),
        ):
            FastEmbedProvider()

    def test_dimension_raises_on_empty_embeddings(self) -> None:
        model_instance = MagicMock()
        model_instance.embed.return_value = []
        module = MagicMock()
        module.TextEmbedding.return_value = model_instance

        with patch.dict("sys.modules", {"fastembed": module}):
            provider = FastEmbedProvider()
            with pytest.raises(RuntimeError, match="cannot determine dimension"):
                _ = provider.dimension


class TestGetEmbeddingProvider:
    def _ollama_mock_client(self, reachable: bool = True) -> MagicMock:
        mock_client = MagicMock()
        mock_client.__enter__ = lambda _: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200 if reachable else 503
        mock_client.get.return_value = mock_response
        return mock_client

    def test_explicit_openai_returns_openai_provider(self) -> None:
        cfg = _config(
            embedding_provider="openai",
            openai_api_key="sk-test",
            openai_base_url="https://api.siliconflow.cn/v1",
            openai_embedding_model="BAAI/bge-m3",
        )
        with patch("httpx.Client"):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "BAAI/bge-m3"

    def test_explicit_ollama_returns_ollama_provider(self) -> None:
        cfg = _config(embedding_provider="ollama")
        with patch("httpx.Client"):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_explicit_fastembed_returns_fastembed_provider(self) -> None:
        cfg = _config(embedding_provider="fastembed")
        module = MagicMock()
        module.TextEmbedding.return_value = MagicMock(embed=lambda *_: [])
        with patch.dict("sys.modules", {"fastembed": module}):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, FastEmbedProvider)

    def test_explicit_unknown_raises_value_error(self) -> None:
        cfg = _config(embedding_provider="unknown_value")
        with pytest.raises(
            ValueError, match="Valid values: 'openai', 'ollama', 'fastembed'"
        ):
            get_embedding_provider(cfg)

    def test_autodetect_openai_key_present(self) -> None:
        cfg = _config(openai_api_key="sk-autodetect")
        with patch("httpx.Client"):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_autodetect_ollama_reachable(self) -> None:
        cfg = _config()
        probe_client = self._ollama_mock_client(reachable=True)
        ollama_client = MagicMock()
        ollama_client.__enter__ = lambda _: ollama_client
        ollama_client.__exit__ = MagicMock(return_value=False)
        with patch("httpx.Client", side_effect=[probe_client, ollama_client]):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_autodetect_fastembed_fallback(self) -> None:
        cfg = _config()

        probe_client = MagicMock()
        probe_client.__enter__ = lambda _: probe_client
        probe_client.__exit__ = MagicMock(return_value=False)
        probe_client.get.side_effect = ConnectionError("refused")

        module = MagicMock()
        module.TextEmbedding.return_value = MagicMock(embed=lambda *_: [])

        with (
            patch("httpx.Client", return_value=probe_client),
            patch.dict("sys.modules", {"fastembed": module}),
        ):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, FastEmbedProvider)

    def test_autodetect_no_providers_raises_runtime_error(self) -> None:
        cfg = _config()

        probe_client = MagicMock()
        probe_client.__enter__ = lambda _: probe_client
        probe_client.__exit__ = MagicMock(return_value=False)
        probe_client.get.side_effect = ConnectionError("refused")

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        with (
            patch("httpx.Client", return_value=probe_client),
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(RuntimeError, match="No embedding provider"),
        ):
            get_embedding_provider(cfg)


# ---------------------------------------------------------------------------
# Token-budgeted batching (AGO-576)
# ---------------------------------------------------------------------------


class _CappedProvider(EmbeddingProvider):
    """Fake provider that refuses any request over ``cap`` estimated tokens.

    Mimics OpenAI's ``400: Requested N tokens, max M tokens per request``
    so the halve-and-retry path is exercised end to end.  Records every
    batch it was asked for in ``calls``.
    """

    def __init__(self, cap: int, *, dim: int = 4) -> None:
        self.cap = cap
        self._dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        requested = sum(estimate_tokens(t) for t in texts)
        if requested > self.cap:
            raise RuntimeError(
                f"OpenAI API error 400: Requested {requested} tokens, "
                f"max {self.cap} tokens per request"
            )
        return [[float(len(t))] * self._dim for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return "capped"

    @property
    def model_name(self) -> str:
        return "capped-test"


class TestEstimateTokens:
    def test_ascii_is_chars_over_four(self) -> None:
        assert estimate_tokens("a" * 400) == 100

    def test_non_ascii_counts_one_per_char(self) -> None:
        assert estimate_tokens("日本語" * 10) == 30

    def test_empty_is_zero_and_short_is_at_least_one(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("ab") == 1


class TestIsTokenCapError:
    def test_openai_shape_matches(self) -> None:
        exc = RuntimeError(
            "OpenAI API error 400: Requested 547830 tokens, "
            "max 300000 tokens per request"
        )
        assert is_token_cap_error(exc)

    def test_other_errors_do_not_match(self) -> None:
        assert not is_token_cap_error(RuntimeError("OpenAI API error 500: boom"))
        assert not is_token_cap_error(RuntimeError("connection refused"))
        assert not is_token_cap_error(RuntimeError("401: invalid api token"[:3]))


class TestEmbedInTokenBatches:
    def test_splits_under_budget_and_preserves_order(self) -> None:
        provider = _CappedProvider(cap=10_000)
        # 50 texts x 100 tokens = 5000 tokens; a 1000-token budget -> 5 requests.
        texts = [chr(ord("a") + (i % 26)) * 400 for i in range(50)]
        out = embed_in_token_batches(provider, texts, max_tokens_per_request=1000)
        assert len(out) == 50
        assert [len(c) for c in provider.calls] == [10] * 5
        # Order preserved: the fake encodes len(text) into the vector.
        assert [v[0] for v in out] == [float(len(t)) for t in texts]

    def test_halves_and_retries_when_estimate_undershoots(self) -> None:
        # Budget says 4000 fits, but the provider's cap is 1500 — the first
        # request is refused and must be halved until it fits.
        provider = _CappedProvider(cap=1500)
        texts = ["x" * 400] * 40  # 40 x 100 tokens = 4000
        out = embed_in_token_batches(provider, texts, max_tokens_per_request=4000)
        assert len(out) == 40
        # Refused at 40 and 20, accepted at 10 (1000 tokens) — the whole
        # reindex never fails, and nothing is re-sent once accepted.
        assert [len(c) for c in provider.calls] == [40, 20, 10, 10, 20, 10, 10]

    def test_single_oversized_text_fails_alone_with_index(self) -> None:
        provider = _CappedProvider(cap=1000)
        texts = ["small", "y" * 8000, "small too"]  # middle one = 2000 tokens
        with pytest.raises(EmbeddingRequestTooLargeError) as info:
            embed_in_token_batches(provider, texts, max_tokens_per_request=500)
        assert info.value.index == 1
        assert "text #1" in str(info.value)

    def test_non_token_errors_propagate_unchanged(self) -> None:
        provider = MagicMock(spec=EmbeddingProvider)
        provider.max_tokens_per_request = 1000
        provider.embed.side_effect = RuntimeError("OpenAI API error 500: boom")
        with pytest.raises(RuntimeError, match="500"):
            embed_in_token_batches(provider, ["a", "b"])
        assert provider.embed.call_count == 1

    def test_uses_provider_default_budget(self) -> None:
        provider = _CappedProvider(cap=10**9)
        provider.max_tokens_per_request = 100
        texts = ["z" * 200] * 4  # 50 tokens each → 2 per request
        embed_in_token_batches(provider, texts)
        assert [len(c) for c in provider.calls] == [2, 2]

    def test_zero_budget_disables_splitting(self) -> None:
        provider = _CappedProvider(cap=10**9)
        embed_in_token_batches(provider, ["a" * 4000] * 10, max_tokens_per_request=0)
        assert len(provider.calls) == 1

    def test_default_budget_is_under_openai_cap(self) -> None:
        assert 0 < DEFAULT_MAX_TOKENS_PER_REQUEST < 300_000
        assert OpenAIProvider.max_tokens_per_request == DEFAULT_MAX_TOKENS_PER_REQUEST

    def test_get_embedding_provider_applies_configured_budget(self) -> None:
        cfg = _config(
            embedding_provider="openai",
            openai_api_key="test-key",
            embedding_max_tokens_per_request=12_345,
        )
        provider = get_embedding_provider(cfg)
        assert provider.max_tokens_per_request == 12_345


class TestVectorIndexAddNamesPath:
    def test_oversized_chunk_error_carries_document_path(self) -> None:
        pytest.importorskip("numpy")
        from markdown_vault_mcp.vector_index import VectorIndex

        provider = _CappedProvider(cap=1000)
        provider.max_tokens_per_request = 500
        index = VectorIndex(provider)
        texts = ["ok", "w" * 8000]
        meta = [{"path": "a.md"}, {"path": "big/lattice.md"}]
        with pytest.raises(EmbeddingRequestTooLargeError) as info:
            index.add(texts, meta)
        assert info.value.path == "big/lattice.md"
        assert "big/lattice.md" in str(info.value)
        assert index.count == 0  # nothing mutated on failure

    def test_add_splits_large_document_into_several_requests(self) -> None:
        pytest.importorskip("numpy")
        from markdown_vault_mcp.vector_index import VectorIndex

        provider = _CappedProvider(cap=10**9)
        provider.max_tokens_per_request = 300
        index = VectorIndex(provider)
        texts = ["q" * 400] * 6  # 100 tokens each → 3 per request
        meta = [{"path": "doc.md"}] * 6
        assert index.add(texts, meta) == 6
        assert [len(c) for c in provider.calls] == [3, 3]
