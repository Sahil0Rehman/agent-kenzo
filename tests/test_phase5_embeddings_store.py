"""Phase 5 "Semantic Memory and Retrieval Engine" tests -- Part 1: the
embedding foundation (`kenzo/memory/vector/embeddings.py`), the vector store
(`kenzo/memory/vector/store.py`'s `QdrantStore`), and `SemanticIndexer`'s
chunking (`kenzo/memory/vector/semantic_indexer.py`).

Faking approach for the LiteLLM-backed provider mirrors `tests/test_router.py`:
`litellm.embedding` is monkeypatched to a fake, in-process function so no real
network call is ever made -- never `litellm.completion` (that's the chat
path; embeddings use the separate `litellm.embedding` entry point).

`QdrantStore` tests use real `":memory:"` Qdrant instances -- this mode is
fully offline/in-process (no server, no mocking needed), per the module's own
docstring.
"""

from __future__ import annotations

from pathlib import Path

import litellm
import pytest

from kenzo.memory.vector.embeddings import (
    EmbeddingProvider,
    EmbeddingRouter,
    HashingEmbeddingProvider,
    LiteLLMEmbeddingProvider,
)
from kenzo.memory.vector.models import CodeChunk
from kenzo.memory.vector.semantic_indexer import SemanticIndexer
from kenzo.memory.vector.store import QdrantStore
from kenzo.config.settings import EmbeddingConfig
from kenzo.intelligence.symbol_index import SymbolIndex

# ---------------------------------------------------------------------------
# 1a. HashingEmbeddingProvider -- deterministic, offline, dimensions honored
# ---------------------------------------------------------------------------


def test_hashing_provider_is_deterministic_across_calls_and_instances():
    provider_a = HashingEmbeddingProvider(dimensions=256)
    provider_b = HashingEmbeddingProvider(dimensions=256)

    text = "AuthService.terminateSession"
    vec1 = provider_a.embed(text)
    vec2 = provider_a.embed(text)  # same instance, second call
    vec3 = provider_b.embed(text)  # brand new instance

    assert vec1 == vec2
    assert vec1 == vec3


def test_hashing_provider_dimensions_match_config():
    provider = HashingEmbeddingProvider(dimensions=64)
    assert provider.dimensions == 64
    assert len(provider.embed("hello world")) == 64

    router = EmbeddingRouter(EmbeddingConfig(provider="local", dimensions=128))
    assert router.dimensions == 128
    assert len(router.embed("hello world")) == 128


def test_hashing_provider_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError):
        HashingEmbeddingProvider(dimensions=0)


def test_hashing_provider_embed_batch_matches_individual_embed_calls():
    provider = HashingEmbeddingProvider(dimensions=32)
    texts = ["logout the user", "revokePermission(userId, resource)", "a totally unrelated sentence"]

    batch_result = provider.embed_batch(texts)
    individual_result = [provider.embed(t) for t in texts]

    assert batch_result == individual_result
    assert len(batch_result) == len(texts)
    for vec in batch_result:
        assert len(vec) == 32


def test_hashing_provider_different_text_yields_different_vectors():
    provider = HashingEmbeddingProvider(dimensions=64)
    assert provider.embed("login") != provider.embed("completely different text about pizza")


def test_embedding_router_selects_hashing_for_local_provider():
    router = EmbeddingRouter(EmbeddingConfig(provider="local", model="hashing-v1", dimensions=32))
    assert isinstance(router._provider, HashingEmbeddingProvider)


# ---------------------------------------------------------------------------
# 1b. LiteLLMEmbeddingProvider -- litellm.embedding() faked, never real network
# ---------------------------------------------------------------------------


class _FakeEmbeddingResponse:
    """Minimal stand-in shaped like litellm's real embedding response: a
    `.data` list of dict-like items each holding an `"embedding"` key --
    mirrors `LiteLLMEmbeddingProvider.embed_batch`'s own
    `[item["embedding"] for item in response.data]` extraction."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [{"embedding": v} for v in vectors]


def test_litellm_provider_extracts_embeddings_from_response_data(monkeypatch):
    captured: dict = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return _FakeEmbeddingResponse([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    provider = LiteLLMEmbeddingProvider(
        provider="openai", model="text-embedding-3-small", api_key="fake-key", api_base=None, dimensions=3
    )

    result = provider.embed_batch(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert captured["model"] == "openai/text-embedding-3-small"
    assert captured["input"] == ["hello", "world"]
    assert captured["api_key"] == "fake-key"
    assert "api_base" not in captured  # None api_base must not be forwarded


def test_litellm_provider_embed_single_delegates_to_embed_batch(monkeypatch):
    def fake_embedding(**kwargs):
        return _FakeEmbeddingResponse([[9.0, 8.0]])

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    provider = LiteLLMEmbeddingProvider(provider="openai", model="text-embedding-3-small", dimensions=2)
    assert provider.embed("solo text") == [9.0, 8.0]


def test_litellm_provider_resolves_already_qualified_model_string(monkeypatch):
    captured: dict = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return _FakeEmbeddingResponse([[1.0]])

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    # A model string that already contains "/" is used verbatim, not re-prefixed.
    provider = LiteLLMEmbeddingProvider(provider="ollama", model="custom/already-qualified-model", dimensions=1)
    provider.embed("x")

    assert captured["model"] == "custom/already-qualified-model"


def test_embedding_router_selects_litellm_for_non_local_provider(monkeypatch):
    def fake_embedding(**kwargs):
        return _FakeEmbeddingResponse([[1.0, 2.0]])

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    router = EmbeddingRouter(EmbeddingConfig(provider="openai", model="text-embedding-3-small", dimensions=2))
    assert isinstance(router._provider, LiteLLMEmbeddingProvider)
    assert router.embed("x") == [1.0, 2.0]


def test_fake_provider_implements_real_embedding_provider_abc():
    """Sanity check that the ABC itself is meaningfully abstract (used by the
    fake provider built for the required semantic-query tests in
    test_phase5_search_and_semantic.py)."""
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 2. QdrantStore -- real ":memory:" round trips (add/search/update/delete/filters)
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, symbol: str, chunk_type: str = "method", language: str = "java", content: str = "") -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        file=f"{symbol}.java",
        symbol=symbol,
        content=content or f"content for {symbol}",
        language=language,
        chunk_type=chunk_type,
        metadata={},
    )


def test_qdrant_store_add_and_search_round_trip():
    store = QdrantStore(collection_name="phase5_add_search", dimensions=4, url=":memory:")

    chunks = [
        _chunk("a", "ClassA"),
        _chunk("b", "ClassB"),
        _chunk("c", "ClassC"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.add(chunks, embeddings)

    assert store.count() == 3

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3
    top_chunk, top_score = results[0]
    assert top_chunk.id == "a"
    assert top_score == pytest.approx(1.0, abs=1e-6)


def test_qdrant_store_update_upserts_without_duplicating():
    store = QdrantStore(collection_name="phase5_update", dimensions=4, url=":memory:")

    original = _chunk("a", "ClassA", content="original content")
    store.add([original], [[1.0, 0.0, 0.0, 0.0]])
    assert store.count() == 1

    updated = _chunk("a", "ClassA", content="UPDATED content, renamed logic")
    store.update([updated], [[0.0, 1.0, 0.0, 0.0]])

    # Same id -> upsert in place, never a second point.
    assert store.count() == 1

    results = store.search([0.0, 1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0][0].content == "UPDATED content, renamed logic"


def test_qdrant_store_delete_removes_chunk():
    store = QdrantStore(collection_name="phase5_delete", dimensions=4, url=":memory:")

    chunks = [_chunk("a", "ClassA"), _chunk("b", "ClassB")]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store.add(chunks, embeddings)
    assert store.count() == 2

    store.delete(["a"])
    assert store.count() == 1

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert all(c.id != "a" for c, _score in results)


def test_qdrant_store_delete_nonexistent_id_is_a_silent_noop():
    store = QdrantStore(collection_name="phase5_delete_noop", dimensions=4, url=":memory:")
    store.add([_chunk("a", "ClassA")], [[1.0, 0.0, 0.0, 0.0]])

    store.delete(["does-not-exist"])  # must not raise

    assert store.count() == 1


def test_qdrant_store_search_filters_by_chunk_type():
    store = QdrantStore(collection_name="phase5_filter_chunk_type", dimensions=4, url=":memory:")

    chunks = [
        _chunk("cls", "AuthService", chunk_type="class"),
        _chunk("m1", "AuthService.login", chunk_type="method"),
        _chunk("m2", "AuthService.logout", chunk_type="method"),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0]] * 3
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, filters={"chunk_type": "method"})

    assert len(results) == 2
    assert {c.id for c, _score in results} == {"m1", "m2"}
    assert all(c.chunk_type == "method" for c, _score in results)


def test_qdrant_store_search_filters_by_language():
    store = QdrantStore(collection_name="phase5_filter_language", dimensions=4, url=":memory:")

    chunks = [
        _chunk("java1", "JavaClass", language="java"),
        _chunk("py1", "python_func", language="python"),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0]] * 2
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, filters={"language": "python"})

    assert len(results) == 1
    assert results[0][0].id == "py1"


def test_qdrant_store_add_with_empty_lists_is_a_noop():
    store = QdrantStore(collection_name="phase5_empty_add", dimensions=4, url=":memory:")
    store.add([], [])
    assert store.count() == 0


def test_qdrant_store_add_raises_on_length_mismatch():
    store = QdrantStore(collection_name="phase5_mismatch", dimensions=4, url=":memory:")
    with pytest.raises(ValueError):
        store.add([_chunk("a", "A")], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def test_fresh_qdrant_memory_instances_do_not_share_state():
    """":memory:" is real and offline, but NOT persisted/shared across
    separate instances -- each fresh QdrantStore(url=":memory:") starts
    empty, per the module's own documented behavior."""
    store1 = QdrantStore(collection_name="phase5_isolation", dimensions=4, url=":memory:")
    store1.add([_chunk("a", "A")], [[1.0, 0.0, 0.0, 0.0]])
    assert store1.count() == 1

    store2 = QdrantStore(collection_name="phase5_isolation", dimensions=4, url=":memory:")
    assert store2.count() == 0


# ---------------------------------------------------------------------------
# 3. SemanticIndexer chunking -- one chunk per class, one per method, one per
#    top-level function, NEVER a fixed-size slice. Real Java fixture, reusing
#    the shared `spring_boot_repo` conftest fixture.
# ---------------------------------------------------------------------------

LOGIN_CONTROLLER_PATH = "src/main/java/com/example/auth/LoginController.java"
LOGOUT_CONTROLLER_PATH = "src/main/java/com/example/auth/LogoutController.java"
AUTH_SERVICE_PATH = "src/main/java/com/example/auth/AuthService.java"
USER_REPOSITORY_PATH = "src/main/java/com/example/auth/UserRepository.java"


def _build_symbol_index(repo_root: Path) -> SymbolIndex:
    symbol_index = SymbolIndex(str(repo_root))
    symbol_index.build()
    return symbol_index


def _pure_indexer(repo_root: Path, symbol_index: SymbolIndex) -> SemanticIndexer:
    # embedding_router/vector_store are never touched by chunk_file() (pure,
    # no side effects) -- None stands in for "not needed for this test",
    # exactly like kenzo.core.code_search.CodeSearchEngine's own `_chunker`
    # reuses SemanticIndexer purely for chunk_file.
    return SemanticIndexer(str(repo_root), symbol_index, embedding_router=None, vector_store=None)


def test_chunk_file_produces_exactly_one_chunk_per_class_and_per_method(spring_boot_repo):
    symbol_index = _build_symbol_index(spring_boot_repo)
    indexer = _pure_indexer(spring_boot_repo, symbol_index)

    total_chunks = 0
    total_expected = 0
    all_chunk_types: set[str] = set()

    for file_path, node in symbol_index.files.items():
        chunks = indexer.chunk_file(file_path)
        expected = len(node.classes) + sum(len(cls.methods) for cls in node.classes) + len(node.functions)

        assert len(chunks) == expected, f"{file_path}: expected {expected} chunks, got {len(chunks)}"
        total_chunks += len(chunks)
        total_expected += expected
        all_chunk_types.update(c.chunk_type for c in chunks)

    assert total_chunks == total_expected
    assert total_chunks > 0
    # Never a fixed-size/arbitrary chunk kind -- only these three ever appear
    # in this fixture (no top-level functions in Java, so "function" doesn't
    # show up here, but "module" must never appear either).
    assert all_chunk_types <= {"class", "method", "function"}
    assert "module" not in all_chunk_types


def test_chunk_file_logout_controller_has_exactly_class_and_method_chunks(spring_boot_repo):
    symbol_index = _build_symbol_index(spring_boot_repo)
    indexer = _pure_indexer(spring_boot_repo, symbol_index)

    chunks = indexer.chunk_file(LOGOUT_CONTROLLER_PATH)

    # One class chunk, plus one method chunk per member of LogoutController's
    # `methods` list -- which (per kenzo.intelligence.models.FunctionSymbol's
    # `is_constructor` flag) includes the constructor alongside logout(), so
    # 1 class + 2 methods (constructor, logout) = 3, never a fixed-size slice.
    assert len(chunks) == 3
    assert {c.chunk_type for c in chunks} == {"class", "method"}

    class_chunk = next(c for c in chunks if c.chunk_type == "class")
    assert class_chunk.symbol == "LogoutController"
    assert class_chunk.language == "java"
    assert "Dependencies: AuthService" in class_chunk.content

    method_chunk = next(c for c in chunks if c.symbol == "LogoutController.logout")
    assert method_chunk.language == "java"
    assert method_chunk.chunk_type == "method"
    # Real signature info -- endpoint and parameters -- must show up in the
    # rendered text actually being embedded, not just structurally in metadata.
    assert "Endpoint: /api/auth/logout" in method_chunk.content
    assert "Parameters: none" in method_chunk.content
    assert "logout" in method_chunk.content.lower()


def test_chunk_file_method_content_includes_real_parameter_signature(spring_boot_repo):
    """UserRepository.findByUsername(String username) -- a method WITH a
    parameter -- proves parameter rendering isn't a hardcoded "none"."""
    symbol_index = _build_symbol_index(spring_boot_repo)
    indexer = _pure_indexer(spring_boot_repo, symbol_index)

    chunks = indexer.chunk_file(USER_REPOSITORY_PATH)
    method_chunk = next(c for c in chunks if c.symbol == "UserRepository.findByUsername")

    assert "username" in method_chunk.content
    assert "String" in method_chunk.content


def test_chunk_file_returns_empty_list_for_unknown_file(spring_boot_repo):
    symbol_index = _build_symbol_index(spring_boot_repo)
    indexer = _pure_indexer(spring_boot_repo, symbol_index)

    assert indexer.chunk_file("does/not/exist.java") == []


def test_index_file_embeds_and_stores_then_cleans_up_stale_chunks_on_rename(spring_boot_repo):
    """index_file() (unlike chunk_file()) actually embeds+stores, and cleans
    up chunks for symbols that no longer exist after a file changes -- e.g. a
    renamed method."""
    symbol_index = _build_symbol_index(spring_boot_repo)
    provider = HashingEmbeddingProvider(dimensions=32)
    store = QdrantStore(collection_name="phase5_index_file", dimensions=32, url=":memory:")
    indexer = SemanticIndexer(str(spring_boot_repo), symbol_index, provider, store)

    written = indexer.index_file(LOGOUT_CONTROLLER_PATH)
    assert written == 3  # class + constructor + logout method
    assert store.count() == 3

    # Rename the method on disk (logout -> signOut) and reindex the symbol
    # table + semantic index for just this file.
    abs_path = spring_boot_repo / LOGOUT_CONTROLLER_PATH
    renamed_source = abs_path.read_text(encoding="utf-8").replace("public String logout()", "public String signOut()")
    abs_path.write_text(renamed_source, encoding="utf-8")
    symbol_index.reindex_file(LOGOUT_CONTROLLER_PATH)

    written_again = indexer.index_file(LOGOUT_CONTROLLER_PATH)
    assert written_again == 3  # still class + constructor + one method, just renamed

    # The old "LogoutController.logout" chunk id must be gone (stale-cleanup),
    # and the store must not have grown (upsert/delete, not just accumulate).
    assert store.count() == 3
    remaining_ids = {c.id for c, _score in store.search(provider.embed("logout"), top_k=10)}
    assert f"{LOGOUT_CONTROLLER_PATH}::LogoutController.logout" not in remaining_ids
    assert f"{LOGOUT_CONTROLLER_PATH}::LogoutController.signOut" in remaining_ids


def test_remove_file_deletes_all_its_chunks(spring_boot_repo):
    symbol_index = _build_symbol_index(spring_boot_repo)
    provider = HashingEmbeddingProvider(dimensions=32)
    store = QdrantStore(collection_name="phase5_remove_file", dimensions=32, url=":memory:")
    indexer = SemanticIndexer(str(spring_boot_repo), symbol_index, provider, store)

    indexer.index_file(LOGOUT_CONTROLLER_PATH)
    indexer.index_file(AUTH_SERVICE_PATH)
    total_before = store.count()
    assert total_before > 0

    indexer.remove_file(LOGOUT_CONTROLLER_PATH)

    assert store.count() < total_before
    remaining_hits = store.search(provider.embed("anything"), top_k=50)
    assert all(not c.file == LOGOUT_CONTROLLER_PATH for c, _score in remaining_hits)
