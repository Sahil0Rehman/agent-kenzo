"""Phase 5 "Semantic Memory and Retrieval Engine" tests -- Part 4:
`kenzo/core/indexing_service.py`'s `IndexingService` (synchronous and
background threading behavior, plus incremental `reindex_file`), and a full
`AgentRuntime` integration test proving the `semantic_search` tool is wired
in end to end.

Faking approach for the integration test mirrors
`tests/test_repo_intelligence_integration.py`: an explicit in-memory
`KenzoConfig` (never `load_config()`), and `ModelRouter.complete` monkeypatched
to raise if ever called, so the real, offline `AgentRuntime` construction (and
the `semantic_search` tool call itself, which never touches the model) is
provably never reaching a real network call.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kenzo.config.settings import KenzoConfig, PermissionsConfig, ProviderConfig
from kenzo.core.indexing_service import IndexingService
from kenzo.core.runtime import AgentRuntime
from kenzo.memory.repository import RepositoryMemory
from kenzo.models.router import ModelRouter

LOGOUT_CONTROLLER_PATH = "src/main/java/com/example/auth/LogoutController.java"

# ---------------------------------------------------------------------------
# 9. IndexingService -- synchronous (background=False) behavior, deterministic
# ---------------------------------------------------------------------------


def _service(repo_root: Path, background: bool = False) -> IndexingService:
    memory = RepositoryMemory(str(repo_root))
    config = KenzoConfig()  # default embedding.provider="local", vector_store.url=":memory:"
    service = IndexingService(str(repo_root), memory, config)
    service.start(background=background)
    return service


def test_indexing_service_synchronous_start_builds_full_index_deterministically(spring_boot_repo):
    service = _service(spring_boot_repo, background=False)

    assert service.is_ready() is True
    assert service.last_error is None
    assert service.wait_until_ready(timeout=0) is True  # already done, returns immediately

    class_names = {cls.name for _path, cls in service.symbol_index.all_classes()}
    assert {"LoginController", "LogoutController", "AuthService", "UserRepository"}.issubset(class_names)

    # Phase 4 structures are populated too, not just the symbol index.
    assert service.reference_graph.find_callers("AuthService.authenticate")
    assert service.vector_store.count() > 0  # semantic index was actually built


def test_indexing_service_synchronous_start_persists_to_disk(spring_boot_repo):
    _service(spring_boot_repo, background=False)

    memory = RepositoryMemory(str(spring_boot_repo))
    assert memory.code_index_path.exists()
    assert memory.references_path.exists()
    assert memory.dependencies_path.exists()
    assert memory.semantic_chunk_ids_path.exists()


def test_indexing_service_reindex_file_updates_symbol_index_and_vector_store(spring_boot_repo):
    service = _service(spring_boot_repo, background=False)
    baseline_count = service.vector_store.count()
    baseline_methods = {
        m.name for _path, cls in service.symbol_index.all_classes() if cls.name == "LogoutController" for m in cls.methods
    }
    assert "logout" in baseline_methods
    assert "signOut" not in baseline_methods

    # Edit the file on disk: add a brand new method alongside logout().
    abs_path = spring_boot_repo / LOGOUT_CONTROLLER_PATH
    original = abs_path.read_text(encoding="utf-8")
    updated = original.replace(
        "    @PostMapping(\"/logout\")\n    public String logout() {\n        authService.terminateSession();\n        return \"logged out\";\n    }\n",
        (
            "    @PostMapping(\"/logout\")\n    public String logout() {\n        authService.terminateSession();\n        return \"logged out\";\n    }\n\n"
            "    @PostMapping(\"/force-logout\")\n    public String forceLogout() {\n        authService.terminateSession();\n        return \"forced\";\n    }\n"
        ),
    )
    assert updated != original, "expected the replacement to actually match and change the source"
    abs_path.write_text(updated, encoding="utf-8")

    service.reindex_file(LOGOUT_CONTROLLER_PATH)

    updated_methods = {
        m.name for _path, cls in service.symbol_index.all_classes() if cls.name == "LogoutController" for m in cls.methods
    }
    assert "forceLogout" in updated_methods
    assert "logout" in updated_methods  # the original method is still there too

    # One brand new chunk (the new method) -- the class chunk was upserted in
    # place (same id), so the count grows by exactly the new method chunk.
    assert service.vector_store.count() == baseline_count + 1


def test_indexing_service_reindex_file_persists_updated_state(spring_boot_repo):
    service = _service(spring_boot_repo, background=False)

    abs_path = spring_boot_repo / LOGOUT_CONTROLLER_PATH
    abs_path.write_text(abs_path.read_text(encoding="utf-8"), encoding="utf-8")  # touch, no real change needed
    service.reindex_file(LOGOUT_CONTROLLER_PATH)

    memory = RepositoryMemory(str(spring_boot_repo))
    assert memory.code_index_path.exists()
    assert memory.semantic_chunk_ids_path.exists()


def test_indexing_service_reindex_file_removes_symbols_for_deleted_file(spring_boot_repo):
    service = _service(spring_boot_repo, background=False)
    assert service.symbol_index.get_file(LOGOUT_CONTROLLER_PATH) is not None

    (spring_boot_repo / LOGOUT_CONTROLLER_PATH).unlink()
    service.reindex_file(LOGOUT_CONTROLLER_PATH)

    assert service.symbol_index.get_file(LOGOUT_CONTROLLER_PATH) is None
    class_names = {cls.name for _path, cls in service.symbol_index.all_classes()}
    assert "LogoutController" not in class_names


# ---------------------------------------------------------------------------
# 9b. IndexingService -- background=True: construction must not block
# ---------------------------------------------------------------------------

_MANY_FILES_COUNT = 60


@pytest.fixture
def many_small_files_repo(tmp_path) -> Path:
    """A repo with enough real Python files (each with a real function) that
    a full symbol-index + reference/dependency-graph + embed-and-store pass
    takes long enough to observe `is_ready()` still being False immediately
    after `start(background=True)` returns."""
    root = tmp_path / "many_small_files_repo"
    root.mkdir()
    for i in range(_MANY_FILES_COUNT):
        (root / f"module_{i:03d}.py").write_text(
            f"def function_{i:03d}(value):\n"
            f"    \"\"\"Doubles the input for module {i:03d}.\"\"\"\n"
            f"    return value * 2 + {i}\n",
            encoding="utf-8",
        )
    return root


def test_indexing_service_background_start_does_not_block_and_becomes_ready(many_small_files_repo):
    memory = RepositoryMemory(str(many_small_files_repo))
    config = KenzoConfig()
    service = IndexingService(str(many_small_files_repo), memory, config)

    start = time.perf_counter()
    service.start(background=True)
    elapsed = time.perf_counter() - start

    # The real claim: construction does not block on a full multi-file
    # walk/parse/embed pass, not a tight microbenchmark -- generous bound.
    assert elapsed < 5.0

    # Immediately after start() returns, the background build is very likely
    # still running against ~60 real files (parse + embed + Qdrant upsert
    # each) -- is_ready() reflects that rather than a pre-completed state.
    initial_ready = service.is_ready()
    assert initial_ready is False

    became_ready = service.wait_until_ready(timeout=30)
    assert became_ready is True
    assert service.is_ready() is True
    assert service.last_error is None

    assert len(service.symbol_index.files) == _MANY_FILES_COUNT
    assert service.vector_store.count() == _MANY_FILES_COUNT  # one function chunk per file


# ---------------------------------------------------------------------------
# 10. Full AgentRuntime integration: semantic_search tool wired end to end
# ---------------------------------------------------------------------------


def _build_config() -> KenzoConfig:
    return KenzoConfig(
        providers={"nemotron": ProviderConfig(api_key="fake-key", default_model="fake-model")},
        permissions=PermissionsConfig(allow_write=True, allow_shell_exec=False),
    )


def _forbid_model_calls(monkeypatch) -> None:
    def _unexpected_complete(self, *args, **kwargs):
        raise AssertionError("ModelRouter.complete must not be called in this test")

    monkeypatch.setattr(ModelRouter, "complete", _unexpected_complete)


def test_agent_runtime_wires_semantic_search_tool_and_executes_it_successfully(spring_boot_repo, monkeypatch):
    _forbid_model_calls(monkeypatch)
    config = _build_config()

    runtime = AgentRuntime(repo_path=str(spring_boot_repo), config=config)

    assert "semantic_search" in runtime.tools.names()

    ready = runtime.indexing_service.wait_until_ready(timeout=30)
    assert ready is True
    assert runtime.indexing_service.last_error is None

    result = runtime.tools.execute("semantic_search", {"query": "LogoutController", "top_k": 5})

    assert result.success is True
    assert result.data["results"], f"expected non-empty results, got: {result.data}"
    top_symbols = {r["symbol"] for r in result.data["results"]}
    assert "LogoutController" in top_symbols


def test_agent_runtime_session_and_architecture_memory_are_wired_into_controller(spring_boot_repo, monkeypatch):
    _forbid_model_calls(monkeypatch)
    config = _build_config()

    runtime = AgentRuntime(repo_path=str(spring_boot_repo), config=config)

    assert runtime.controller.session_memory is runtime.session_memory
    assert runtime.controller.architecture_memory is runtime.architecture_memory
    assert runtime.controller.indexing_service is runtime.indexing_service
