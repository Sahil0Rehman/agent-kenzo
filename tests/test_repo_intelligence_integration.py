"""End-to-end integration test for Phase 2 "Repository Intelligence": proves
that `kenzo.core.runtime.AgentRuntime` actually wires project detection and
Java symbol extraction together at construction time (not just each piece in
isolation), using the realistic `spring_boot_repo` fixture.

Faking approach mirrors test_runtime.py: an explicit in-memory `KenzoConfig`
(never `load_config()`), and `kenzo.models.router.ModelRouter.complete`
monkeypatched so the model is provably never invoked -- `AgentRuntime.__init__`
only detects the project/symbols and builds the graph, it never calls the
model, so this doubles as a check that construction stays offline.
"""

from __future__ import annotations

import json

from kenzo.config.settings import KenzoConfig, PermissionsConfig, ProviderConfig
from kenzo.core.runtime import AgentRuntime
from kenzo.memory.repository import RepositoryMemory
from kenzo.models.router import ModelRouter


def _build_config() -> KenzoConfig:
    return KenzoConfig(
        providers={"nemotron": ProviderConfig(api_key="fake-key", default_model="fake-model")},
        permissions=PermissionsConfig(allow_write=True, allow_shell_exec=False),
    )


def _forbid_model_calls(monkeypatch) -> None:
    """AgentRuntime construction must never reach the model layer -- project
    detection and Java symbol extraction are both pure local analysis."""

    def _unexpected_complete(self, *args, **kwargs):
        raise AssertionError("ModelRouter.complete must not be called during AgentRuntime construction")

    monkeypatch.setattr(ModelRouter, "complete", _unexpected_complete)


def test_runtime_detects_java_maven_spring_boot_and_persists_symbols(spring_boot_repo, monkeypatch):
    _forbid_model_calls(monkeypatch)
    config = _build_config()

    runtime = AgentRuntime(repo_path=str(spring_boot_repo), config=config)

    assert runtime.project_info.language == "java"
    assert runtime.project_info.build == "maven"
    assert runtime.project_info.framework == "spring-boot"
    assert runtime.project_info.test_command == "mvn test"


def test_runtime_persists_symbols_json_with_expected_classes(spring_boot_repo, monkeypatch):
    _forbid_model_calls(monkeypatch)
    config = _build_config()

    AgentRuntime(repo_path=str(spring_boot_repo), config=config)

    symbols_path = spring_boot_repo / ".kenzo" / "symbols.json"
    assert symbols_path.exists()

    symbols_data = json.loads(symbols_path.read_text(encoding="utf-8"))
    symbol_names = {s["name"] for s in symbols_data["symbols"]}
    assert {"LogoutController", "AuthService", "UserRepository", "LoginController"}.issubset(symbol_names)

    # Also reachable via the RepositoryMemory facade, not just the raw file.
    loaded = RepositoryMemory(str(spring_boot_repo)).load_symbols()
    assert loaded is not None
    assert {s["name"] for s in loaded["symbols"]} == symbol_names


def test_runtime_project_json_persisted_via_repository_memory(spring_boot_repo, monkeypatch):
    _forbid_model_calls(monkeypatch)
    config = _build_config()

    AgentRuntime(repo_path=str(spring_boot_repo), config=config)

    project_info = RepositoryMemory(str(spring_boot_repo)).load_project_info()
    assert project_info is not None
    assert project_info["language"] == "java"
    assert project_info["build"] == "maven"
    assert project_info["test_command"] == "mvn test"


def test_context_bundle_surfaces_project_and_test_command_after_runtime_construction(
    spring_boot_repo, monkeypatch
):
    _forbid_model_calls(monkeypatch)
    config = _build_config()

    AgentRuntime(repo_path=str(spring_boot_repo), config=config)

    bundle = RepositoryMemory(str(spring_boot_repo)).build_context_bundle()

    assert "spring-boot" in bundle
    assert "mvn test" in bundle
    assert "# Project" in bundle
    assert "# Java Symbols" in bundle
