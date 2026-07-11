"""Tests for kenzo.tools.symbols.find_symbol / map_flow, backed by the Java
analyzer's output as persisted via RepositoryMemory.save_symbols
(.kenzo/symbols.json), using the spring_boot_repo fixture.
"""

from __future__ import annotations

from kenzo.memory.java_analyzer import analyze_java_repository
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.symbols import find_symbol, map_flow


def _index_symbols(repo_root) -> None:
    symbols = analyze_java_repository(str(repo_root))
    RepositoryMemory(str(repo_root)).save_symbols(symbols)


# ---------------------------------------------------------------------------
# find_symbol
# ---------------------------------------------------------------------------


def test_find_symbol_full_class_match(spring_boot_repo):
    _index_symbols(spring_boot_repo)

    result = find_symbol("LogoutController", repo_root=str(spring_boot_repo))

    assert result.success is True
    assert result.data["symbol"] is not None
    assert result.data["symbol"]["name"] == "LogoutController"
    assert result.data["symbol"]["kind"] == "class"
    assert result.data["symbol"]["type"] == "controller"
    class_entries = [m for m in result.data["matches"] if m.get("kind") != "method"]
    assert len(class_entries) == 1


def test_find_symbol_method_name_match(spring_boot_repo):
    _index_symbols(spring_boot_repo)

    result = find_symbol("logout", repo_root=str(spring_boot_repo))

    assert result.success is True
    method_entries = [m for m in result.data["matches"] if m.get("kind") == "method"]
    assert len(method_entries) >= 1
    assert any(
        entry["class"] == "LogoutController" and entry["name"] == "logout"
        for entry in method_entries
    )


def test_find_symbol_no_match_returns_success_with_empty_list(spring_boot_repo):
    _index_symbols(spring_boot_repo)

    result = find_symbol("doesnotexist12345", repo_root=str(spring_boot_repo))

    assert result.success is True
    assert result.data["matches"] == []
    assert result.data["symbol"] is None


def test_find_symbol_without_indexed_symbols_returns_clear_error(tmp_path):
    repo_root = tmp_path / "unindexed_repo"
    repo_root.mkdir()

    result = find_symbol("Anything", repo_root=str(repo_root))

    assert result.success is False
    assert "kenzo index" in result.error


# ---------------------------------------------------------------------------
# map_flow
# ---------------------------------------------------------------------------


def test_map_flow_traces_controller_to_service(spring_boot_repo):
    _index_symbols(spring_boot_repo)

    result = map_flow("/api/auth/logout", repo_root=str(spring_boot_repo))

    assert result.success is True
    chain = result.data["chain"]
    assert len(chain) >= 2

    controller_hop = chain[0]
    assert controller_hop["role"] == "controller"
    assert controller_hop["class"] == "LogoutController"
    assert controller_hop["method"] == "logout"

    service_hop = chain[1]
    assert service_hop["role"] == "service"
    assert service_hop["class"] == "AuthService"
    assert service_hop["method"] == "terminateSession"


def test_map_flow_nonexistent_endpoint_returns_failure(spring_boot_repo):
    _index_symbols(spring_boot_repo)

    result = map_flow("/no/such/endpoint", repo_root=str(spring_boot_repo))

    assert result.success is False
    assert result.error is not None


def test_map_flow_without_indexed_symbols_returns_clear_error(tmp_path):
    repo_root = tmp_path / "unindexed_repo"
    repo_root.mkdir()

    result = map_flow("/api/auth/logout", repo_root=str(repo_root))

    assert result.success is False
    assert "kenzo index" in result.error
