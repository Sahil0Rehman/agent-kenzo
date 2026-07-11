"""Tests for Phase 4 "Deep Code Intelligence Engine"
(`kenzo/intelligence/*.py` + `kenzo/tools/code_intelligence.py`), covering:

  1. Symbol extraction over a richer Spring Boot project (controller/service/
     repository/entity/DTO/test) -- stereotypes, fields, endpoint composition,
     constructor-injection dependencies.
  2. Cross-file reference resolution (`ReferenceGraph`).
  3. Forward flow tracing (`kenzo.tools.code_intelligence.trace_flow`) for
     both an HTTP endpoint and a qualified method reference.
  4. Architecture summarization (`DependencyGraph.explain_architecture`).
  5. Test discovery (`kenzo.intelligence.test_discovery.find_related_tests`).
  6. Incremental single-file re-indexing correctness (`SymbolIndex.reindex_file`).
  7. The `kenzo.tools.code_intelligence` tool layer's real persist-then-fresh-
     load round trip via `.kenzo/code_index.json` / `references.json` /
     `dependencies.json`, plus its graceful "nothing indexed yet" failure mode.

Deliberately does NOT touch/import the shared `spring_boot_repo` fixture in
`tests/conftest.py` (used by many Phase 2 tests) -- a new, richer fixture is
defined locally below instead, so there is zero risk of perturbing those
tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.reference_graph import ReferenceGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.intelligence.test_discovery import find_related_tests
from kenzo.memory.repository import RepositoryMemory
from kenzo.tools.code_intelligence import (
    explain_architecture,
    find_callers,
    find_related_tests_tool,
    trace_flow,
)

# ---------------------------------------------------------------------------
# deep_spring_boot_repo: a richer fake Spring Boot Maven project than the
# shared Phase 2 `spring_boot_repo` fixture -- adds an Entity, a DTO, and a
# test file, specifically for Phase 4 intelligence tests.
# ---------------------------------------------------------------------------

_POM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>

    <groupId>com.example</groupId>
    <artifactId>user-service</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <java.version>17</java.version>
    </properties>
</project>
"""

_USER_CONTROLLER_JAVA = """\
package com.example.app;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/{id}")
    public UserDto getUser(@PathVariable Long id) {
        return userService.findUserById(id);
    }
}
"""

_USER_SERVICE_JAVA = """\
package com.example.app;

import org.springframework.stereotype.Service;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public UserDto findUserById(Long id) {
        User user = userRepository.findById(id);
        return toDto(user);
    }

    private UserDto toDto(User user) {
        UserDto dto = new UserDto();
        return dto;
    }
}
"""

# Same as `_USER_SERVICE_JAVA` plus a brand new `deleteUser` method that calls
# a *pre-existing but previously uncalled* `UserRepository.deleteById` method
# -- used by the incremental-reindex test (only the Service file is modified
# on disk and re-indexed; `UserRepository.java`, which already declares
# `deleteById`, is never touched).
_USER_SERVICE_JAVA_WITH_DELETE = """\
package com.example.app;

import org.springframework.stereotype.Service;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public UserDto findUserById(Long id) {
        User user = userRepository.findById(id);
        return toDto(user);
    }

    public void deleteUser(Long id) {
        userRepository.deleteById(id);
    }

    private UserDto toDto(User user) {
        UserDto dto = new UserDto();
        return dto;
    }
}
"""

_USER_REPOSITORY_JAVA = """\
package com.example.app;

import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository {

    User findById(Long id);

    void save(User user);

    void deleteById(Long id);
}
"""

_USER_ENTITY_JAVA = """\
package com.example.app;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;

@Entity
public class User {

    @Id
    private Long id;

    private String name;

    private String email;
}
"""

_USER_DTO_JAVA = """\
package com.example.app;

public class UserDto {

    private Long id;
    private String name;
    private String email;
}
"""

_USER_SERVICE_TEST_JAVA = """\
package com.example.app;

import org.junit.jupiter.api.Test;

public class UserServiceTest {

    @Test
    public void testFindUserById() {
        UserService userService = new UserService(null);
        UserDto result = userService.findUserById(1L);
    }
}
"""

# Repo-relative (posix) paths shared by every test in this file.
CONTROLLER_PATH = "src/main/java/com/example/app/UserController.java"
SERVICE_PATH = "src/main/java/com/example/app/UserService.java"
REPOSITORY_PATH = "src/main/java/com/example/app/UserRepository.java"
ENTITY_PATH = "src/main/java/com/example/app/User.java"
DTO_PATH = "src/main/java/com/example/app/UserDto.java"
SERVICE_TEST_PATH = "src/test/java/com/example/app/UserServiceTest.java"


@pytest.fixture
def deep_spring_boot_repo(tmp_path) -> Path:
    """A richer fake Spring Boot Maven project than the shared
    `spring_boot_repo` fixture: Controller -> Service -> Repository, plus an
    Entity, a DTO, and one JUnit test file.

    Layout::

        pom.xml
        src/main/java/com/example/app/
            UserController.java   (@RestController /api/users, GET /{id};
                                    constructor-injects UserService)
            UserService.java      (@Service; constructor-injects
                                    UserRepository; findUserById() calls
                                    userRepository.findById())
            UserRepository.java   (@Repository interface; findById/save/
                                    deleteById -- deleteById has no caller
                                    until the incremental-update test adds
                                    one)
            User.java             (@Entity; id/name/email fields)
            UserDto.java           (plain class, no stereotype)
        src/test/java/com/example/app/
            UserServiceTest.java  (JUnit test referencing UserService)
    """
    root = tmp_path / "deep_spring_boot_repo"
    root.mkdir()

    (root / "pom.xml").write_text(_POM_XML, encoding="utf-8")

    main_java = root / "src" / "main" / "java" / "com" / "example" / "app"
    main_java.mkdir(parents=True)
    (main_java / "UserController.java").write_text(_USER_CONTROLLER_JAVA, encoding="utf-8")
    (main_java / "UserService.java").write_text(_USER_SERVICE_JAVA, encoding="utf-8")
    (main_java / "UserRepository.java").write_text(_USER_REPOSITORY_JAVA, encoding="utf-8")
    (main_java / "User.java").write_text(_USER_ENTITY_JAVA, encoding="utf-8")
    (main_java / "UserDto.java").write_text(_USER_DTO_JAVA, encoding="utf-8")

    test_java = root / "src" / "test" / "java" / "com" / "example" / "app"
    test_java.mkdir(parents=True)
    (test_java / "UserServiceTest.java").write_text(_USER_SERVICE_TEST_JAVA, encoding="utf-8")

    return root


# ---------------------------------------------------------------------------
# Shared helpers (not fixtures -- individual tests need different combinations
# of "just build in memory" vs. "build and persist to .kenzo/").
# ---------------------------------------------------------------------------


def _build_graphs(repo_root: Path) -> tuple[SymbolIndex, ReferenceGraph, DependencyGraph]:
    symbol_index = SymbolIndex(str(repo_root))
    symbol_index.build()
    reference_graph = ReferenceGraph(symbol_index)
    reference_graph.build()
    dependency_graph = DependencyGraph(symbol_index)
    dependency_graph.build()
    return symbol_index, reference_graph, dependency_graph


def _persist_graphs(repo_root: Path) -> tuple[SymbolIndex, ReferenceGraph, DependencyGraph]:
    symbol_index, reference_graph, dependency_graph = _build_graphs(repo_root)
    memory = RepositoryMemory(str(repo_root))
    symbol_index.save(str(memory.code_index_path))
    reference_graph.save(str(memory.references_path))
    dependency_graph.save(str(memory.dependencies_path))
    return symbol_index, reference_graph, dependency_graph


def _class_by_name(symbol_index: SymbolIndex, name: str):
    matches = symbol_index.find_class(name)
    assert len(matches) == 1, f"expected exactly one class named {name!r}, got {matches}"
    return matches[0][1]


# ---------------------------------------------------------------------------
# 1. Symbol extraction
# ---------------------------------------------------------------------------


def test_symbol_index_finds_all_expected_classes_and_files(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    stats = symbol_index.stats()
    assert stats["file_count"] == 6
    assert stats["total_files_found"] == 6
    assert stats["truncated"] is False
    assert stats["class_count"] == 6
    assert stats["by_language"] == {"java": 6}

    all_names = {cls.name for _path, cls in symbol_index.all_classes()}
    assert all_names == {
        "UserController",
        "UserService",
        "UserRepository",
        "User",
        "UserDto",
        "UserServiceTest",
    }


def test_symbol_index_assigns_correct_stereotypes(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    assert _class_by_name(symbol_index, "UserController").stereotype == "controller"
    assert _class_by_name(symbol_index, "UserService").stereotype == "service"
    assert _class_by_name(symbol_index, "UserRepository").stereotype == "repository"
    assert _class_by_name(symbol_index, "User").stereotype == "entity"
    assert _class_by_name(symbol_index, "UserDto").stereotype is None
    assert _class_by_name(symbol_index, "UserServiceTest").stereotype is None

    assert _class_by_name(symbol_index, "UserRepository").kind == "interface"
    assert _class_by_name(symbol_index, "UserController").kind == "class"


def test_symbol_index_entity_fields(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    user = _class_by_name(symbol_index, "User")
    fields_by_name = {f.name: f for f in user.fields}
    assert set(fields_by_name) == {"id", "name", "email"}
    assert fields_by_name["id"].type == "Long"
    assert fields_by_name["name"].type == "String"
    assert fields_by_name["email"].type == "String"
    assert "Id" in fields_by_name["id"].annotations


def test_symbol_index_controller_endpoint_composition(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    controller = _class_by_name(symbol_index, "UserController")
    get_user = next(m for m in controller.methods if m.name == "getUser")

    assert get_user.endpoint == "/api/users/{id}"
    assert controller.endpoints == ["/api/users/{id}"]


def test_symbol_index_constructor_injection_dependencies(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    controller = _class_by_name(symbol_index, "UserController")
    service = _class_by_name(symbol_index, "UserService")

    assert controller.dependencies == ["UserService"]
    assert service.dependencies == ["UserRepository"]


# ---------------------------------------------------------------------------
# 2. Reference resolution
# ---------------------------------------------------------------------------


def test_reference_graph_resolves_controller_to_service_to_repository_chain(deep_spring_boot_repo):
    _symbol_index, reference_graph, _dependency_graph = _build_graphs(deep_spring_boot_repo)

    service_callers = reference_graph.find_callers("UserService.findUserById")
    assert len(service_callers) == 1
    assert service_callers[0]["caller"] == "UserController.getUser"
    assert service_callers[0]["callee"] == "UserService.findUserById"

    repository_callers = reference_graph.find_callers("UserRepository.findById")
    assert len(repository_callers) == 1
    assert repository_callers[0]["caller"] == "UserService.findUserById"
    assert repository_callers[0]["callee"] == "UserRepository.findById"

    # Bare-name lookup should find the same edge via `find_references`.
    bare_refs = reference_graph.find_references("findUserById")
    assert any(edge["caller"] == "UserController.getUser" for edge in bare_refs)


def test_reference_graph_find_usages_of_service_in_controller(deep_spring_boot_repo):
    _symbol_index, reference_graph, _dependency_graph = _build_graphs(deep_spring_boot_repo)

    usages = reference_graph.find_usages("UserService")
    usage_kinds = {u["usage_kind"] for u in usages}

    assert {"dependency", "field_type", "parameter_type"}.issubset(usage_kinds)
    assert all(u["used_in_class"] == "UserController" for u in usages)


# ---------------------------------------------------------------------------
# 3. Flow tracing (kenzo.tools.code_intelligence.trace_flow)
# ---------------------------------------------------------------------------


def _assert_three_hop_chain(chain: list[dict]) -> None:
    assert len(chain) == 3

    assert chain[0]["role"] == "controller"
    assert chain[0]["class"] == "UserController"
    assert chain[0]["method"] == "getUser"

    assert chain[1]["role"] == "service"
    assert chain[1]["class"] == "UserService"
    assert chain[1]["method"] == "findUserById"

    assert chain[2]["role"] == "repository"
    assert chain[2]["class"] == "UserRepository"
    assert chain[2]["method"] == "findById"


def test_trace_flow_via_http_endpoint(deep_spring_boot_repo):
    _persist_graphs(deep_spring_boot_repo)

    result = trace_flow("/api/users/{id}", repo_root=str(deep_spring_boot_repo))

    assert result.success is True
    assert result.data["is_endpoint"] is True
    _assert_three_hop_chain(result.data["chain"])


def test_trace_flow_via_qualified_method(deep_spring_boot_repo):
    _persist_graphs(deep_spring_boot_repo)

    result = trace_flow("UserController.getUser", repo_root=str(deep_spring_boot_repo))

    assert result.success is True
    _assert_three_hop_chain(result.data["chain"])


def test_trace_flow_unknown_target_fails_cleanly(deep_spring_boot_repo):
    _persist_graphs(deep_spring_boot_repo)

    result = trace_flow("TotallyUnknownClass.doesNotExist", repo_root=str(deep_spring_boot_repo))

    assert result.success is False
    assert result.error


# ---------------------------------------------------------------------------
# 4. Architecture explanation (DependencyGraph)
# ---------------------------------------------------------------------------


def test_explain_architecture_buckets_classes_by_stereotype(deep_spring_boot_repo):
    _symbol_index, _reference_graph, dependency_graph = _build_graphs(deep_spring_boot_repo)

    result = dependency_graph.explain_architecture()
    modules = result["modules"]

    assert "app" in modules
    buckets = modules["app"]

    assert set(buckets["controllers"]) == {"UserController"}
    assert set(buckets["services"]) == {"UserService"}
    assert set(buckets["repositories"]) == {"UserRepository"}
    assert set(buckets["entities"]) == {"User"}
    assert set(buckets["components"]) == set()
    assert set(buckets["configurations"]) == set()
    # No-stereotype classes (the DTO and the JUnit test class) land in "other".
    assert set(buckets["other"]) == {"UserDto", "UserServiceTest"}


# ---------------------------------------------------------------------------
# 5. Test discovery
# ---------------------------------------------------------------------------


def test_find_related_tests_finds_the_junit_test_for_the_service(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    tests = find_related_tests(symbol_index, SERVICE_PATH)

    assert tests == [SERVICE_TEST_PATH]


def test_find_related_tests_returns_empty_for_file_with_no_test(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    tests = find_related_tests(symbol_index, REPOSITORY_PATH)

    assert tests == []


# ---------------------------------------------------------------------------
# 6. Incremental update correctness (SymbolIndex.reindex_file)
# ---------------------------------------------------------------------------


def test_reindex_file_is_isolated_and_picks_up_new_call_edge_end_to_end(deep_spring_boot_repo):
    symbol_index = SymbolIndex(str(deep_spring_boot_repo))
    symbol_index.build()

    # Baseline: nobody calls UserRepository.deleteById yet.
    baseline_reference_graph = ReferenceGraph(symbol_index)
    baseline_reference_graph.build()
    assert baseline_reference_graph.find_callers("UserRepository.deleteById") == []

    entity_node_id_before = id(symbol_index.files[ENTITY_PATH])
    service_node_id_before = id(symbol_index.files[SERVICE_PATH])

    # Modify ONLY the Service file on disk (add deleteUser(), which calls the
    # pre-existing-but-previously-uncalled UserRepository.deleteById()).
    service_abs_path = deep_spring_boot_repo / SERVICE_PATH
    service_abs_path.write_text(_USER_SERVICE_JAVA_WITH_DELETE, encoding="utf-8")

    symbol_index.reindex_file(SERVICE_PATH)

    # (a) An unrelated file's FileNode object identity must be UNCHANGED --
    # proves reindex_file() did not re-walk/re-parse the whole repo.
    assert id(symbol_index.files[ENTITY_PATH]) == entity_node_id_before

    # (b) The Service's FileNode object itself WAS replaced, and now reflects
    # the new method.
    assert id(symbol_index.files[SERVICE_PATH]) != service_node_id_before
    updated_service = _class_by_name(symbol_index, "UserService")
    method_names = {m.name for m in updated_service.methods}
    assert "deleteUser" in method_names

    # Rebuild ReferenceGraph/DependencyGraph fresh on top of the mutated
    # SymbolIndex and confirm the new edge is now discoverable end-to-end.
    reference_graph = ReferenceGraph(symbol_index)
    reference_graph.build()
    dependency_graph = DependencyGraph(symbol_index)
    dependency_graph.build()

    new_callers = reference_graph.find_callers("UserRepository.deleteById")
    assert len(new_callers) == 1
    assert new_callers[0]["caller"] == "UserService.deleteUser"
    assert new_callers[0]["callee"] == "UserRepository.deleteById"

    # The pre-existing controller->service->repository chain must still work.
    assert len(reference_graph.find_callers("UserService.findUserById")) == 1


# ---------------------------------------------------------------------------
# 7. kenzo.tools.code_intelligence tool-layer: real persist-then-fresh-load
# ---------------------------------------------------------------------------


def test_find_callers_tool_reads_persisted_index(deep_spring_boot_repo):
    _persist_graphs(deep_spring_boot_repo)

    result = find_callers("UserRepository.findById", repo_root=str(deep_spring_boot_repo))

    assert result.success is True
    callers = result.data["callers"]
    assert len(callers) == 1
    assert callers[0]["caller"] == "UserService.findUserById"


def test_explain_architecture_tool_reads_persisted_index(deep_spring_boot_repo):
    _persist_graphs(deep_spring_boot_repo)

    result = explain_architecture(repo_root=str(deep_spring_boot_repo))

    assert result.success is True
    assert set(result.data["modules"]["app"]["controllers"]) == {"UserController"}
    assert set(result.data["modules"]["app"]["services"]) == {"UserService"}


def test_find_related_tests_tool_reads_persisted_index(deep_spring_boot_repo):
    _persist_graphs(deep_spring_boot_repo)

    result = find_related_tests_tool(SERVICE_PATH, repo_root=str(deep_spring_boot_repo))

    assert result.success is True
    assert result.data["tests"] == [SERVICE_TEST_PATH]


def test_trace_flow_tool_reads_persisted_index(deep_spring_boot_repo):
    _persist_graphs(deep_spring_boot_repo)

    result = trace_flow("UserController.getUser", repo_root=str(deep_spring_boot_repo))

    assert result.success is True
    _assert_three_hop_chain(result.data["chain"])


# ---------------------------------------------------------------------------
# 8. kenzo.tools.code_intelligence tool-layer: graceful "nothing indexed yet"
# ---------------------------------------------------------------------------


def test_trace_flow_without_index_returns_clear_error(tmp_path):
    unindexed_repo = tmp_path / "unindexed_repo"
    unindexed_repo.mkdir()

    result = trace_flow("/api/users/{id}", repo_root=str(unindexed_repo))

    assert result.success is False
    assert result.error


def test_find_callers_without_index_returns_clear_error(tmp_path):
    unindexed_repo = tmp_path / "unindexed_repo"
    unindexed_repo.mkdir()

    result = find_callers("UserService.findUserById", repo_root=str(unindexed_repo))

    assert result.success is False
    assert result.error


def test_explain_architecture_without_index_returns_clear_error(tmp_path):
    unindexed_repo = tmp_path / "unindexed_repo"
    unindexed_repo.mkdir()

    result = explain_architecture(repo_root=str(unindexed_repo))

    assert result.success is False
    assert result.error


def test_find_related_tests_tool_without_index_returns_clear_error(tmp_path):
    unindexed_repo = tmp_path / "unindexed_repo"
    unindexed_repo.mkdir()

    result = find_related_tests_tool("src/main/java/com/example/app/UserService.java", repo_root=str(unindexed_repo))

    assert result.success is False
    assert result.error
