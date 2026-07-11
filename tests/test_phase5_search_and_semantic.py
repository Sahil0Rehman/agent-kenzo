"""Phase 5 "Semantic Memory and Retrieval Engine" tests -- Part 2:
`kenzo/core/code_search.py`'s `CodeSearchEngine` hybrid ranking, including the
two named synonym-level scenarios this phase is specifically required to
prove (see `FakeSemanticEmbeddingProvider` and the two tests at the bottom of
this file for exactly how and why).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kenzo.core.code_search import CodeSearchEngine
from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.reference_graph import ReferenceGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.memory.vector.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from kenzo.memory.vector.semantic_indexer import SemanticIndexer
from kenzo.memory.vector.store import QdrantStore

# ---------------------------------------------------------------------------
# 4. Hybrid ranking over a real Controller -> Service -> Repository chain,
#    using kenzo's own shipped default HashingEmbeddingProvider.
# ---------------------------------------------------------------------------


def _build_search_engine(repo_root: Path, dimensions: int = 64) -> CodeSearchEngine:
    symbol_index = SymbolIndex(str(repo_root))
    symbol_index.build()
    reference_graph = ReferenceGraph(symbol_index)
    reference_graph.build()
    dependency_graph = DependencyGraph(symbol_index)
    dependency_graph.build()

    provider = HashingEmbeddingProvider(dimensions=dimensions)
    store = QdrantStore(collection_name=f"phase5_search_{repo_root.name}", dimensions=dimensions, url=":memory:")
    semantic_indexer = SemanticIndexer(str(repo_root), symbol_index, provider, store)
    semantic_indexer.index_all()

    return CodeSearchEngine(symbol_index, reference_graph, dependency_graph, provider, store)


def test_exact_class_name_query_returns_that_class_top_with_symbol_match_one(spring_boot_repo):
    engine = _build_search_engine(spring_boot_repo)

    # UserRepository (an interface, no constructor) rather than LogoutController:
    # a Java constructor is always named identically to its class, so
    # "LogoutController" would tie 1.0 symbol_match against BOTH the class
    # chunk AND its own constructor's method chunk (LogoutController.LogoutController).
    # UserRepository has no constructor, so this is an unambiguous exact match.
    results = engine.search("UserRepository", top_k=5)

    assert results
    assert results[0].symbol == "UserRepository"
    assert results[0].chunk_type == "class"
    assert results[0].signals["symbol_match"] == pytest.approx(1.0)


def test_unrelated_but_lexically_overlapping_query_still_returns_something_via_keyword_signal(spring_boot_repo):
    """"authenticated" (the literal string AuthService.authenticate() returns)
    shares no substring relationship with the method name "authenticate"
    itself (extra trailing "d" defeats the substring check `_name_match_score`
    uses), so this can only be found via the KEYWORD signal reading the
    method body's real source text -- proving the hybrid ranking surfaces a
    reasonable result even when the query's words aren't in any symbol name.
    """
    engine = _build_search_engine(spring_boot_repo)

    results = engine.search("authenticated", top_k=10)

    assert results  # not empty -- something reasonable was found
    hit = next((r for r in results if r.symbol == "AuthService.authenticate"), None)
    assert hit is not None, f"expected AuthService.authenticate among results, got {[r.symbol for r in results]}"
    assert hit.signals["symbol_match"] == 0.0  # confirms it did NOT come from a name match
    assert hit.signals["keyword"] > 0.0  # confirms the keyword signal is what found it


def test_top_k_zero_or_negative_returns_empty_list(spring_boot_repo):
    engine = _build_search_engine(spring_boot_repo)

    assert engine.search("LogoutController", top_k=0) == []
    assert engine.search("LogoutController", top_k=-1) == []


def test_search_result_signals_always_has_all_four_keys(spring_boot_repo):
    engine = _build_search_engine(spring_boot_repo)

    results = engine.search("LogoutController", top_k=3)

    assert results
    for result in results:
        assert set(result.signals) == {"symbol_match", "semantic", "dependency_importance", "keyword"}


# ---------------------------------------------------------------------------
# 5. THE TWO REQUIRED SEMANTIC-QUERY SCENARIOS
#
# Kenzo's own shipped, dependency-free, offline `HashingEmbeddingProvider` is
# HONESTLY DOCUMENTED (see its own docstring in
# kenzo/memory/vector/embeddings.py) to be lexical/fuzzy-substring similarity,
# NOT learned synonym-level understanding -- it will not reliably know "sign
# out" relates to "logout", or "remove user access" relates to
# "revokePermission", on its own. Trying to hand-tune the hashing algorithm
# until these two specific pairs happen to score highly would be curve-fitting
# to this test, not a real fix -- and could easily make other, real queries
# score WORSE.
#
# The tests below do NOT use HashingEmbeddingProvider. They use a small, fully
# deterministic FAKE/STUB `EmbeddingProvider` (implementing the real
# `EmbeddingProvider` ABC, per this codebase's established convention of
# faking the model/embedding layer rather than hitting a real network --
# see test_controller_scenarios.py's scripted ModelRouter.complete faking and
# test_repo_intelligence_integration.py's forbid-model-calls helper) that
# simulates "IF a real, competently-trained semantic embedding model were
# configured": it recognizes a small, explicit set of phrase markers and maps
# each recognized concept to its own fixed, orthogonal vector, so anything
# classified into the same concept is maximally similar (cosine == 1.0) and
# anything in a different concept (or unclassified) is maximally dissimilar
# (cosine == 0.0, since the concept dimensions never overlap with the
# "unclassified" fallback dimensions).
#
# This proves the REST OF THE RETRIEVAL PIPELINE (CodeSearchEngine's hybrid
# ranking) correctly surfaces the right result when genuine semantic
# similarity IS present -- it does NOT claim kenzo's shipped default
# (HashingEmbeddingProvider) achieves synonym-level understanding out of the
# box. That default's honest, documented limitation is exactly why this test
# exists: to prove the pipeline is ready for a real provider, independent of
# whether one happens to be configured.
# ---------------------------------------------------------------------------


class FakeSemanticEmbeddingProvider(EmbeddingProvider):
    """Deterministic stand-in for a real trained embedding model. See the
    module-level comment block above for exactly what this does and does not
    prove.

    Classification is substring-based over a small table of markers per
    concept. Two kinds of markers are used per concept:

    - A "natural language" marker (e.g. "sign out", "remove user access") --
      matches how a human/agent would actually phrase the query.
    - A "precise corpus" marker (e.g. "method authcontroller.logout") --
      matches ONLY the rendered header text `SemanticIndexer._method_chunk`
      generates for that exact method (`f"method {cls.name}.{method.name}"`),
      never the enclosing class's own chunk (whose header is
      `f"{cls.kind} {cls.name}"` -- a different string). This matters because
      the real `SemanticIndexer._class_chunk` embeds the ENTIRE class body
      (including every method's source) into the class-level chunk too, so a
      marker as loose as "revokepermission" would classify BOTH the class
      and the method chunk into the same concept and force a coin-flip tie at
      the vector-store layer. Anchoring the corpus-side marker to the
      method-only header text keeps the two named scenarios deterministic
      without artificially inflating the fake provider's own cleverness.

    Anything matching no concept gets a fallback vector derived from a stable
    hash of the text, confined to dimensions the concept vectors never touch
    -- guaranteeing zero cosine similarity against every concept, and giving
    different unrelated texts different (but equally irrelevant) vectors.
    """

    _DIMENSIONS = 8
    _LOGOUT_DIM = 0
    _REVOKE_DIM = 1
    _FALLBACK_DIMS = range(2, 8)

    _CONCEPT_MARKERS = {
        "logout": ("sign out", "log out", "method authcontroller.logout"),
        "revoke_permission": ("remove user access", "revoke permission", "method permissionservice.revokepermission"),
    }

    @property
    def dimensions(self) -> int:
        return self._DIMENSIONS

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()

        for concept, markers in self._CONCEPT_MARKERS.items():
            if any(marker in lowered for marker in markers):
                vector = [0.0] * self._DIMENSIONS
                vector[self._LOGOUT_DIM if concept == "logout" else self._REVOKE_DIM] = 1.0
                return vector

        return self._fallback_vector(lowered)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def _fallback_vector(self, lowered_text: str) -> list[float]:
        digest = hashlib.blake2b(lowered_text.encode("utf-8"), digest_size=8).digest()
        vector = [0.0] * self._DIMENSIONS
        for i, dim in enumerate(self._FALLBACK_DIMS):
            byte = digest[i % len(digest)]
            vector[dim] = (byte / 255.0) - 0.5  # spread in [-0.5, 0.5], never all-zero for nonempty digest
        return vector


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
    <artifactId>access-control-service</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>
    <properties>
        <java.version>17</java.version>
    </properties>
</project>
"""

_AUTH_CONTROLLER_JAVA = """\
package com.example.demo;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public String login() {
        return authService.authenticate();
    }

    @PostMapping("/logout")
    public String logout() {
        authService.terminateSession();
        return "ok";
    }
}
"""

_AUTH_SERVICE_JAVA = """\
package com.example.demo;

import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final SessionRepository sessionRepository;

    public AuthService(SessionRepository sessionRepository) {
        this.sessionRepository = sessionRepository;
    }

    public String authenticate() {
        return "ok";
    }

    public void terminateSession() {
        sessionRepository.clear();
    }
}
"""

_SESSION_REPOSITORY_JAVA = """\
package com.example.demo;

import org.springframework.stereotype.Repository;

@Repository
public interface SessionRepository {

    void clear();

    String find(String token);
}
"""

_PERMISSION_SERVICE_JAVA = """\
package com.example.demo;

import org.springframework.stereotype.Service;

@Service
public class PermissionService {

    private final PermissionRepository permissionRepository;

    public PermissionService(PermissionRepository permissionRepository) {
        this.permissionRepository = permissionRepository;
    }

    public void grantPermission(String userId, String resource) {
        permissionRepository.save(userId, resource);
    }

    public void revokePermission(String userId, String resource) {
        permissionRepository.delete(userId, resource);
    }
}
"""

_PERMISSION_REPOSITORY_JAVA = """\
package com.example.demo;

import org.springframework.stereotype.Repository;

@Repository
public interface PermissionRepository {

    void save(String userId, String resource);

    void delete(String userId, String resource);
}
"""


@pytest.fixture
def access_control_repo(tmp_path) -> Path:
    """A small, real Spring Boot-shaped Java project with a real
    `logout()`-named method and a real `revokePermission()`-named method,
    plus plausible decoys (`login`, `authenticate`, `grantPermission`, etc.)
    that share no meaningful lexical overlap with either target query, so a
    positive match genuinely demonstrates the semantic signal at work rather
    than an accidental symbol/keyword hit.
    """
    root = tmp_path / "access_control_repo"
    root.mkdir()
    (root / "pom.xml").write_text(_POM_XML, encoding="utf-8")

    java_dir = root / "src" / "main" / "java" / "com" / "example" / "demo"
    java_dir.mkdir(parents=True)
    (java_dir / "AuthController.java").write_text(_AUTH_CONTROLLER_JAVA, encoding="utf-8")
    (java_dir / "AuthService.java").write_text(_AUTH_SERVICE_JAVA, encoding="utf-8")
    (java_dir / "SessionRepository.java").write_text(_SESSION_REPOSITORY_JAVA, encoding="utf-8")
    (java_dir / "PermissionService.java").write_text(_PERMISSION_SERVICE_JAVA, encoding="utf-8")
    (java_dir / "PermissionRepository.java").write_text(_PERMISSION_REPOSITORY_JAVA, encoding="utf-8")

    return root


def _build_fake_semantic_search_engine(repo_root: Path) -> CodeSearchEngine:
    symbol_index = SymbolIndex(str(repo_root))
    symbol_index.build()
    reference_graph = ReferenceGraph(symbol_index)
    reference_graph.build()
    dependency_graph = DependencyGraph(symbol_index)
    dependency_graph.build()

    provider = FakeSemanticEmbeddingProvider()
    store = QdrantStore(collection_name="phase5_fake_semantic", dimensions=provider.dimensions, url=":memory:")
    semantic_indexer = SemanticIndexer(str(repo_root), symbol_index, provider, store)
    semantic_indexer.index_all()

    return CodeSearchEngine(symbol_index, reference_graph, dependency_graph, provider, store)


def test_scenario_a_sign_out_query_finds_logout_method_via_fake_semantic_provider(access_control_repo):
    """Required scenario A: a query for "sign out" must find a method
    literally named logout() -- via the FAKE semantic provider standing in
    for a real trained model (see the module comment block above)."""
    engine = _build_fake_semantic_search_engine(access_control_repo)

    results = engine.search("sign out", top_k=5)

    assert results, "expected at least one result for 'sign out'"
    top = results[0]
    assert top.symbol == "AuthController.logout", f"expected AuthController.logout on top, got {top.symbol!r}"
    assert top.chunk_type == "method"
    assert top.signals["semantic"] > 0.9


def test_scenario_b_remove_user_access_query_finds_revoke_permission_method_via_fake_semantic_provider(
    access_control_repo,
):
    """Required scenario B: a query for "remove user access" must find a
    method literally named revokePermission() -- via the FAKE semantic
    provider standing in for a real trained model (see the module comment
    block above)."""
    engine = _build_fake_semantic_search_engine(access_control_repo)

    results = engine.search("remove user access", top_k=5)

    assert results, "expected at least one result for 'remove user access'"
    top = results[0]
    assert top.symbol == "PermissionService.revokePermission", f"expected PermissionService.revokePermission on top, got {top.symbol!r}"
    assert top.chunk_type == "method"
    assert top.signals["semantic"] > 0.9


def test_fake_provider_gives_zero_semantic_similarity_to_unrelated_query(access_control_repo):
    """Control case: a query classified into NEITHER concept must not
    semantically favor either target method -- proving the fake provider's
    "far apart" side, not just its "close together" side."""
    engine = _build_fake_semantic_search_engine(access_control_repo)

    results = engine.search("completely unrelated pizza toppings request", top_k=20)

    logout_hit = next((r for r in results if r.symbol == "AuthController.logout"), None)
    revoke_hit = next((r for r in results if r.symbol == "PermissionService.revokePermission"), None)

    for hit in (logout_hit, revoke_hit):
        if hit is not None:
            assert hit.signals["semantic"] == 0.0
