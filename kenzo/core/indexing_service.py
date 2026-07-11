"""Background indexing: keeps `AgentRuntime` construction fast even on a
large, never-before-indexed repository by loading whatever `.kenzo/` already
has persisted (instant) and running the full rebuild -- symbol index,
reference/dependency graphs, and the Phase 5 semantic index -- on a
background thread, never blocking the caller.

Thread-safety model: `IndexingService` exposes `symbol_index`/
`reference_graph`/`dependency_graph`/`semantic_indexer` as plain instance
attributes. The background thread builds entirely new objects and only
reassigns these attributes (under `self._lock`) once every step has
succeeded -- a single attribute reassignment is GIL-atomic, so a reader that
doesn't take the lock (e.g. a tool doing a one-off read) can never observe a
half-built object, only "the old one" or "the new one." Callers that mutate
(`reindex_file`) take the same lock so they never race the background
thread's own swap. A failed background build leaves whatever was already
loaded (or an empty index, on a repo indexed for the first time) in place --
it never crashes the process or leaves `AgentRuntime` unusable.
"""

from __future__ import annotations

import threading
from typing import Optional

from kenzo.config.settings import KenzoConfig
from kenzo.intelligence.dependency_graph import DependencyGraph
from kenzo.intelligence.reference_graph import ReferenceGraph
from kenzo.intelligence.symbol_index import SymbolIndex
from kenzo.memory.repository import RepositoryMemory
from kenzo.memory.vector.embeddings import EmbeddingRouter
from kenzo.memory.vector.semantic_indexer import SemanticIndexer
from kenzo.memory.vector.store import QdrantStore, VectorStore


class IndexingService:
    """Owns the Phase 4 + Phase 5 index state for one repository, refreshing
    it on a background thread so `AgentRuntime.__init__` never blocks on a
    full repo walk/parse/embed pass.
    """

    def __init__(self, repo_root: str, memory: RepositoryMemory, config: KenzoConfig) -> None:
        self.repo_root = repo_root
        self.memory = memory
        self.config = config

        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

        self.embedding_router = EmbeddingRouter(config.embedding)
        self.vector_store: VectorStore = QdrantStore(
            collection_name=memory.vector_collection_name,
            dimensions=self.embedding_router.dimensions,
            url=config.vector_store.url,
        )

        # Placeholders until _load_persisted()/the background build populate
        # them for real -- an empty SymbolIndex is a valid, usable (just
        # empty) state, never None, so every caller can rely on these
        # attributes always being the real objects, not Optional.
        self.symbol_index = SymbolIndex(repo_root)
        self.reference_graph = ReferenceGraph(self.symbol_index)
        self.dependency_graph = DependencyGraph(self.symbol_index)
        self.semantic_indexer = SemanticIndexer(
            repo_root, self.symbol_index, self.embedding_router, self.vector_store
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, background: bool = True) -> None:
        """Load whatever's already persisted (fast, synchronous), then
        refresh it. `background=False` runs the refresh synchronously
        instead -- used by `kenzo index` and tests that need a deterministic,
        fully-indexed result before proceeding, rather than a racy background
        thread.
        """
        self._load_persisted()

        if not background:
            self._build()
            return

        self._thread = threading.Thread(target=self._build, daemon=True)
        self._thread.start()

    def _load_persisted(self) -> None:
        loaded_index = SymbolIndex.load(self.repo_root, str(self.memory.code_index_path))
        if loaded_index is None:
            return  # nothing persisted yet -- the empty placeholders from __init__ stand

        loaded_refs = ReferenceGraph.load(loaded_index, str(self.memory.references_path))
        loaded_deps = DependencyGraph.load(loaded_index, str(self.memory.dependencies_path))

        semantic_indexer = SemanticIndexer(self.repo_root, loaded_index, self.embedding_router, self.vector_store)
        semantic_indexer.load_bookkeeping(str(self.memory.semantic_chunk_ids_path))

        with self._lock:
            self.symbol_index = loaded_index
            self.reference_graph = loaded_refs if loaded_refs is not None else ReferenceGraph(loaded_index)
            self.dependency_graph = loaded_deps if loaded_deps is not None else DependencyGraph(loaded_index)
            self.semantic_indexer = semantic_indexer

    def _build(self) -> None:
        try:
            fresh_index = SymbolIndex(self.repo_root)
            fresh_index.build(
                ignored_dirs=self.config.indexing.ignored_dirs,
                max_files=self.config.indexing.max_indexed_files,
            )

            fresh_refs = ReferenceGraph(fresh_index)
            fresh_refs.build()

            fresh_deps = DependencyGraph(fresh_index)
            fresh_deps.build()

            fresh_index.save(str(self.memory.code_index_path))
            fresh_refs.save(str(self.memory.references_path))
            fresh_deps.save(str(self.memory.dependencies_path))

            fresh_semantic = SemanticIndexer(self.repo_root, fresh_index, self.embedding_router, self.vector_store)
            fresh_semantic.load_bookkeeping(str(self.memory.semantic_chunk_ids_path))
            fresh_semantic.index_all()
            fresh_semantic.save_bookkeeping(str(self.memory.semantic_chunk_ids_path))

            with self._lock:
                self.symbol_index = fresh_index
                self.reference_graph = fresh_refs
                self.dependency_graph = fresh_deps
                self.semantic_indexer = fresh_semantic
            self._last_error = None
        except Exception as exc:  # noqa: BLE001 - a background build failure must never crash the process
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready_event.set()

    def is_ready(self) -> bool:
        """True once the background (or synchronous) build has finished --
        successfully or not. Before this, callers still get a usable (if
        possibly stale or empty) index via the attributes above; this is for
        callers that specifically want to know "has the fresh build landed
        yet," e.g. a status line or a test."""
        return self._ready_event.is_set()

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the build finishes (or `timeout` elapses). Returns
        whether it's ready. Used by `kenzo index`/tests that need a
        deterministic end state, not by the interactive agent loop."""
        return self._ready_event.wait(timeout)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    # ------------------------------------------------------------------
    # Incremental update (after a successful write_file/edit_file)
    # ------------------------------------------------------------------

    def reindex_file(self, rel_path: str) -> None:
        """Incrementally refresh exactly `rel_path` across the symbol index,
        reference/dependency graphs, and the semantic index, then persist.
        Always operates on `self.symbol_index` etc. as they are AT CALL TIME
        (under the lock) -- never a reference captured earlier -- so this
        stays correct even if called while a background full rebuild is
        still in flight (see module docstring).
        """
        with self._lock:
            self.symbol_index.reindex_file(rel_path)
            self.reference_graph.build()
            self.dependency_graph.build()
            self.symbol_index.save(str(self.memory.code_index_path))
            self.reference_graph.save(str(self.memory.references_path))
            self.dependency_graph.save(str(self.memory.dependencies_path))

            self.semantic_indexer.index_file(rel_path)
            self.semantic_indexer.save_bookkeeping(str(self.memory.semantic_chunk_ids_path))
