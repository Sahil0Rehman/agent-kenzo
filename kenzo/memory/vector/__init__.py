"""Phase 5 semantic memory: embeddings, a vector store abstraction, and the
`CodeChunk` contract that flows between them.

Additive to Phase 4's `kenzo.intelligence` package -- `SemanticIndexer` (see
`kenzo.memory.vector.semantic_indexer`) builds `CodeChunk`s FROM an already-
built `SymbolIndex`/`ReferenceGraph` rather than re-parsing anything, so this
package depends on Phase 4 but Phase 4 has no knowledge of this one.
"""
