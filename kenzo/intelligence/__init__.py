"""Deep code intelligence: tree-sitter-backed AST parsing, a cross-language
symbol index, a reference graph (definitions/callers/callees), and a
dependency graph (Controller -> Service -> Repository, imports, packages).

This package is additive to everything built in Phases 1-3: the existing
regex-based `kenzo.memory.java_analyzer` and the Phase 2/3 tools
(`find_symbol`, `map_flow`, `.kenzo/symbols.json`) are untouched and keep
working exactly as before. This package's output backs a new, separate set
of tools (`kenzo.tools.code_intelligence`) built on genuinely parsed ASTs
instead of regex/keyword guessing.
"""
