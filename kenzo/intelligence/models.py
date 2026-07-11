"""Shared data contracts for the code intelligence layer.

Every parser (Java/Python/JavaScript/TypeScript) produces the SAME
`FileNode` shape regardless of source language, so `symbol_index.py`,
`reference_graph.py`, and `dependency_graph.py` can work uniformly across
languages instead of each needing language-specific branches.

Kept dependency-free of the rest of `kenzo.intelligence` (mirrors the
`kenzo.core.schemas` convention) so every other module here can import it
without cycles.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    """1-indexed source location, half-open on lines like most editors."""

    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0


class CallSite(BaseModel):
    """A naive call detected inside a function/method body: `target` is a
    best-effort dotted name (e.g. "authService.terminateSession" or
    "self.repository.save") -- not a resolved reference. Resolution into a
    real caller/callee edge happens in `reference_graph.py`, which has
    visibility across the whole repo, not just one file.
    """

    target: str
    location: Location


class ParameterSymbol(BaseModel):
    name: str
    type: str = ""


class FieldSymbol(BaseModel):
    name: str
    type: str = ""
    annotations: list[str] = Field(default_factory=list)
    location: Location


class FunctionSymbol(BaseModel):
    """A top-level function OR a class method -- `is_method`/`is_constructor`
    distinguish them; `qualified_name` is "ClassName.method_name" for a
    method, or just the bare name for a top-level function.
    """

    name: str
    qualified_name: str = ""
    parameters: list[ParameterSymbol] = Field(default_factory=list)
    return_type: str = ""
    annotations: list[str] = Field(default_factory=list)
    endpoint: Optional[str] = None  # Spring-mapping-derived HTTP path, if any
    calls: list[CallSite] = Field(default_factory=list)
    location: Location
    is_method: bool = False
    is_constructor: bool = False


class ClassSymbol(BaseModel):
    """A class, interface, enum, or record (Java) / class (Python, JS, TS)."""

    name: str
    kind: str = "class"  # "class" | "interface" | "enum" | "record"
    qualified_name: str = ""  # package/module-qualified, e.g. "com.example.auth.LogoutController"
    extends: Optional[str] = None
    implements: list[str] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    stereotype: Optional[str] = None  # controller/service/repository/entity/component/configuration
    fields: list[FieldSymbol] = Field(default_factory=list)
    methods: list[FunctionSymbol] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # DI-derived type names
    endpoints: list[str] = Field(default_factory=list)
    location: Location


class ImportSymbol(BaseModel):
    """One import/require statement. `module` is the raw imported path/
    package (e.g. "org.springframework.web.bind.annotation.RestController",
    "os.path", or "react"); `imported_names` covers `from x import a, b` /
    `import {a, b} from 'x'` style partial imports.
    """

    module: str
    alias: Optional[str] = None
    imported_names: list[str] = Field(default_factory=list)


class FileNode(BaseModel):
    """The canonical, language-agnostic parse result for one source file."""

    path: str  # relative to repo_root, POSIX form
    language: str  # "java" | "python" | "javascript" | "typescript"
    package: str = ""  # Java package / Python module dotted path / "" for JS/TS
    classes: list[ClassSymbol] = Field(default_factory=list)
    functions: list[FunctionSymbol] = Field(default_factory=list)  # top-level only
    imports: list[ImportSymbol] = Field(default_factory=list)
    exports: list[str] = Field(default_factory=list)  # JS/TS export names
    annotations: list[str] = Field(default_factory=list)  # file/module-level, rare
    parse_errors: list[str] = Field(default_factory=list)
    fallback_used: bool = False  # True if tree-sitter failed and a regex fallback was used
