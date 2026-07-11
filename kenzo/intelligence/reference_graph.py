"""Cross-file reference resolution: turns the naive, per-file `CallSite`s
recorded by `parser.py` (raw dotted text like "authService.terminateSession",
attached to `FunctionSymbol.calls`) into real caller/callee edges between
actual classes and methods across the whole indexed repository.

This is a *useful approximation* over a real compiler/type-checker, not a
resolved reference the way an IDE's "find usages" would give you: it only
ever "knows" about a variable's type if that type came from a constructor
parameter or a field declaration recorded by the parser, and ambiguous
lookups (more than one class sharing a simple name) intentionally fan out to
every candidate rather than arbitrarily picking one. See `ReferenceGraph.build`
for the exact algorithm and its documented, deliberate limitations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from kenzo.intelligence.models import ClassSymbol, FunctionSymbol
from kenzo.intelligence.symbol_index import SymbolIndex


class ReferenceGraph:
    """Resolves naive `CallSite`s into caller/callee edges and tracks where
    every class is *used* (extended, implemented, injected, referenced as a
    field/parameter type, or called into), keyed by "symbol key" -- either
    `"ClassName.method_name"` for a method or the bare function name for a
    top-level function.
    """

    def __init__(self, symbol_index: SymbolIndex) -> None:
        self.symbol_index = symbol_index
        # callee_key -> list of caller edge dicts (who calls INTO this symbol)
        self.callers: dict[str, list[dict[str, Any]]] = {}
        # caller_key -> list of callee edge dicts (what this symbol calls OUT to)
        self.callees: dict[str, list[dict[str, Any]]] = {}
        # class_name.lower() -> list of usage-site dicts
        self.class_usages: dict[str, list[dict[str, Any]]] = {}

    # ----------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------

    def build(self) -> None:
        """(Re)populate `self.callers`/`self.callees`/`self.class_usages`
        from the current state of `self.symbol_index`.

        Algorithm:
          1. For every class, build a per-class "variable name -> type name"
             map from its fields (`FieldSymbol.name -> FieldSymbol.type`)
             union its constructor(s) parameters (`param.name -> param.type`).
             Fields take priority on a name collision (constructor params are
             applied first, then fields overwrite) -- an arbitrary but
             consistent tie-break, since in practice a field and a
             same-named constructor parameter almost always describe the
             same dependency anyway.
          2. Record class-usage sites from `extends`/`implements`/
             `dependencies`/field types/parameter types up front (independent
             of call resolution).
          3. For every method of every class, and every top-level function,
             resolve each `CallSite` in `.calls` against that function's
             variable->type map (empty for top-level functions -- they have
             no enclosing class, so by construction none of their calls can
             resolve via this algorithm; see module docstring). A resolved
             call target class also gets a "call_target" usage-site entry.
        """
        self.callers = {}
        self.callees = {}
        self.class_usages = {}

        all_classes = self.symbol_index.all_classes()

        # --- Step 2: non-call-derived usage sites -----------------------
        for file_path, cls in all_classes:
            if cls.extends:
                self._record_usage(cls.extends, file_path, cls.name, "extends")
            for iface in cls.implements:
                self._record_usage(iface, file_path, cls.name, "implements")
            for dep in cls.dependencies:
                self._record_usage(dep, file_path, cls.name, "dependency")
            for field in cls.fields:
                if field.type:
                    self._record_usage(field.type, file_path, cls.name, "field_type")
            for method in cls.methods:
                for param in method.parameters:
                    if param.type:
                        self._record_usage(param.type, file_path, cls.name, "parameter_type")

        for file_path, fn in self.symbol_index.all_functions():
            for param in fn.parameters:
                if param.type:
                    self._record_usage(param.type, file_path, None, "parameter_type")

        # --- Step 1 + 3: per-class var->type map, then resolve calls ----
        for file_path, cls in all_classes:
            var_type_map = self._build_var_type_map(cls)
            for method in cls.methods:
                self._resolve_calls(method, file_path, cls.name, var_type_map)

        for file_path, fn in self.symbol_index.all_functions():
            # Top-level functions have no enclosing class -- no variable->
            # type map is derivable for them under this algorithm, so every
            # one of their CallSites is (correctly, honestly) unresolvable
            # here and will simply be skipped by `_resolve_calls`.
            self._resolve_calls(fn, file_path, None, {})

    def _build_var_type_map(self, cls: ClassSymbol) -> dict[str, str]:
        var_map: dict[str, str] = {}
        for method in cls.methods:
            if method.is_constructor:
                for param in method.parameters:
                    if param.name:
                        var_map[param.name] = param.type
        for field in cls.fields:
            if field.name:
                var_map[field.name] = field.type  # fields win on collision
        return var_map

    def _resolve_calls(
        self,
        fn: FunctionSymbol,
        file_path: str,
        used_in_class: Optional[str],
        var_type_map: dict[str, str],
    ) -> None:
        caller_key = fn.qualified_name if fn.qualified_name else fn.name

        for call in fn.calls:
            target = call.target
            if "." not in target:
                # No resolvable object -- parsers only emit these off known
                # dependency variables in the first place, but be defensive.
                continue

            var_part, method_part = target.rsplit(".", 1)
            candidate_type = var_type_map.get(var_part)
            if not candidate_type:
                # var_part is not a known dependency variable of the
                # enclosing class (or there is no enclosing class at all) --
                # do not fabricate an edge.
                continue

            candidates = self.symbol_index.find_class(candidate_type)
            for target_file, target_cls in candidates:
                # Every resolved candidate target class counts as a usage,
                # even if no specific method on it ends up matching below.
                self._record_usage(target_cls.name, file_path, used_in_class, "call_target")

                matched_method: Optional[FunctionSymbol] = None
                for m in target_cls.methods:
                    if m.name.lower() == method_part.lower():
                        matched_method = m
                        break
                if matched_method is None:
                    continue

                callee_key = f"{target_cls.name}.{matched_method.name}"
                edge = {
                    "caller": caller_key,
                    "caller_file": file_path,
                    "callee": callee_key,
                    "callee_file": target_file,
                    "location": call.location.model_dump(),
                }
                self.callees.setdefault(caller_key, []).append(edge)
                self.callers.setdefault(callee_key, []).append(edge)

    def _record_usage(
        self,
        class_name: str,
        used_in_file: str,
        used_in_class: Optional[str],
        usage_kind: str,
    ) -> None:
        if not class_name:
            return
        self.class_usages.setdefault(class_name.lower(), []).append(
            {
                "used_in_file": used_in_file,
                "used_in_class": used_in_class,
                "usage_kind": usage_kind,
            }
        )

    # ----------------------------------------------------------------
    # Lookups
    # ----------------------------------------------------------------

    def find_references(self, symbol_name: str) -> list[dict[str, Any]]:
        """`symbol_name` may be bare ("terminateSession") or qualified
        ("AuthService.terminateSession"). Qualified names are looked up as
        an exact key in `self.callers`. Bare names are matched against every
        key's method-name part (after the last "."), case-insensitively,
        and the union of all matching keys' edges is returned.
        """
        if "." in symbol_name:
            return list(self.callers.get(symbol_name, []))
        return self._find_by_bare_name(symbol_name)

    def find_callers(self, qualified_method_name: str) -> list[dict[str, Any]]:
        """Alias of `find_references` with one difference: a dotted input is
        matched case-insensitively against the FULL key (not stored via a
        plain dict `.get`), since callers of this method tend to pass a
        qualified name whose casing they aren't 100% sure of. A bare
        (non-dotted) input falls back to the exact same search
        `find_references` performs.
        """
        if "." in qualified_method_name:
            target = qualified_method_name.lower()
            results: list[dict[str, Any]] = []
            for key, edges in self.callers.items():
                if key.lower() == target:
                    results.extend(edges)
            return results
        return self._find_by_bare_name(qualified_method_name)

    def _find_by_bare_name(self, name: str) -> list[dict[str, Any]]:
        target = name.lower()
        results: list[dict[str, Any]] = []
        for key, edges in self.callers.items():
            method_part = key.rsplit(".", 1)[-1] if "." in key else key
            if method_part.lower() == target:
                results.extend(edges)
        return results

    def find_usages(self, class_name: str) -> list[dict[str, Any]]:
        return list(self.class_usages.get(class_name.lower(), []))

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "callers": self.callers,
            "callees": self.callees,
            "class_usages": self.class_usages,
        }

    @classmethod
    def from_dict(cls, symbol_index: SymbolIndex, data: dict[str, Any]) -> "ReferenceGraph":
        graph = cls(symbol_index)
        graph.callers = {k: list(v) for k, v in (data.get("callers") or {}).items()}
        graph.callees = {k: list(v) for k, v in (data.get("callees") or {}).items()}
        graph.class_usages = {k: list(v) for k, v in (data.get("class_usages") or {}).items()}
        return graph

    def save(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, symbol_index: SymbolIndex, path: str) -> Optional["ReferenceGraph"]:
        in_path = Path(path)
        if not in_path.exists():
            return None
        try:
            data = json.loads(in_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return cls.from_dict(symbol_index, data)
        except Exception:  # noqa: BLE001 - malformed persisted data should degrade to None, not crash the caller
            return None
