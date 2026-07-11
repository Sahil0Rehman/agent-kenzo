"""Tree-sitter-backed AST parsing: turns a source file into a `FileNode`.

Java is fully implemented here on real tree-sitter ASTs (see `_parse_java`).
Python/JavaScript/TypeScript follow the identical `FileNode` contract --
see the `_parse_python`/`_parse_javascript`/`_parse_typescript` functions.

Every parse is defensive: a language whose grammar isn't installed, a file
tree-sitter can't parse cleanly (`has_error`), or any unexpected exception
during extraction all degrade to a best-effort `FileNode` rather than
raising -- for Java specifically, that fallback is the existing regex-based
`kenzo.memory.java_analyzer` (Phase 2), so this layer is strictly additive:
nothing that worked before stops working if tree-sitter is unavailable or
chokes on a given file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from kenzo.intelligence.models import (
    CallSite,
    ClassSymbol,
    FieldSymbol,
    FileNode,
    FunctionSymbol,
    ImportSymbol,
    Location,
    ParameterSymbol,
)
from kenzo.memory.java_analyzer import _analyze_java_file as _regex_analyze_java_file

SUPPORTED_LANGUAGES = ("java", "python", "javascript", "typescript")

_EXTENSION_LANGUAGE = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def language_for_extension(extension: str) -> Optional[str]:
    return _EXTENSION_LANGUAGE.get(extension.lower())


# --------------------------------------------------------------------------
# Lazy tree-sitter Language/Parser construction -- importing the language
# packages at module import time would make ALL of kenzo fail to import if
# tree-sitter or a grammar package were ever missing/broken in some
# environment; instead, failures are contained to whichever parse actually
# needs that language, and degrade to the regex fallback (Java) or an
# empty-but-valid FileNode (Python/JS/TS) instead of crashing the caller.
# --------------------------------------------------------------------------

_LANGUAGE_CACHE: dict[str, Any] = {}
_PARSER_CACHE: dict[str, Any] = {}


def _get_language(language: str) -> Any:
    if language in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[language]

    from tree_sitter import Language

    if language == "java":
        import tree_sitter_java as ts_lang

        lang = Language(ts_lang.language())
    elif language == "python":
        import tree_sitter_python as ts_lang

        lang = Language(ts_lang.language())
    elif language == "javascript":
        import tree_sitter_javascript as ts_lang

        lang = Language(ts_lang.language())
    elif language == "typescript":
        import tree_sitter_typescript as ts_lang

        lang = Language(ts_lang.language_typescript())
    else:
        raise ValueError(f"unsupported language: {language}")

    _LANGUAGE_CACHE[language] = lang
    return lang


def _get_parser(language: str) -> Any:
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    from tree_sitter import Parser

    parser = Parser(_get_language(language))
    _PARSER_CACHE[language] = parser
    return parser


# --------------------------------------------------------------------------
# Small shared AST helpers
# --------------------------------------------------------------------------


def _text(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _location(node: Any) -> Location:
    return Location(
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        start_col=node.start_point[1],
        end_col=node.end_point[1],
    )


def _first_child_of_type(node: Any, types: tuple[str, ...]) -> Optional[Any]:
    for child in node.children:
        if child.type in types:
            return child
    return None


# --------------------------------------------------------------------------
# Java
# --------------------------------------------------------------------------

_JAVA_TYPE_DECLARATIONS = ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration")
_JAVA_KIND_BY_NODE_TYPE = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
}

_JAVA_MAPPING_ANNOTATIONS = {"GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "RequestMapping"}

_JAVA_STEREOTYPE_PRIORITY: list[tuple[set[str], str]] = [
    ({"RestController", "Controller"}, "controller"),
    ({"Service"}, "service"),
    ({"Repository"}, "repository"),
    ({"Entity"}, "entity"),
    ({"Configuration"}, "configuration"),
    ({"Component"}, "component"),
]


def _java_annotations_from_modifiers(source: bytes, node: Any) -> list[tuple[str, Optional[str]]]:
    """Every Java declaration (class/interface/enum/record/field/method/
    constructor) that can carry annotations has a `modifiers` child
    containing `marker_annotation` (no args, e.g. @Autowired) and
    `annotation` (with args, e.g. @RequestMapping("/x")) nodes -- this is a
    real AST relationship, not a line-based heuristic like the regex
    analyzer's lookback. Returns [(name, first_string_literal_arg_or_None)].
    """
    modifiers = node.child_by_field_name("modifiers") or _first_child_of_type(node, ("modifiers",))
    if modifiers is None:
        return []

    results: list[tuple[str, Optional[str]]] = []
    for child in modifiers.children:
        if child.type not in ("marker_annotation", "annotation"):
            continue
        name_node = child.child_by_field_name("name")
        name = _text(source, name_node) if name_node is not None else ""
        first_arg: Optional[str] = None
        args_node = child.child_by_field_name("arguments")
        if args_node is not None:
            first_arg = _first_string_literal(source, args_node)
        results.append((name, first_arg))
    return results


def _first_string_literal(source: bytes, node: Any) -> Optional[str]:
    if node.type == "string_literal":
        raw = _text(source, node)
        return raw[1:-1] if len(raw) >= 2 else raw
    for child in node.children:
        found = _first_string_literal(source, child)
        if found is not None:
            return found
    return None


def _java_endpoint(annotations: list[tuple[str, Optional[str]]], base_path: str) -> Optional[str]:
    for name, arg in annotations:
        if name in _JAVA_MAPPING_ANNOTATIONS:
            sub_path = arg or ""
            if base_path and sub_path:
                if base_path.endswith("/") and sub_path.startswith("/"):
                    return base_path + sub_path[1:]
                return base_path + sub_path
            return base_path or sub_path or "/"
    return None


def _java_stereotype(annotation_names: list[str]) -> Optional[str]:
    names = set(annotation_names)
    for keys, stereotype in _JAVA_STEREOTYPE_PRIORITY:
        if names & keys:
            return stereotype
    return None


def _java_calls(source: bytes, body_node: Any, known_deps: dict[str, str]) -> list[CallSite]:
    """Naive call detection: for every `method_invocation` in `body_node`
    whose `object` resolves to a known dependency variable (by exact name,
    or by the last dotted segment for a `field_access` like `this.repo`),
    record a CallSite. Not a resolved reference -- `reference_graph.py`
    does real cross-file resolution using this as raw material.
    """
    calls: list[CallSite] = []

    def walk(node: Any) -> None:
        if node.type == "method_invocation":
            object_node = node.child_by_field_name("object")
            name_node = node.child_by_field_name("name")
            if object_node is not None and name_node is not None:
                object_text = _text(source, object_node)
                var_name = object_text.rsplit(".", 1)[-1]
                if var_name in known_deps:
                    method_name = _text(source, name_node)
                    calls.append(CallSite(target=f"{var_name}.{method_name}", location=_location(node)))
        for child in node.children:
            walk(child)

    walk(body_node)
    return calls


def _java_field(source: bytes, node: Any) -> Optional[FieldSymbol]:
    type_node = node.child_by_field_name("type")
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return None
    name_node = declarator.child_by_field_name("name")
    if name_node is None:
        return None
    annotations = _java_annotations_from_modifiers(source, node)
    return FieldSymbol(
        name=_text(source, name_node),
        type=_text(source, type_node) if type_node is not None else "",
        annotations=[n for n, _ in annotations],
        location=_location(node),
    )


def _java_parameters(source: bytes, params_node: Any) -> list[ParameterSymbol]:
    params: list[ParameterSymbol] = []
    for child in params_node.children:
        if child.type not in ("formal_parameter", "spread_parameter"):
            continue
        type_node = child.child_by_field_name("type")
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        params.append(
            ParameterSymbol(
                name=_text(source, name_node),
                type=_text(source, type_node) if type_node is not None else "",
            )
        )
    return params


def _java_class(source: bytes, node: Any, package: str) -> ClassSymbol:
    kind = _JAVA_KIND_BY_NODE_TYPE.get(node.type, "class")
    name_node = node.child_by_field_name("name")
    name = _text(source, name_node) if name_node is not None else "<anonymous>"

    class_annotations = _java_annotations_from_modifiers(source, node)
    annotation_names = [n for n, _ in class_annotations]
    base_path = ""
    for ann_name, arg in class_annotations:
        if ann_name == "RequestMapping" and arg:
            base_path = arg
            break

    extends_: Optional[str] = None
    superclass = node.child_by_field_name("superclass")
    if superclass is not None:
        type_node = _first_child_of_type(superclass, ("type_identifier", "generic_type"))
        extends_ = _text(source, type_node) if type_node is not None else None

    implements_: list[str] = []
    interfaces = node.child_by_field_name("interfaces")
    if interfaces is not None:
        type_list = _first_child_of_type(interfaces, ("type_list",))
        if type_list is not None:
            implements_ = [
                _text(source, c) for c in type_list.children if c.type in ("type_identifier", "generic_type")
            ]

    fields: list[FieldSymbol] = []
    methods: list[FunctionSymbol] = []
    known_deps: dict[str, str] = {}

    body = node.child_by_field_name("body")
    if body is not None:
        # Pass 1: fields (+ constructor parameters) establish known DI
        # dependency variable names before we scan method bodies for calls.
        for child in body.children:
            if child.type == "field_declaration":
                field = _java_field(source, child)
                if field is not None:
                    fields.append(field)
                    if "Autowired" in field.annotations:
                        known_deps[field.name] = field.type
            elif child.type == "constructor_declaration":
                params_node = child.child_by_field_name("parameters")
                if params_node is not None:
                    for param in _java_parameters(source, params_node):
                        known_deps.setdefault(param.name, param.type)

        for child in body.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                is_ctor = child.type == "constructor_declaration"
                method_name_node = child.child_by_field_name("name")
                method_name = _text(source, method_name_node) if method_name_node is not None else ""
                method_annotations = _java_annotations_from_modifiers(source, child)
                method_annotation_names = [n for n, _ in method_annotations]
                params_node = child.child_by_field_name("parameters")
                parameters = _java_parameters(source, params_node) if params_node is not None else []
                return_type_node = child.child_by_field_name("type")
                body_node = child.child_by_field_name("body")
                calls = _java_calls(source, body_node, known_deps) if body_node is not None else []
                endpoint = None if is_ctor else _java_endpoint(method_annotations, base_path)

                methods.append(
                    FunctionSymbol(
                        name=method_name,
                        qualified_name=f"{name}.{method_name}",
                        parameters=parameters,
                        return_type=_text(source, return_type_node) if return_type_node is not None else "",
                        annotations=method_annotation_names,
                        endpoint=endpoint,
                        calls=calls,
                        location=_location(child),
                        is_method=not is_ctor,
                        is_constructor=is_ctor,
                    )
                )

    endpoints: list[str] = []
    seen_endpoints: set[str] = set()
    for method in methods:
        if method.endpoint and method.endpoint not in seen_endpoints:
            endpoints.append(method.endpoint)
            seen_endpoints.add(method.endpoint)
    if not endpoints and base_path:
        endpoints.append(base_path)

    dependencies: list[str] = []
    seen_types: set[str] = set()
    for dep_type in known_deps.values():
        if dep_type not in seen_types:
            dependencies.append(dep_type)
            seen_types.add(dep_type)

    qualified_name = f"{package}.{name}" if package else name

    return ClassSymbol(
        name=name,
        kind=kind,
        qualified_name=qualified_name,
        extends=extends_,
        implements=implements_,
        annotations=annotation_names,
        stereotype=_java_stereotype(annotation_names),
        fields=fields,
        methods=methods,
        dependencies=dependencies,
        endpoints=endpoints,
        location=_location(node),
    )


def _parse_java(path: str, source: bytes) -> FileNode:
    parser = _get_parser("java")
    tree = parser.parse(source)
    root = tree.root_node

    package = ""
    imports: list[ImportSymbol] = []
    classes: list[ClassSymbol] = []
    parse_errors: list[str] = []

    for child in root.children:
        if child.type == "package_declaration":
            ident = _first_child_of_type(child, ("scoped_identifier", "identifier"))
            package = _text(source, ident) if ident is not None else ""
        elif child.type == "import_declaration":
            ident = _first_child_of_type(child, ("scoped_identifier", "identifier"))
            if ident is not None:
                imports.append(ImportSymbol(module=_text(source, ident)))
        elif child.type in _JAVA_TYPE_DECLARATIONS:
            try:
                classes.append(_java_class(source, child, package))
            except Exception as exc:  # noqa: BLE001 - one malformed declaration must not sink the file
                parse_errors.append(f"failed to extract {child.type} at line {child.start_point[0] + 1}: {exc}")

    if root.has_error:
        parse_errors.append("tree-sitter reported a syntax error somewhere in this file (partial results returned)")

    return FileNode(
        path=path,
        language="java",
        package=package,
        classes=classes,
        imports=imports,
        parse_errors=parse_errors,
    )


def _java_fallback(path: str, abs_path: Path) -> FileNode:
    """Regex-based fallback (Phase 2's `kenzo.memory.java_analyzer`), used
    when tree-sitter itself can't be loaded/used for Java at all. A
    per-declaration syntax error from tree-sitter is NOT routed here --
    tree-sitter still returns a real (partial) tree for those; this
    fallback is for when tree-sitter/the Java grammar is unusable outright.
    """
    old_symbols = _regex_analyze_java_file(abs_path, path)
    classes = [
        ClassSymbol(
            name=s.get("name", ""),
            kind=s.get("kind", "class"),
            qualified_name=(f"{s['package']}.{s['name']}" if s.get("package") else s.get("name", "")),
            extends=s.get("extends"),
            implements=s.get("implements") or [],
            annotations=s.get("annotations") or [],
            stereotype=s.get("type"),
            fields=[
                FieldSymbol(
                    name=f.get("name", ""),
                    type=f.get("type", ""),
                    annotations=f.get("annotations") or [],
                    location=Location(start_line=0, end_line=0),
                )
                for f in (s.get("fields") or [])
            ],
            methods=[
                FunctionSymbol(
                    name=m.get("name", ""),
                    qualified_name=f"{s.get('name', '')}.{m.get('name', '')}",
                    annotations=m.get("annotations") or [],
                    endpoint=m.get("endpoint"),
                    calls=[
                        CallSite(target=f"{c.get('target_field', '')}.{c.get('method', '')}", location=Location(start_line=0, end_line=0))
                        for c in (m.get("calls") or [])
                    ],
                    location=Location(start_line=0, end_line=0),
                )
                for m in (s.get("methods") or [])
            ],
            dependencies=s.get("dependencies") or [],
            endpoints=s.get("endpoints") or [],
            location=Location(start_line=0, end_line=0),
        )
        for s in old_symbols
    ]
    package = old_symbols[0].get("package", "") if old_symbols else ""
    return FileNode(path=path, language="java", package=package, classes=classes, fallback_used=True)


# --------------------------------------------------------------------------
# Python / JavaScript / TypeScript
#
# Implemented in the same style as `_parse_java` above: lazy Language/Parser
# construction via `_get_language`/`_get_parser`, `child_by_field_name`-based
# structural extraction (not regex), assembled into the same `FileNode`
# contract. See each function's docstring for the specific grammar node
# names it relies on.
# --------------------------------------------------------------------------


_PYTHON_ROUTE_KEYWORDS = ("route", "get", "post", "put", "delete")


def _python_first_string_literal(source: bytes, node: Any) -> Optional[str]:
    """Mirrors Java's `_first_string_literal`, adapted to tree-sitter-python's
    `string` node (which wraps `string_start`/`string_content`/`string_end`
    instead of being a single leaf token). Depth-first, so for a call like
    `app.route("/logout", methods=["POST"])` this returns the first
    *positional* string ("/logout") before ever reaching the keyword
    argument's list contents -- verified against a real parse.
    """
    if node.type == "string":
        content = _first_child_of_type(node, ("string_content",))
        return _text(source, content) if content is not None else ""
    for child in node.children:
        found = _python_first_string_literal(source, child)
        if found is not None:
            return found
    return None


def _python_render_decorators(source: bytes, decorator_nodes: list[Any]) -> list[tuple[str, Optional[str]]]:
    """Each `decorator` node is `@` (anonymous/unnamed token) + exactly one
    named expression child (`identifier`, `attribute`, or `call` -- verified
    against a real parse). Python decorators are arbitrary expressions, not
    a fixed vocabulary like Java annotations, so we render the expression's
    full source text verbatim (e.g. "app.route(\"/logout\")" or
    "staticmethod") rather than trying to normalize it. Returns
    [(rendered_text, first_positional_string_arg_or_None)], parallel to
    Java's `_java_annotations_from_modifiers`.
    """
    results: list[tuple[str, Optional[str]]] = []
    for dec in decorator_nodes:
        expr = None
        for child in dec.children:
            if child.is_named:
                expr = child
                break
        if expr is None:
            continue
        text = _text(source, expr)
        first_arg = _python_first_string_literal(source, expr)
        results.append((text, first_arg))
    return results


def _python_endpoint(decorators: list[tuple[str, Optional[str]]]) -> Optional[str]:
    """Best-effort Flask/FastAPI-style endpoint heuristic (NOT a real
    framework model): if a decorator's rendered text contains one of
    route/get/post/put/delete (case-insensitive) and it carries a string
    literal argument, treat that string as the endpoint path -- loosely
    mirroring Java's Spring-mapping-annotation-to-endpoint pattern, since
    decorators commonly look like `@app.route("/logout")` or
    `@router.post("/logout")`. Deliberately does not attempt to combine a
    class-level base path with a method-level sub-path the way Java's
    RequestMapping+GetMapping combination does -- Flask/FastAPI class-based
    views don't have a single standard convention for that, so we keep this
    to a single, per-decorator match rather than over-engineering it.
    """
    for text, first_arg in decorators:
        if first_arg is not None and any(kw in text.lower() for kw in _PYTHON_ROUTE_KEYWORDS):
            return first_arg
    return None


def _python_parameters(source: bytes, params_node: Any) -> list[ParameterSymbol]:
    """`parameters` node's children are, per a real parse: bare `identifier`
    (untyped param), `typed_parameter` (`identifier` + `type` field),
    `default_parameter` (`name`/`value` fields, no type),
    `typed_default_parameter` (`name`/`type`/`value` fields), and
    `list_splat_pattern`/`dictionary_splat_pattern` (`*args`/`**kwargs`,
    rendered with their star(s) kept in the name). Punctuation and the
    positional-only `/` / keyword-only `*` separators are plain unnamed
    tokens that match none of these `if` branches and are simply skipped.
    """
    params: list[ParameterSymbol] = []
    for child in params_node.children:
        if child.type == "identifier":
            params.append(ParameterSymbol(name=_text(source, child), type=""))
        elif child.type == "typed_parameter":
            name_node = _first_child_of_type(child, ("identifier",))
            type_node = child.child_by_field_name("type")
            if name_node is not None:
                params.append(
                    ParameterSymbol(
                        name=_text(source, name_node),
                        type=_text(source, type_node) if type_node is not None else "",
                    )
                )
        elif child.type == "default_parameter":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                params.append(ParameterSymbol(name=_text(source, name_node), type=""))
        elif child.type == "typed_default_parameter":
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            if name_node is not None:
                params.append(
                    ParameterSymbol(
                        name=_text(source, name_node),
                        type=_text(source, type_node) if type_node is not None else "",
                    )
                )
        elif child.type in ("list_splat_pattern", "dictionary_splat_pattern"):
            params.append(ParameterSymbol(name=_text(source, child), type=""))
    return params


def _python_calls(source: bytes, body_node: Any, known_names: set[str]) -> list[CallSite]:
    """Naive call detection mirroring Java's `_java_calls`: for every `call`
    node whose `function` is an `attribute` access, resolve the attribute's
    `object` to a variable name -- if `object` is itself a bare `identifier`
    (e.g. `requests` in `requests.get(...)`) use it exactly, otherwise (a
    chained attribute like `self.repository` in `self.repository.delete(...)`)
    take its last dotted segment, the same way Java's version simplifies
    `this.repo` down to `repo`. If that name is a known dependency (for
    methods: a `self.x = ...` attribute discovered in `__init__`; for
    top-level functions: a module-level imported name), record a CallSite
    like "repository.save" (leading "self."/module chain dropped).
    """
    calls: list[CallSite] = []

    def walk(node: Any) -> None:
        if node.type == "call":
            func = node.child_by_field_name("function")
            if func is not None and func.type == "attribute":
                obj = func.child_by_field_name("object")
                name_node = func.child_by_field_name("attribute")
                if obj is not None and name_node is not None:
                    obj_text = _text(source, obj)
                    var_name = obj_text if obj.type == "identifier" else obj_text.rsplit(".", 1)[-1]
                    if var_name in known_names:
                        method_name = _text(source, name_node)
                        calls.append(CallSite(target=f"{var_name}.{method_name}", location=_location(node)))
        for child in node.children:
            walk(child)

    walk(body_node)
    return calls


def _python_dependencies(
    source: bytes, init_body: Any, init_params: list[ParameterSymbol]
) -> dict[str, str]:
    """Scans `__init__`'s top-level statements for `self.x = x` (x is an
    `__init__` parameter with a type annotation -> dependency type is that
    annotation) or `self.x = SomeType(...)` (direct instantiation -> the
    called name is the dependency type), e.g. `self.repository = repository`
    and `self.cache = Cache()`. Deliberately does not attempt to unwrap
    richer right-hand sides (e.g. `logger or logging.getLogger(...)`) --
    this is a useful approximation of Java's @Autowired-field-driven
    `known_deps`, not a full expression evaluator.
    """
    param_types = {p.name: p.type for p in init_params if p.type}
    dependencies: dict[str, str] = {}

    for stmt in init_body.children:
        if stmt.type != "expression_statement":
            continue
        assign = _first_child_of_type(stmt, ("assignment",))
        if assign is None:
            continue
        left = assign.child_by_field_name("left")
        right = assign.child_by_field_name("right")
        if left is None or right is None or left.type != "attribute":
            continue
        obj = left.child_by_field_name("object")
        attr = left.child_by_field_name("attribute")
        if obj is None or attr is None or _text(source, obj) != "self":
            continue
        attr_name = _text(source, attr)

        if right.type == "identifier":
            rhs_name = _text(source, right)
            if rhs_name in param_types:
                dependencies[attr_name] = param_types[rhs_name]
        elif right.type == "call":
            func = right.child_by_field_name("function")
            if func is not None and func.type == "identifier":
                dependencies[attr_name] = _text(source, func)

    return dependencies


def _python_function(
    source: bytes,
    node: Any,
    decorators: list[tuple[str, Optional[str]]],
    class_name: Optional[str],
    known_names: set[str],
    location_node: Optional[Any] = None,
) -> FunctionSymbol:
    """Builds a FunctionSymbol for either a top-level function
    (`class_name=None`) or a method (`class_name` set). `location_node`
    lets callers pass the outer `decorated_definition` span (so a
    decorator's source lines are included in `location`) while `node`
    itself stays the `function_definition`.
    """
    name_node = node.child_by_field_name("name")
    name = _text(source, name_node) if name_node is not None else ""

    params_node = node.child_by_field_name("parameters")
    parameters = _python_parameters(source, params_node) if params_node is not None else []
    is_method = class_name is not None
    # self/cls is an implicit receiver, not a "real" parameter -- dropped
    # consistently for every method (matched by name, not just position, so
    # a @staticmethod's real first parameter isn't accidentally discarded).
    if is_method and parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]

    return_type_node = node.child_by_field_name("return_type")
    body_node = node.child_by_field_name("body")
    calls = _python_calls(source, body_node, known_names) if body_node is not None else []

    annotation_names = [text for text, _ in decorators]
    endpoint = _python_endpoint(decorators)
    qualified_name = f"{class_name}.{name}" if class_name else name
    loc_node = location_node if location_node is not None else node

    return FunctionSymbol(
        name=name,
        qualified_name=qualified_name,
        parameters=parameters,
        return_type=_text(source, return_type_node) if return_type_node is not None else "",
        annotations=annotation_names,
        endpoint=endpoint,
        calls=calls,
        location=_location(loc_node),
        is_method=is_method,
        is_constructor=(is_method and name == "__init__"),
    )


def _python_class(source: bytes, node: Any, decorators: list[tuple[str, Optional[str]]], location_node: Any) -> ClassSymbol:
    name_node = node.child_by_field_name("name")
    name = _text(source, name_node) if name_node is not None else "<anonymous>"
    annotation_names = [text for text, _ in decorators]

    extends_: Optional[str] = None
    superclasses = node.child_by_field_name("superclasses")
    if superclasses is not None:
        base_node = _first_child_of_type(superclasses, ("identifier", "attribute"))
        extends_ = _text(source, base_node) if base_node is not None else None

    # Python has no `field_declaration`-equivalent AST node at class-body
    # level worth modeling separately here (unlike Java) -- attribute
    # assignments discovered while scanning `__init__` feed `dependencies`
    # instead (see below), so `fields` is intentionally left empty.
    fields: list[FieldSymbol] = []
    methods: list[FunctionSymbol] = []

    body = node.child_by_field_name("body")
    dependencies_dict: dict[str, str] = {}
    if body is not None:
        # Pass 1: locate __init__ (if any) to seed known dependency
        # attribute names/types before scanning ANY method body for calls --
        # mirrors Java's two-pass fields-then-methods approach.
        init_node = None
        for child in body.children:
            candidate = child
            if candidate.type == "decorated_definition":
                candidate = candidate.child_by_field_name("definition")
            if candidate is not None and candidate.type == "function_definition":
                candidate_name_node = candidate.child_by_field_name("name")
                if candidate_name_node is not None and _text(source, candidate_name_node) == "__init__":
                    init_node = candidate
                    break

        if init_node is not None:
            init_params_node = init_node.child_by_field_name("parameters")
            init_params = _python_parameters(source, init_params_node) if init_params_node is not None else []
            init_body = init_node.child_by_field_name("body")
            if init_body is not None:
                dependencies_dict = _python_dependencies(source, init_body, init_params)

        known_names = set(dependencies_dict.keys())

        # Pass 2: build every method (including __init__ itself), using
        # known_names for naive call detection.
        for child in body.children:
            method_node = child
            method_decorators: list[tuple[str, Optional[str]]] = []
            location_for_method = child
            if child.type == "decorated_definition":
                method_node = child.child_by_field_name("definition")
                decorator_nodes = [c for c in child.children if c.type == "decorator"]
                method_decorators = _python_render_decorators(source, decorator_nodes)
            if method_node is None or method_node.type != "function_definition":
                continue
            methods.append(
                _python_function(
                    source,
                    method_node,
                    method_decorators,
                    class_name=name,
                    known_names=known_names,
                    location_node=location_for_method,
                )
            )

    endpoints: list[str] = []
    seen_endpoints: set[str] = set()
    for method in methods:
        if method.endpoint and method.endpoint not in seen_endpoints:
            endpoints.append(method.endpoint)
            seen_endpoints.add(method.endpoint)

    dependencies = list(dict.fromkeys(dependencies_dict.values()))  # de-dup, preserve first-seen order

    return ClassSymbol(
        name=name,
        kind="class",
        qualified_name=name,  # no package concept in Python -- see FileNode.package below
        extends=extends_,
        implements=[],  # Python has no interface concept at the syntax level
        annotations=annotation_names,
        stereotype=None,  # Python v1: no framework-aware stereotype inference (see _python_endpoint)
        fields=fields,
        methods=methods,
        dependencies=dependencies,
        endpoints=endpoints,
        location=_location(location_node),
    )


def _python_imports(source: bytes, root: Any) -> tuple[list[ImportSymbol], set[str]]:
    """Handles `import_statement` (`import x.y.z`, `import x as y`, and
    comma-separated `import a, b`) and `import_from_statement` (`from x.y
    import a, b`, `from x import a as b`, and relative variants `from .
    import utils` / `from .helpers import foo as bar`). Both node types
    expose a REPEATED `name` field (verified via `children_by_field_name`
    against a real parse -- `child_by_field_name` alone would only return
    the first), each `dotted_name` or `aliased_import`.

    For `from x import a as b`: per spec, `imported_names` records the
    original name ("a"), not the alias -- ImportSymbol.alias is a single
    field meant for the `import x as y` whole-module-alias case, and a
    from-import can have several independently-aliased names, which doesn't
    fit one `alias` slot. The alias ("b") is NOT stored on the ImportSymbol;
    it's only tracked in the returned `bound_names` set (used internally by
    `_python_calls` for top-level-function call resolution), since bound
    names are exactly the names actually usable as bare identifiers in this
    module regardless of aliasing.
    """
    imports: list[ImportSymbol] = []
    bound_names: set[str] = set()

    for child in root.children:
        if child.type == "import_statement":
            for name_node in child.children_by_field_name("name"):
                if name_node.type == "dotted_name":
                    module = _text(source, name_node)
                    imports.append(ImportSymbol(module=module))
                    bound_names.add(module.split(".")[0])
                elif name_node.type == "aliased_import":
                    dotted = name_node.child_by_field_name("name")
                    alias_node = name_node.child_by_field_name("alias")
                    module = _text(source, dotted) if dotted is not None else ""
                    alias = _text(source, alias_node) if alias_node is not None else None
                    imports.append(ImportSymbol(module=module, alias=alias))
                    if alias:
                        bound_names.add(alias)
        elif child.type == "import_from_statement":
            module_name_node = child.child_by_field_name("module_name")
            module = _text(source, module_name_node) if module_name_node is not None else ""
            imported_names: list[str] = []
            for name_node in child.children_by_field_name("name"):
                if name_node.type == "dotted_name":
                    imported_name = _text(source, name_node)
                    imported_names.append(imported_name)
                    bound_names.add(imported_name)
                elif name_node.type == "aliased_import":
                    orig_node = name_node.child_by_field_name("name")
                    alias_node = name_node.child_by_field_name("alias")
                    orig = _text(source, orig_node) if orig_node is not None else ""
                    alias = _text(source, alias_node) if alias_node is not None else None
                    imported_names.append(orig)
                    bound_names.add(alias if alias else orig)
            imports.append(ImportSymbol(module=module, imported_names=imported_names))

    return imports, bound_names


def _parse_python(path: str, source: bytes) -> FileNode:
    parser = _get_parser("python")
    tree = parser.parse(source)
    root = tree.root_node

    imports, bound_names = _python_imports(source, root)
    classes: list[ClassSymbol] = []
    functions: list[FunctionSymbol] = []
    parse_errors: list[str] = []

    for child in root.children:
        node = child
        decorators: list[tuple[str, Optional[str]]] = []
        location_node = child

        if child.type == "decorated_definition":
            definition = child.child_by_field_name("definition")
            decorator_nodes = [c for c in child.children if c.type == "decorator"]
            decorators = _python_render_decorators(source, decorator_nodes)
            node = definition

        if node is None:
            continue

        try:
            if node.type == "function_definition":
                functions.append(
                    _python_function(
                        source,
                        node,
                        decorators,
                        class_name=None,
                        known_names=bound_names,
                        location_node=location_node,
                    )
                )
            elif node.type == "class_definition":
                classes.append(_python_class(source, node, decorators, location_node))
        except Exception as exc:  # noqa: BLE001 - one malformed declaration must not sink the file
            parse_errors.append(f"failed to extract {node.type} at line {node.start_point[0] + 1}: {exc}")

    if root.has_error:
        parse_errors.append("tree-sitter reported a syntax error somewhere in this file (partial results returned)")

    return FileNode(
        path=path,
        language="python",
        package="",  # no package concept in Python -- always empty for this language
        classes=classes,
        functions=functions,
        imports=imports,
        parse_errors=parse_errors,
    )


# --------------------------------------------------------------------------
# JavaScript / TypeScript shared helpers
#
# TypeScript's grammar is a superset of JavaScript's for everything needed
# here (classes, functions, imports, exports, decorators) plus optional type
# annotations, so almost all extraction logic below is shared -- `_jsts_*`
# helpers are called from both `_parse_javascript` and `_parse_typescript`
# via the common `_jsts_parse` driver, parameterized by `is_typescript`.
#
# Every node/field name referenced below was confirmed against real
# tree-sitter-javascript/tree-sitter-typescript parses of representative
# JS/TS/NestJS-shaped samples in a throwaway scratch script (not checked
# into this repo) before being relied on here, per the same methodology
# `_parse_java` above was built with.
# --------------------------------------------------------------------------

_JSTS_CLASS_DECLARATIONS = ("class_declaration", "abstract_class_declaration")

# NestJS/Angular-style HTTP-method decorators used for endpoint derivation.
_JSTS_MAPPING_DECORATORS = {"Get", "Post", "Put", "Delete", "Patch"}

# Priority-ordered like Java's `_JAVA_STEREOTYPE_PRIORITY`: a decorator name
# *containing* the keyword (not necessarily an exact match) triggers the
# stereotype, since real decorators are invoked as `@Controller(...)` etc.
_JSTS_STEREOTYPE_RULES: list[tuple[str, str]] = [
    ("Controller", "controller"),
    ("Injectable", "service"),
    ("Service", "service"),
    ("Repository", "repository"),
]


def _jsts_string_value(source: bytes, node: Any) -> Optional[str]:
    """Recursively find the first `string` node under `node` and return its
    `string_fragment` text ("" for an empty `''`/`""` literal with no
    fragment child, since tree-sitter omits that child when there's no
    content) -- mirrors Java's `_first_string_literal`.
    """
    if node.type == "string":
        for child in node.children:
            if child.type == "string_fragment":
                return _text(source, child)
        return ""
    for child in node.children:
        found = _jsts_string_value(source, child)
        if found is not None:
            return found
    return None


def _jsts_type_text(source: bytes, type_annotation_node: Any) -> str:
    """A `type_annotation` node's text is `: <type>` -- strip the leading
    colon (confirmed present as the first token child in every sample).
    """
    raw = _text(source, type_annotation_node)
    return raw[1:].strip() if raw.startswith(":") else raw.strip()


def _jsts_decorator_info(source: bytes, decorator_node: Any) -> tuple[str, Optional[str]]:
    """A `decorator` node is `@` followed by either a bare `identifier`
    (`@Injectable`, no args) or a `call_expression` (`@Controller('/auth')`).
    Returns (name, first_string_literal_arg_or_None) -- the arg mirrors
    Java's annotation-argument handling for `RequestMapping`/`GetMapping`.
    """
    target = None
    for child in decorator_node.children:
        if child.type != "@":
            target = child
            break
    if target is None:
        return "", None
    if target.type == "call_expression":
        fn_node = target.child_by_field_name("function")
        name = _text(source, fn_node) if fn_node is not None else _text(source, target)
        args_node = target.child_by_field_name("arguments")
        arg = _jsts_string_value(source, args_node) if args_node is not None else None
        return name, arg
    return _text(source, target), None


def _jsts_collect_decorators(source: bytes, node: Any) -> list[tuple[str, Optional[str]]]:
    """Decorators are direct children of `node` with type `decorator` --
    structurally siblings of the declaration they annotate, NOT nested
    inside it (confirmed: a class's decorator is a child of the enclosing
    `export_statement` when exported, or a child of the `class_declaration`
    itself when not; a method's decorator is a child of `class_body`,
    immediately preceding the `method_definition` it annotates). Callers
    are responsible for checking the right container(s) -- see `_jsts_class`
    for the class case and the `class_body` walk for the method case.
    """
    return [_jsts_decorator_info(source, c) for c in node.children if c.type == "decorator"]


def _jsts_endpoint(decorators: list[tuple[str, Optional[str]]], base_path: str) -> Optional[str]:
    for name, arg in decorators:
        if name in _JSTS_MAPPING_DECORATORS:
            sub_path = arg or ""
            if base_path and sub_path:
                if base_path.endswith("/") and sub_path.startswith("/"):
                    return base_path + sub_path[1:]
                return base_path + sub_path
            return base_path or sub_path or "/"
    return None


def _jsts_stereotype(decorator_names: list[str]) -> Optional[str]:
    for keyword, stereotype in _JSTS_STEREOTYPE_RULES:
        if any(keyword in n for n in decorator_names):
            return stereotype
    return None


def _jsts_pattern_name(source: bytes, node: Any) -> str:
    """Best-effort name for a parameter's binding pattern: plain identifiers
    and `...rest` get their real name; object/array destructuring patterns
    (`{a, b}`, `[c, d]`) don't bind a single name, so they fall back to
    their raw source text -- a reasonable, not-necessarily-exhaustive v1.
    """
    if node.type == "identifier":
        return _text(source, node)
    if node.type == "rest_pattern":
        inner = _first_child_of_type(node, ("identifier",))
        return "..." + _text(source, inner) if inner is not None else _text(source, node)
    return _text(source, node)


def _jsts_parameters(source: bytes, params_node: Any) -> list[ParameterSymbol]:
    params: list[ParameterSymbol] = []
    for child in params_node.children:
        if child.type == "identifier":
            params.append(ParameterSymbol(name=_text(source, child)))
        elif child.type in ("required_parameter", "optional_parameter"):
            # TypeScript-only shape: `pattern` (name) + optional `type`
            # (`type_annotation`); `pattern` can itself be a `rest_pattern`
            # (`...rest: string[]`), hence reusing `_jsts_pattern_name`.
            pattern_node = child.child_by_field_name("pattern")
            type_node = child.child_by_field_name("type")
            name = _jsts_pattern_name(source, pattern_node) if pattern_node is not None else ""
            type_text = _jsts_type_text(source, type_node) if type_node is not None else ""
            params.append(ParameterSymbol(name=name, type=type_text))
        elif child.type == "assignment_pattern":
            # Default value: `x = 1` -- `left` is the bound name/pattern.
            left = child.child_by_field_name("left")
            if left is not None:
                params.append(ParameterSymbol(name=_jsts_pattern_name(source, left)))
        elif child.type == "rest_pattern":
            params.append(ParameterSymbol(name=_jsts_pattern_name(source, child)))
        elif child.type in ("object_pattern", "array_pattern"):
            params.append(ParameterSymbol(name=_text(source, child)))
        # '(' ')' ',' tokens are ignored.
    return params


def _jsts_calls(source: bytes, body_node: Any, known_deps: dict[str, str], gate: bool) -> list[CallSite]:
    """Naive call detection mirroring `_java_calls`: for every
    `call_expression` whose `function` is a `member_expression`
    (`x.y(...)` or `this.x.y(...)`), record a CallSite keyed by the
    right-most object segment + method name.

    `gate` controls scope (a deliberate, documented choice -- see the
    module-level note above `_parse_javascript`/`_parse_typescript`):
      - True (TypeScript class method with constructor-injected params):
        only recorded if the right-most segment names a known dependency,
        exactly like Java.
      - False (plain JavaScript, or any function/method with no known
        constructor dependencies): gating is skipped and every `x.y(...)`/
        `this.x.y(...)` call is recorded; a bare `this.y(...)` (no field in
        between) is skipped as too uninformative to be useful.
    """
    calls: list[CallSite] = []

    def walk(node: Any) -> None:
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node is not None and fn_node.type == "member_expression":
                object_node = fn_node.child_by_field_name("object")
                property_node = fn_node.child_by_field_name("property")
                if object_node is not None and property_node is not None:
                    object_text = _text(source, object_node)
                    var_name = object_text.rsplit(".", 1)[-1]
                    method_name = _text(source, property_node)
                    if gate:
                        if var_name in known_deps:
                            calls.append(CallSite(target=f"{var_name}.{method_name}", location=_location(node)))
                    elif var_name != "this":
                        calls.append(CallSite(target=f"{var_name}.{method_name}", location=_location(node)))
        for child in node.children:
            walk(child)

    walk(body_node)
    return calls


def _jsts_function(
    source: bytes,
    node: Any,
    *,
    name_override: Optional[str] = None,
    qualified_name: str = "",
    is_method: bool = False,
    is_constructor: bool = False,
    annotations: Optional[list[str]] = None,
    endpoint: Optional[str] = None,
    known_deps: Optional[dict[str, str]] = None,
    gate_calls: bool = False,
) -> FunctionSymbol:
    """Builds a FunctionSymbol from any node with `name`/`parameters`/`body`
    fields (`function_declaration`, `method_definition`) plus optionally
    `return_type` (TypeScript only). `name_override` covers the arrow-
    function/function-expression case, where the node itself
    (`arrow_function`/`function_expression`) has no `name` field -- the name
    comes from the enclosing `variable_declarator` instead.
    """
    name_node = node.child_by_field_name("name")
    name = name_override if name_override is not None else (_text(source, name_node) if name_node is not None else "")
    params_node = node.child_by_field_name("parameters")
    parameters = _jsts_parameters(source, params_node) if params_node is not None else []
    return_type_node = node.child_by_field_name("return_type")
    return_type = _jsts_type_text(source, return_type_node) if return_type_node is not None else ""
    body_node = node.child_by_field_name("body")
    calls = _jsts_calls(source, body_node, known_deps or {}, gate_calls) if body_node is not None else []

    return FunctionSymbol(
        name=name,
        qualified_name=qualified_name or name,
        parameters=parameters,
        return_type=return_type,
        annotations=annotations or [],
        endpoint=endpoint,
        calls=calls,
        location=_location(node),
        is_method=is_method,
        is_constructor=is_constructor,
    )


def _jsts_class(source: bytes, decl_node: Any, top_node: Any, is_typescript: bool) -> ClassSymbol:
    name_node = decl_node.child_by_field_name("name")
    name = _text(source, name_node) if name_node is not None else "<anonymous>"

    # Class-level decorators live on `top_node` (the `export_statement`,
    # when exported) and/or directly on `decl_node` (when not exported);
    # these are mutually exclusive in practice, so concatenating is safe.
    class_decorators = _jsts_collect_decorators(source, top_node)
    if decl_node is not top_node:
        class_decorators += _jsts_collect_decorators(source, decl_node)
    annotation_names = [n for n, _ in class_decorators]

    base_path = ""
    for dec_name, arg in class_decorators:
        if dec_name == "Controller" and arg:
            base_path = arg
            break

    extends_: Optional[str] = None
    implements_: list[str] = []
    heritage = _first_child_of_type(decl_node, ("class_heritage",))
    if heritage is not None:
        children = heritage.children
        for i, child in enumerate(children):
            if child.type == "extends_clause":  # TypeScript form
                value_node = child.child_by_field_name("value")
                if value_node is not None:
                    extends_ = _text(source, value_node)
            elif child.type == "implements_clause":  # TypeScript only
                implements_ = [
                    _text(source, c) for c in child.children if c.type in ("type_identifier", "generic_type")
                ]
            elif child.type == "extends" and i + 1 < len(children):  # plain JS form
                nxt = children[i + 1]
                if nxt.type in ("identifier", "type_identifier", "member_expression"):
                    extends_ = _text(source, nxt)

    methods: list[FunctionSymbol] = []
    known_deps: dict[str, str] = {}

    body = decl_node.child_by_field_name("body")
    if body is not None:
        # Pass 1: constructor parameters establish known DI dependency
        # variable names (mirrors Java's field/constructor-param pass)
        # before method bodies are scanned for calls. Every constructor
        # parameter counts, typed or not -- for plain JS this just means
        # every param's `type` is "" (filtered out of `dependencies` below,
        # but still usable as a known call-target name).
        for child in body.children:
            if child.type == "method_definition":
                method_name_node = child.child_by_field_name("name")
                if method_name_node is not None and _text(source, method_name_node) == "constructor":
                    params_node = child.child_by_field_name("parameters")
                    if params_node is not None:
                        for param in _jsts_parameters(source, params_node):
                            known_deps.setdefault(param.name, param.type)

        gate_calls = is_typescript and bool(known_deps)

        # Pass 2: methods. Decorators are NOT nested inside the
        # `method_definition` they annotate -- they're preceding siblings
        # in `class_body`, so accumulate them until the next method.
        pending_decorators: list[tuple[str, Optional[str]]] = []
        for child in body.children:
            if child.type == "decorator":
                pending_decorators.append(_jsts_decorator_info(source, child))
                continue
            if child.type != "method_definition":
                continue
            method_decorators = pending_decorators
            pending_decorators = []

            method_name_node = child.child_by_field_name("name")
            method_name = _text(source, method_name_node) if method_name_node is not None else ""
            is_ctor = method_name == "constructor"
            method_annotation_names = [n for n, _ in method_decorators]
            endpoint = None if is_ctor else _jsts_endpoint(method_decorators, base_path)

            methods.append(
                _jsts_function(
                    source,
                    child,
                    qualified_name=f"{name}.{method_name}",
                    is_method=not is_ctor,
                    is_constructor=is_ctor,
                    annotations=method_annotation_names,
                    endpoint=endpoint,
                    known_deps=known_deps,
                    gate_calls=gate_calls,
                )
            )

    endpoints: list[str] = []
    seen_endpoints: set[str] = set()
    for method in methods:
        if method.endpoint and method.endpoint not in seen_endpoints:
            endpoints.append(method.endpoint)
            seen_endpoints.add(method.endpoint)
    if not endpoints and base_path:
        endpoints.append(base_path)

    # Only typed (TypeScript) dependencies are surfaced -- plain JS
    # constructor params have type "" and are filtered out here, so a
    # plain-JS class naturally ends up with an empty `dependencies` list.
    dependencies: list[str] = []
    seen_types: set[str] = set()
    for dep_type in known_deps.values():
        if dep_type and dep_type not in seen_types:
            dependencies.append(dep_type)
            seen_types.add(dep_type)

    return ClassSymbol(
        name=name,
        kind="class",
        qualified_name=name,  # no package/module-path concept for JS/TS
        extends=extends_,
        implements=implements_,
        annotations=annotation_names,
        stereotype=_jsts_stereotype(annotation_names),
        fields=[],  # not modeled for JS/TS in this v1 -- not required by the shared contract
        methods=methods,
        dependencies=dependencies,
        endpoints=endpoints,
        location=_location(decl_node),
    )


def _jsts_import(source: bytes, node: Any) -> ImportSymbol:
    """`import_statement.source` is the module string. Import shapes are
    split across `alias`/`imported_names` as a documented choice:
      - `import Foo from 'x'`           -> alias="Foo"
      - `import * as ns from 'x'`       -> alias="ns" (namespace binding)
      - `import { a, b as c } from 'x'` -> imported_names=["a", "c"] (the
        LOCAL binding name is used for aliased specifiers, since that's the
        name actually referenced in the rest of this file)
      - `import Foo, { a, b } from 'x'` -> alias="Foo", imported_names=["a","b"]
      - `import 'x'` (side-effect only) -> neither set
      - `import Foo, * as ns from 'x'` (rare default+namespace combo) ->
        alias ends up as "ns" (namespace wins over default) -- an
        acknowledged minor gap for an uncommon shape.
    """
    source_node = node.child_by_field_name("source")
    module = (_jsts_string_value(source, source_node) if source_node is not None else "") or ""

    alias: Optional[str] = None
    imported_names: list[str] = []

    clause = _first_child_of_type(node, ("import_clause",))
    if clause is not None:
        for child in clause.children:
            if child.type == "identifier":
                alias = _text(source, child)
            elif child.type == "namespace_import":
                ident = _first_child_of_type(child, ("identifier",))
                if ident is not None:
                    alias = _text(source, ident)
            elif child.type == "named_imports":
                for spec in child.children:
                    if spec.type != "import_specifier":
                        continue
                    alias_node = spec.child_by_field_name("alias")
                    name_node = spec.child_by_field_name("name")
                    target = alias_node if alias_node is not None else name_node
                    if target is not None:
                        imported_names.append(_text(source, target))

    return ImportSymbol(module=module, alias=alias, imported_names=imported_names)


def _jsts_collect_export_names(source: bytes, export_node: Any) -> list[str]:
    """Best-effort export name extraction (a reasonable, not-necessarily-
    exhaustive v1, per the shared `FileNode.exports` contract):
      - `export { a, b as c };` -> local export_specifier name, or its
        alias when present (the alias is the externally-visible name)
      - `export class X` / `export function X` / `export interface X` /
        `export type X = ...` -> the declaration's own `name` field
      - `export const a = 1, b = 2;` -> each `variable_declarator`'s name
      - `export default function foo() {}` / `export default class Foo {}`
        -> the underlying declaration's name
      - `export default someIdentifier;` -> that identifier's text
    NOT covered: `export * from '...'` (wildcard re-export has no local
    name to record) -- an acknowledged v1 gap.
    """
    names: list[str] = []

    export_clause = _first_child_of_type(export_node, ("export_clause",))
    if export_clause is not None:
        for child in export_clause.children:
            if child.type != "export_specifier":
                continue
            alias_node = child.child_by_field_name("alias")
            name_node = child.child_by_field_name("name")
            target = alias_node if alias_node is not None else name_node
            if target is not None:
                names.append(_text(source, target))
        return names

    decl = export_node.child_by_field_name("declaration")
    if decl is not None:
        if decl.type in ("lexical_declaration", "variable_declaration"):
            for child in decl.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        names.append(_text(source, name_node))
        else:
            name_node = decl.child_by_field_name("name")
            if name_node is not None:
                names.append(_text(source, name_node))
        return names

    value_node = export_node.child_by_field_name("value")
    if value_node is not None and value_node.type in ("identifier", "type_identifier"):
        names.append(_text(source, value_node))

    return names


def _jsts_parse(path: str, source: bytes, *, language: str, is_typescript: bool) -> FileNode:
    parser = _get_parser(language)
    tree = parser.parse(source)
    root = tree.root_node

    imports: list[ImportSymbol] = []
    exports: list[str] = []
    classes: list[ClassSymbol] = []
    functions: list[FunctionSymbol] = []
    parse_errors: list[str] = []

    def handle_declaration(decl_node: Any, top_node: Any) -> None:
        if decl_node.type in _JSTS_CLASS_DECLARATIONS:
            classes.append(_jsts_class(source, decl_node, top_node, is_typescript))
        elif decl_node.type == "function_declaration":
            functions.append(_jsts_function(source, decl_node))
        elif decl_node.type in ("lexical_declaration", "variable_declaration"):
            # `const name = (...) => {...}` / `const name = function(...) {...}`
            # -- a common real-world shape worth covering; anything else
            # bound by a top-level `const`/`let`/`var` (non-function values)
            # is intentionally not modeled as a FunctionSymbol.
            for child in decl_node.children:
                if child.type != "variable_declarator":
                    continue
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")
                if name_node is None or value_node is None:
                    continue
                if value_node.type in ("arrow_function", "function_expression"):
                    functions.append(
                        _jsts_function(source, value_node, name_override=_text(source, name_node))
                    )
        # `interface_declaration` / `type_alias_declaration` / `enum_declaration`
        # etc. are intentionally not modeled as ClassSymbol/FunctionSymbol --
        # the shared FileNode contract only expects kind="class" from JS/TS
        # (see ClassSymbol's docstring). `exports` tracking above still
        # records their names on a best-effort basis regardless.

    for child in root.children:
        try:
            if child.type == "import_statement":
                imports.append(_jsts_import(source, child))
            elif child.type == "export_statement":
                exports.extend(_jsts_collect_export_names(source, child))
                decl = child.child_by_field_name("declaration")
                if decl is not None:
                    handle_declaration(decl, child)
            elif child.type in _JSTS_CLASS_DECLARATIONS or child.type in (
                "function_declaration",
                "lexical_declaration",
                "variable_declaration",
            ):
                handle_declaration(child, child)
        except Exception as exc:  # noqa: BLE001 - one malformed top-level statement must not sink the file
            parse_errors.append(f"failed to extract {child.type} at line {child.start_point[0] + 1}: {exc}")

    if root.has_error:
        parse_errors.append("tree-sitter reported a syntax error somewhere in this file (partial results returned)")

    # De-dup while preserving first-seen order -- the same name can
    # legitimately appear twice (e.g. `export function foo() {}` followed by
    # `export default foo;`), which would otherwise show up as a spurious
    # repeat in FileNode.exports.
    deduped_exports = list(dict.fromkeys(exports))

    return FileNode(
        path=path,
        language=language,
        package="",
        classes=classes,
        functions=functions,
        imports=imports,
        exports=deduped_exports,
        parse_errors=parse_errors,
    )


def _parse_javascript(path: str, source: bytes) -> FileNode:
    return _jsts_parse(path, source, language="javascript", is_typescript=False)


def _parse_typescript(path: str, source: bytes) -> FileNode:
    return _jsts_parse(path, source, language="typescript", is_typescript=True)


_PARSERS: dict[str, Callable[[str, bytes], FileNode]] = {
    "java": _parse_java,
    "python": _parse_python,
    "javascript": _parse_javascript,
    "typescript": _parse_typescript,
}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def parse_file(repo_root: str, rel_path: str) -> Optional[FileNode]:
    """Parse one file (given repo-root-relative, POSIX-style `rel_path`)
    into a `FileNode`. Returns None if the file's extension isn't a
    supported language at all (the caller should simply skip it, not treat
    it as an error).

    Never raises: any parser failure degrades to the best available
    fallback (Java: the Phase 2 regex analyzer; others: an empty-but-valid
    FileNode with `parse_errors` populated) so one bad file can never sink
    an entire repository index.
    """
    language = language_for_extension(Path(rel_path).suffix)
    if language is None:
        return None

    abs_path = Path(repo_root) / rel_path
    try:
        source = abs_path.read_bytes()
    except OSError as exc:
        return FileNode(path=rel_path, language=language, parse_errors=[f"could not read file: {exc}"])

    parse_fn = _PARSERS[language]
    try:
        return parse_fn(rel_path, source)
    except Exception as exc:  # noqa: BLE001 - a parser failure must degrade, never crash the caller
        if language == "java":
            try:
                return _java_fallback(rel_path, abs_path)
            except Exception as fallback_exc:  # noqa: BLE001
                return FileNode(
                    path=rel_path,
                    language=language,
                    parse_errors=[f"tree-sitter failed ({exc}); fallback also failed ({fallback_exc})"],
                    fallback_used=True,
                )
        return FileNode(path=rel_path, language=language, parse_errors=[f"parse failed: {exc}"])
