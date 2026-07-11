"""Test discovery: given a source file, find the test file(s) that most
likely exercise it.

Naming-convention heuristics only (no semantic understanding beyond the
`SymbolIndex` we already have) -- covers the common conventions across the
four supported languages: Java (`FooTest.java`, `FooTests.java`, `FooIT.java`
under `src/test/...`), Python (`test_foo.py`, `foo_test.py` under `tests/`),
and JS/TS (`foo.test.ts`, `foo.spec.ts`).
"""

from __future__ import annotations

from pathlib import PurePosixPath

from kenzo.intelligence.symbol_index import SymbolIndex

_TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs", "it", "integration-test", "integrationtest"}


def _candidate_test_stems(base_name: str) -> set[str]:
    lower = base_name.lower()
    return {
        f"{lower}test",
        f"{lower}tests",
        f"{lower}it",
        f"{lower}itcase",
        f"test{lower}",
        f"test_{lower}",
        f"{lower}_test",
        f"{lower}_tests",
        f"{lower}.test",
        f"{lower}.spec",
    }


def find_related_tests(symbol_index: SymbolIndex, file_path: str) -> list[str]:
    """Return every indexed file path that looks like a test for `file_path`,
    sorted for determinism. Two independent signals are combined:

    1. Naming convention: the test file's stem matches one of the common
       `FooTest`/`test_foo`/`foo.spec` etc. patterns for `file_path`'s stem,
       optionally with a "test"-ish directory segment.
    2. Reference: any file that lives under a "test"-ish directory AND
       whose imports/package reference mentions `file_path`'s class name(s)
       -- catches integration tests that don't follow a naming convention
       but do import the class under test.
    """
    base_name = PurePosixPath(file_path).stem
    candidate_stems = _candidate_test_stems(base_name)

    file_node = symbol_index.get_file(file_path)
    class_names = {c.name.lower() for c in file_node.classes} if file_node else set()

    matches: set[str] = set()

    for path, file in symbol_index.files.items():
        if path == file_path:
            continue

        path_obj = PurePosixPath(path)
        stem_lower = path_obj.stem.lower()
        parts_lower = {p.lower() for p in path_obj.parts}
        in_test_dir = bool(parts_lower & _TEST_DIR_NAMES)

        if stem_lower in candidate_stems:
            matches.add(path)
            continue

        # A looser match: the file lives under a test-ish directory AND its
        # stem contains the source file's base name (e.g.
        # "auth/LogoutControllerIntegrationTest.java" for "LogoutController").
        if in_test_dir and base_name.lower() in stem_lower:
            matches.add(path)
            continue

        # Reference signal: a test-directory file that imports/mentions one
        # of the source file's class names, even without a matching stem.
        if in_test_dir and class_names:
            imported_modules = " ".join(imp.module.lower() for imp in file.imports)
            if any(name in imported_modules for name in class_names):
                matches.add(path)

    return sorted(matches)
