"""Tests for kenzo.memory.java_analyzer.analyze_java_repository against the
`spring_boot_repo` fixture: stereotype/type classification, endpoint
extraction, constructor-injection dependency capture, call-graph extraction,
target/ exclusion, and graceful handling of a malformed .java file.
"""

from __future__ import annotations

from kenzo.memory.java_analyzer import analyze_java_repository


def _by_name(symbols: list[dict], name: str) -> dict:
    matches = [s for s in symbols if s["name"] == name]
    assert len(matches) == 1, f"expected exactly one symbol named {name!r}, found {len(matches)}"
    return matches[0]


def _method(symbol: dict, name: str) -> dict:
    matches = [m for m in symbol["methods"] if m["name"] == name]
    assert len(matches) == 1, f"expected exactly one method named {name!r}, found {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


def test_login_and_logout_controllers_found_with_type_controller(spring_boot_repo):
    result = analyze_java_repository(str(spring_boot_repo))
    symbols = result["symbols"]

    login = _by_name(symbols, "LoginController")
    logout = _by_name(symbols, "LogoutController")

    assert login["type"] == "controller"
    assert logout["type"] == "controller"
    assert login["kind"] == "class"
    assert logout["kind"] == "class"


def test_endpoints_correctly_extracted(spring_boot_repo):
    result = analyze_java_repository(str(spring_boot_repo))
    symbols = result["symbols"]

    login = _by_name(symbols, "LoginController")
    logout = _by_name(symbols, "LogoutController")

    assert login["endpoints"] == ["/api/auth/login"]
    assert logout["endpoints"] == ["/api/auth/logout"]


# ---------------------------------------------------------------------------
# AuthService (service layer)
# ---------------------------------------------------------------------------


def test_auth_service_found_as_service_with_user_repository_dependency(spring_boot_repo):
    result = analyze_java_repository(str(spring_boot_repo))
    symbols = result["symbols"]

    auth_service = _by_name(symbols, "AuthService")

    assert auth_service["type"] == "service"
    assert "UserRepository" in auth_service["dependencies"]


# ---------------------------------------------------------------------------
# UserRepository (repository layer, interface, no method bodies)
# ---------------------------------------------------------------------------


def test_user_repository_found_as_interface_repository(spring_boot_repo):
    result = analyze_java_repository(str(spring_boot_repo))
    symbols = result["symbols"]

    user_repository = _by_name(symbols, "UserRepository")

    assert user_repository["kind"] == "interface"
    assert user_repository["type"] == "repository"
    # An interface with only method signatures (no bodies) yields no methods
    # under the regex-based method extractor (it requires a trailing `{`).
    assert user_repository["methods"] == []


# ---------------------------------------------------------------------------
# LogoutController's captured dependency + call graph
# ---------------------------------------------------------------------------


def test_logout_controller_dependencies_include_auth_service(spring_boot_repo):
    result = analyze_java_repository(str(spring_boot_repo))
    symbols = result["symbols"]

    logout = _by_name(symbols, "LogoutController")

    assert "AuthService" in logout["dependencies"]


def test_logout_method_calls_include_auth_service_target(spring_boot_repo):
    result = analyze_java_repository(str(spring_boot_repo))
    symbols = result["symbols"]

    logout = _by_name(symbols, "LogoutController")
    logout_method = _method(logout, "logout")

    assert any(call["target_type"] == "AuthService" for call in logout_method["calls"])
    matching_call = next(call for call in logout_method["calls"] if call["target_type"] == "AuthService")
    assert matching_call["method"] == "terminateSession"


# ---------------------------------------------------------------------------
# target/ exclusion
# ---------------------------------------------------------------------------


def test_garbage_class_and_target_dir_never_appear_in_analyzed_output(spring_boot_repo):
    result = analyze_java_repository(str(spring_boot_repo))

    # Only the four real .java source files should ever be analyzed --
    # target/classes/Garbage.class is neither a .java file nor reachable
    # (target/ is in DEFAULT_IGNORED_DIRS).
    assert result["file_count"] == 4
    for symbol in result["symbols"]:
        assert "target" not in symbol["file"].split("/")
        assert "Garbage" not in symbol["file"]


def test_java_file_under_target_directory_is_excluded_by_ignored_dirs(tmp_path):
    root = tmp_path / "proj"
    src_dir = root / "src" / "main" / "java" / "com" / "example"
    src_dir.mkdir(parents=True)
    (src_dir / "RealService.java").write_text(
        "package com.example;\n\n"
        "public class RealService {\n"
        "    public void run() {\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    target_generated = root / "target" / "generated-sources"
    target_generated.mkdir(parents=True)
    (target_generated / "ShouldBeExcluded.java").write_text(
        "package com.example.generated;\n\n"
        "public class ShouldBeExcluded {\n"
        "}\n",
        encoding="utf-8",
    )

    result = analyze_java_repository(str(root))

    assert result["file_count"] == 1
    names = {s["name"] for s in result["symbols"]}
    assert names == {"RealService"}
    assert not any("target" in s["file"].split("/") for s in result["symbols"])


# ---------------------------------------------------------------------------
# Malformed .java file must not abort the whole scan
# ---------------------------------------------------------------------------


def test_malformed_java_file_does_not_raise_and_other_files_still_analyzed(tmp_path):
    root = tmp_path / "proj"
    src_dir = root / "src" / "main" / "java" / "com" / "example"
    src_dir.mkdir(parents=True)

    (src_dir / "GoodService.java").write_text(
        "package com.example;\n\n"
        "import org.springframework.stereotype.Service;\n\n"
        "@Service\n"
        "public class GoodService {\n"
        "    public void doWork() {\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    # Deliberately malformed: an opened method body that is never closed
    # (unbalanced braces / truncated source).
    (src_dir / "BadFile.java").write_text(
        "package com.example;\n\n"
        "public class BadFile {\n"
        "    public void broken() {\n"
        "        System.out.println(\"never closed\");\n",
        encoding="utf-8",
    )

    result = analyze_java_repository(str(root))  # must not raise

    assert result["file_count"] == 2
    names = {s["name"] for s in result["symbols"]}
    assert "GoodService" in names
    good_service = next(s for s in result["symbols"] if s["name"] == "GoodService")
    assert good_service["type"] == "service"
