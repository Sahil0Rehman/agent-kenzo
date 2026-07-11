"""Tests for kenzo.core.project.ProjectAnalyzer: Java (Maven/Gradle, with and
without wrapper scripts), Python (pyproject.toml/requirements.txt/setup.py,
poetry), Node (npm/yarn), framework sniffing, and the "unknown" fallback.
"""

from __future__ import annotations

from kenzo.core.project import ProjectAnalyzer, detect_project
from kenzo.core.schemas import ProjectInfo


def _write(path, name: str, content: str = "") -> None:
    (path / name).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Java / Maven
# ---------------------------------------------------------------------------


def test_maven_detected_without_wrapper_uses_plain_mvn_test(tmp_path):
    _write(tmp_path, "pom.xml", "<project></project>")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "java"
    assert info.build == "maven"
    assert info.test_command == "mvn test"
    assert info.framework is None


def test_maven_detected_with_mvnw_wrapper_flips_test_command(tmp_path):
    _write(tmp_path, "pom.xml", "<project></project>")
    _write(tmp_path, "mvnw")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "java"
    assert info.build == "maven"
    assert info.test_command == "./mvnw test"


def test_maven_detected_with_mvnw_cmd_wrapper_flips_test_command(tmp_path):
    _write(tmp_path, "pom.xml", "<project></project>")
    _write(tmp_path, "mvnw.cmd")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.test_command == "./mvnw test"


def test_spring_boot_framework_detected_via_pom_content(tmp_path):
    _write(
        tmp_path,
        "pom.xml",
        "<project><dependencies><dependency>"
        "<artifactId>spring-boot-starter-web</artifactId>"
        "</dependency></dependencies></project>",
    )

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "java"
    assert info.framework == "spring-boot"


def test_spring_framework_detected_via_pom_without_boot(tmp_path):
    _write(
        tmp_path,
        "pom.xml",
        "<project><dependencies><dependency>"
        "<groupId>org.springframework</groupId>"
        "</dependency></dependencies></project>",
    )

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.framework == "spring"


# ---------------------------------------------------------------------------
# Java / Gradle
# ---------------------------------------------------------------------------


def test_gradle_groovy_detected_without_wrapper(tmp_path):
    _write(tmp_path, "build.gradle", "plugins { id 'java' }")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "java"
    assert info.build == "gradle"
    assert info.test_command == "gradle test"


def test_gradle_kotlin_dsl_detected(tmp_path):
    _write(tmp_path, "build.gradle.kts", "plugins { java }")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "java"
    assert info.build == "gradle"
    assert info.test_command == "gradle test"


def test_gradle_detected_with_gradlew_wrapper_flips_test_command(tmp_path):
    _write(tmp_path, "build.gradle", "plugins { id 'java' }")
    _write(tmp_path, "gradlew")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.test_command == "./gradlew test"


def test_gradle_detected_with_gradlew_bat_wrapper_flips_test_command(tmp_path):
    _write(tmp_path, "build.gradle", "plugins { id 'java' }")
    _write(tmp_path, "gradlew.bat")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.test_command == "./gradlew test"


def test_maven_takes_priority_when_both_pom_and_gradle_files_present(tmp_path):
    _write(tmp_path, "pom.xml", "<project></project>")
    _write(tmp_path, "build.gradle", "plugins { id 'java' }")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.build == "maven"


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def test_python_detected_via_pyproject_toml_alone(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\nname = \"demo\"\n")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "python"
    assert info.build == "pip"
    assert info.test_command == "pytest"
    assert info.framework is None


def test_python_detected_via_requirements_txt_alone(tmp_path):
    _write(tmp_path, "requirements.txt", "requests==2.31.0\n")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "python"
    assert info.test_command == "pytest"


def test_python_detected_via_setup_py_alone(tmp_path):
    _write(tmp_path, "setup.py", "from setuptools import setup\nsetup(name='demo')\n")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "python"
    assert info.test_command == "pytest"


def test_python_poetry_pyproject_sets_build_poetry(tmp_path):
    _write(
        tmp_path,
        "pyproject.toml",
        "[tool.poetry]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    )

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "python"
    assert info.build == "poetry"


def test_python_framework_detection_django_fastapi_flask(tmp_path):
    _write(tmp_path, "requirements.txt", "django==5.0\n")
    assert ProjectAnalyzer().analyze(str(tmp_path)).framework == "django"

    (tmp_path / "requirements.txt").write_text("fastapi==0.111.0\n", encoding="utf-8")
    assert ProjectAnalyzer().analyze(str(tmp_path)).framework == "fastapi"

    (tmp_path / "requirements.txt").write_text("flask==3.0.0\n", encoding="utf-8")
    assert ProjectAnalyzer().analyze(str(tmp_path)).framework == "flask"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def test_node_detected_via_package_json_without_yarn_lock(tmp_path):
    _write(tmp_path, "package.json", '{"name": "demo", "dependencies": {}}')

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "node"
    assert info.build == "npm"
    assert info.test_command == "npm test"


def test_node_detected_via_package_json_with_yarn_lock_changes_test_command(tmp_path):
    _write(tmp_path, "package.json", '{"name": "demo", "dependencies": {}}')
    _write(tmp_path, "yarn.lock", "")

    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info.language == "node"
    assert info.build == "yarn"
    assert info.test_command == "yarn test"


def test_node_framework_detection_react_express_nestjs_nextjs(tmp_path):
    _write(tmp_path, "package.json", '{"dependencies": {"react": "^18.0.0"}}')
    assert ProjectAnalyzer().analyze(str(tmp_path)).framework == "react"

    (tmp_path / "package.json").write_text('{"dependencies": {"express": "^4.0.0"}}', encoding="utf-8")
    assert ProjectAnalyzer().analyze(str(tmp_path)).framework == "express"

    (tmp_path / "package.json").write_text('{"dependencies": {"@nestjs/core": "^10.0.0"}}', encoding="utf-8")
    assert ProjectAnalyzer().analyze(str(tmp_path)).framework == "nestjs"

    (tmp_path / "package.json").write_text('{"dependencies": {"next": "^14.0.0"}}', encoding="utf-8")
    assert ProjectAnalyzer().analyze(str(tmp_path)).framework == "next.js"


# ---------------------------------------------------------------------------
# Unknown fallback
# ---------------------------------------------------------------------------


def test_unknown_fallback_for_empty_directory(tmp_path):
    info = ProjectAnalyzer().analyze(str(tmp_path))

    assert info == ProjectInfo(language="unknown", framework=None, build=None, test_command=None)


def test_detect_project_convenience_wrapper_matches_analyzer(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\nname = \"demo\"\n")

    assert detect_project(str(tmp_path)) == ProjectAnalyzer().analyze(str(tmp_path))
