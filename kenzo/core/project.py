"""Project type detection: what language/framework/build system a repository
uses, and what command actually validates it.

This is the fix for a very concrete Phase 1.5 gap -- Tester's default tooling
(command allowlist, prompt wording, "pytest"/"python" examples) was silently
Python-centric. `ProjectAnalyzer` runs once per `AgentRuntime` and its result
is threaded into the context bundle every agent reads and into Tester's
prompt directly, so a Java/Maven or Node repo gets the right test command
instead of a Python-shaped guess.

Detection is deliberately simple (marker-file presence, not a build-tool
plugin system): it does not need to be exhaustive, only good enough to stop
recommending `pytest` for a Spring Boot project.
"""

from __future__ import annotations

from pathlib import Path

from kenzo.core.schemas import ProjectInfo


def _has(repo_root: Path, name: str) -> bool:
    return (repo_root / name).is_file()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


class ProjectAnalyzer:
    """Detects a repository's language, framework, build system, and the
    command that should be used to validate a change to it."""

    def analyze(self, repo_root: str) -> ProjectInfo:
        root = Path(repo_root).resolve()

        java_info = self._detect_java(root)
        if java_info is not None:
            return java_info

        python_info = self._detect_python(root)
        if python_info is not None:
            return python_info

        node_info = self._detect_node(root)
        if node_info is not None:
            return node_info

        return ProjectInfo(language="unknown", framework=None, build=None, test_command=None)

    # -- Java -----------------------------------------------------------

    def _detect_java(self, root: Path) -> ProjectInfo | None:
        has_maven = _has(root, "pom.xml")
        # Gradle projects may use either the Groovy or Kotlin DSL.
        has_gradle = _has(root, "build.gradle") or _has(root, "build.gradle.kts")

        if not has_maven and not has_gradle:
            return None

        if has_maven:
            build = "maven"
            wrapper = _has(root, "mvnw") or _has(root, "mvnw.cmd")
            test_command = "./mvnw test" if wrapper else "mvn test"
            framework = self._detect_java_framework(_read_text(root / "pom.xml"))
        else:
            build = "gradle"
            wrapper = _has(root, "gradlew") or _has(root, "gradlew.bat")
            test_command = "./gradlew test" if wrapper else "gradle test"
            gradle_file = "build.gradle.kts" if _has(root, "build.gradle.kts") else "build.gradle"
            framework = self._detect_java_framework(_read_text(root / gradle_file))

        return ProjectInfo(language="java", framework=framework, build=build, test_command=test_command)

    def _detect_java_framework(self, build_file_text: str) -> str | None:
        lowered = build_file_text.lower()
        if "spring-boot" in lowered or "springframework.boot" in lowered:
            return "spring-boot"
        if "springframework" in lowered:
            return "spring"
        if "quarkus" in lowered:
            return "quarkus"
        if "micronaut" in lowered:
            return "micronaut"
        return None

    # -- Python -----------------------------------------------------------

    def _detect_python(self, root: Path) -> ProjectInfo | None:
        has_pyproject = _has(root, "pyproject.toml")
        has_requirements = _has(root, "requirements.txt")
        has_setup_py = _has(root, "setup.py")

        if not (has_pyproject or has_requirements or has_setup_py):
            return None

        pyproject_text = _read_text(root / "pyproject.toml") if has_pyproject else ""
        requirements_text = _read_text(root / "requirements.txt") if has_requirements else ""
        combined = f"{pyproject_text}\n{requirements_text}".lower()

        framework = None
        if "django" in combined:
            framework = "django"
        elif "fastapi" in combined:
            framework = "fastapi"
        elif "flask" in combined:
            framework = "flask"

        build = "poetry" if "[tool.poetry]" in pyproject_text.lower() else "pip"
        test_command = "pytest"

        return ProjectInfo(language="python", framework=framework, build=build, test_command=test_command)

    # -- Node ---------------------------------------------------------------

    def _detect_node(self, root: Path) -> ProjectInfo | None:
        if not _has(root, "package.json"):
            return None

        package_json_text = _read_text(root / "package.json")
        lowered = package_json_text.lower()

        framework = None
        if "\"next\"" in lowered or "\"next.js\"" in lowered:
            framework = "next.js"
        elif "\"express\"" in lowered:
            framework = "express"
        elif "\"@nestjs/core\"" in lowered:
            framework = "nestjs"
        elif "\"react\"" in lowered:
            framework = "react"

        build = "yarn" if _has(root, "yarn.lock") else "npm"
        test_command = "yarn test" if build == "yarn" else "npm test"

        return ProjectInfo(language="node", framework=framework, build=build, test_command=test_command)


def detect_project(repo_root: str) -> ProjectInfo:
    """Convenience wrapper around `ProjectAnalyzer().analyze(repo_root)`."""
    return ProjectAnalyzer().analyze(repo_root)
