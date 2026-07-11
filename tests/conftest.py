"""Shared pytest fixtures for the Kenzo test suite.

Three fixtures are used across most test modules:

- `kenzo_home`: points the `KENZO_HOME` environment variable at an isolated
  tmp directory so config tests never read/write the real `~/.kenzo` on the
  machine running the suite.
- `repo_root`: a tmp git repository seeded with a couple of sample files, a
  `.gitignore`, and one commit, for tools/memory tests that need a real
  repository to operate on.
- `spring_boot_repo`: a small, realistic fake Spring Boot Maven project (Phase
  2 "Repository Intelligence" tests: project detection, the Java symbol
  analyzer, repository_tree, find_symbol/map_flow, Tester's Java awareness,
  and the full AgentRuntime wiring).

`run_git` is a plain helper (not a fixture) for issuing extra git commands
against `repo_root` from within individual test bodies (e.g. to create a
second commit, stage a change, or add an untracked file).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def kenzo_home(tmp_path, monkeypatch):
    """Point KENZO_HOME at an isolated tmp directory for config tests."""
    home_dir = tmp_path / "kenzo_home"
    home_dir.mkdir()
    monkeypatch.setenv("KENZO_HOME", str(home_dir))
    return home_dir


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git command against `cwd`, raising if it fails.

    Kept as a plain function (not a fixture) so individual tests can issue
    additional git commands (add/commit/etc.) against the `repo_root`
    fixture's repository.
    """
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo_root(tmp_path) -> Path:
    """A tmp git repository with a couple of sample files and one commit.

    Branch is pinned to "main" via `symbolic-ref` (rather than relying on
    `git init -b main` or the machine's `init.defaultBranch`) so tests can
    assert on the branch name deterministically regardless of the git
    version or global config of the machine running the suite.
    """
    root = tmp_path / "repo"
    root.mkdir()

    (root / "README.md").write_text(
        "# Sample repo\n\nUsed for Kenzo tests.\n", encoding="utf-8"
    )
    (root / "main.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "*.log\nnode_modules/\ndist/\nbuild/\n", encoding="utf-8"
    )

    run_git(["init"], cwd=root)
    run_git(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=root)
    run_git(["config", "user.email", "kenzo-tests@example.com"], cwd=root)
    run_git(["config", "user.name", "Kenzo Tests"], cwd=root)
    run_git(["add", "-A"], cwd=root)
    run_git(["commit", "-m", "Initial commit"], cwd=root)

    return root


# ---------------------------------------------------------------------------
# spring_boot_repo: a small, realistic fake Spring Boot Maven project.
# ---------------------------------------------------------------------------

_POM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>

    <groupId>com.example</groupId>
    <artifactId>auth-service</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <java.version>17</java.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""

_LOGIN_CONTROLLER_JAVA = """\
package com.example.auth;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
public class LoginController {

    private final AuthService authService;

    public LoginController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public String login() {
        return authService.authenticate();
    }
}
"""

_LOGOUT_CONTROLLER_JAVA = """\
package com.example.auth;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
public class LogoutController {

    private final AuthService authService;

    public LogoutController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/logout")
    public String logout() {
        authService.terminateSession();
        return "logged out";
    }
}
"""

_AUTH_SERVICE_JAVA = """\
package com.example.auth;

import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final UserRepository userRepository;

    public AuthService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public String authenticate() {
        return "authenticated";
    }

    public void terminateSession() {
        userRepository.deleteSession();
    }
}
"""

_USER_REPOSITORY_JAVA = """\
package com.example.auth;

import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository {

    void deleteSession();

    String findByUsername(String username);
}
"""


@pytest.fixture
def spring_boot_repo(tmp_path) -> Path:
    """A small, realistic fake Spring Boot Maven project, used by the Phase 2
    "Repository Intelligence" tests.

    Layout::

        pom.xml                              (mentions spring-boot-starter-web)
        .gitignore
        target/classes/Garbage.class          (build output that must be excluded)
        src/main/java/com/example/auth/
            LoginController.java              (@RestController, POST /api/auth/login)
            LogoutController.java             (@RestController, POST /api/auth/logout;
                                                constructor-injects AuthService and
                                                calls authService.terminateSession())
            AuthService.java                  (@Service; constructor-injects
                                                UserRepository; terminateSession()
                                                calls userRepository.deleteSession())
            UserRepository.java               (@Repository interface, no method
                                                bodies)

    Not a git repository (none of the Phase 2 tooling exercised against this
    fixture needs one; `kenzo.tools.git` already degrades gracefully to a
    "not a git repository" ToolResult when asked to operate on a plain
    directory).
    """
    root = tmp_path / "spring_boot_repo"
    root.mkdir()

    (root / "pom.xml").write_text(_POM_XML, encoding="utf-8")
    (root / ".gitignore").write_text("target/\n*.class\n", encoding="utf-8")

    target_classes = root / "target" / "classes"
    target_classes.mkdir(parents=True)
    (target_classes / "Garbage.class").write_bytes(b"\xca\xfe\xba\xbeJUNK-NOT-REAL-BYTECODE")

    java_dir = root / "src" / "main" / "java" / "com" / "example" / "auth"
    java_dir.mkdir(parents=True)
    (java_dir / "LoginController.java").write_text(_LOGIN_CONTROLLER_JAVA, encoding="utf-8")
    (java_dir / "LogoutController.java").write_text(_LOGOUT_CONTROLLER_JAVA, encoding="utf-8")
    (java_dir / "AuthService.java").write_text(_AUTH_SERVICE_JAVA, encoding="utf-8")
    (java_dir / "UserRepository.java").write_text(_USER_REPOSITORY_JAVA, encoding="utf-8")

    return root
