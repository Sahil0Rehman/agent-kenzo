"""Phase 6 "MCP Integration" required end-to-end test: the full scripted
workflow from the Phase 6 spec --

    User: "Fix issue #431"
    Kenzo: 1. Fetch GitHub issue -> 2. Understand description -> 3. Search
           repo -> 4. Fix code -> 5. Run tests -> 6. Prepare PR

-- driven through a REAL `AgentController` via a REAL `AgentRuntime`, with
the GitHub MCP tool calls (`github_get_issue`, `github_create_pull_request`)
going through the REAL fake stdio subprocess server
(`tests/fake_mcp_servers/github_server.py`). Only the model/router layer is
faked (a scripted `ModelRouter.complete`), exactly matching
`tests/test_controller_scenarios.py`'s Scenario 3 convention (which this
file copies its scaffolding from, kept self-contained rather than imported,
matching how `test_phase6_mcp_foundation.py`/`test_phase6_mcp_error_recovery
.py` don't share helpers with each other either).

Bug fixture: `github_server.py`'s `get_issue()` always returns the same
canned text ("Logout button does nothing after OIDC redirect ... clicking
'Logout' ... no outgoing request at all ... Expected: clicking Logout should
call /api/auth/logout and redirect ..."), regardless of which issue number is
passed. `logout_bug_repo` below is a small polyglot (Java backend + static
JS frontend) fixture whose JS logout click-handler has EXACTLY that bug --
`onLogoutClick()` only hides a menu locally and never calls
`POST /api/auth/logout` -- so the scripted EDIT step applies a genuine,
verifiable fix (asserted against the real post-edit file content and diff),
not just an arbitrary file write.

Proof that `github_get_issue`/`github_create_pull_request` really executed
against the real subprocess (not merely requested by the scripted router):
an instance-level spy wraps the actual `MCPServerConnection.call_tool` bound
method used by every MCP `ToolSpec.func` closure (see
`kenzo.mcp.adapter.build_tool_spec`'s `_call`) -- the one true boundary
where the real MCP round trip happens, underneath every tool-registry
wrapping layer (`CachedToolRegistry`/`CheckpointingToolRegistry`/
`ReindexingToolRegistry`). The spy captures the real arguments sent and the
real `ToolResult` (including the real canned response text) that came back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kenzo.config.settings import (
    KenzoConfig,
    MCPConfig,
    MCPServerConfig,
    PermissionsConfig,
    ProviderConfig,
)
from kenzo.core.controller import AgentController, ControllerAction
from kenzo.core.runtime import AgentRuntime
from kenzo.models.router import ModelRouter

FAKE_SERVERS_DIR = Path(__file__).parent / "fake_mcp_servers"
PY = sys.executable


# ---------------------------------------------------------------------------
# Fake litellm-shaped response scaffolding + scripted router -- copied from
# tests/test_controller_scenarios.py's exact pattern (see that module's
# docstring for the full rationale), kept self-contained in this file rather
# than imported.
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeModelResponse:
    def __init__(self, content, tool_calls=None):
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls))]


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FakeFunction(name, arguments)


def _is_decide_call(tool_choice) -> bool:
    return (
        isinstance(tool_choice, dict)
        and tool_choice.get("function", {}).get("name") == "decide_next_action"
    )


class _ControllerScriptedRouter:
    """See tests/test_controller_scenarios.py's module docstring for the
    full rationale: a dedicated ordered queue for `decide_next_action` calls,
    and a separate flat ordered queue for every action-execution call."""

    def __init__(self, decisions, actions):
        self._decisions = list(decisions)
        self._actions = list(actions)
        self.decision_calls: list[dict] = []
        self.action_calls: list[dict] = []

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        snapshot = [dict(m) for m in messages]
        if _is_decide_call(tool_choice):
            self.decision_calls.append({"task_type": task_type, "messages": snapshot})
            if not self._decisions:
                raise AssertionError("ran out of scripted decisions")
            return self._decisions.pop(0)

        self.action_calls.append({"task_type": task_type, "messages": snapshot})
        if not self._actions:
            raise AssertionError("ran out of scripted action responses")
        return self._actions.pop(0)


def _decision(action: ControllerAction, rationale: str, target: str = ""):
    call = _FakeToolCall(
        f"decide-{action.value}",
        "decide_next_action",
        json.dumps({"action": action.value, "rationale": rationale, "target": target}),
    )
    return _FakeModelResponse(None, tool_calls=[call])


def _tool_call_response(call_id: str, tool_name: str, arguments: dict):
    return _FakeModelResponse(None, tool_calls=[_FakeToolCall(call_id, tool_name, json.dumps(arguments))])


def _final_text(text: str):
    return _FakeModelResponse(text)


# ---------------------------------------------------------------------------
# logout_bug_repo: a small polyglot fixture with a REAL, verifiable bug that
# matches github_server.py's get_issue() canned description exactly.
# ---------------------------------------------------------------------------

_POM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>auth-dashboard</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>
    <properties>
        <java.version>17</java.version>
    </properties>
</project>
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

    public void terminateSession() {
        // Real session/token invalidation lives here -- irrelevant to issue
        // #431 since the client never even sends the request that would
        // reach it (see dashboard.js's onLogoutClick, the real bug).
    }
}
"""

_DASHBOARD_JS_PRELUDE = """\
// Dashboard client-side interactions.

function onLoginSuccess(user) {
    document.getElementById('welcome').textContent = 'Welcome, ' + user;
}

"""

# The genuine bug matching github_server.py's get_issue() text verbatim:
# "the button click is registered ... but the user session remains active
# ... network tab shows no outgoing request at all". onLogoutClick() below
# really does exactly that -- it only touches local DOM state.
_BUGGY_ONLOGOUT_BLOCK = """\
function onLogoutClick() {
    // BUG (issue #431): only hides the menu locally -- never actually tells
    // the server to end the session, so the backend session (and therefore
    // the OIDC provider's own session) stays alive and the user is never
    // redirected anywhere. Network tab shows no outgoing request at all.
    document.getElementById('user-menu').classList.add('hidden');
}
"""

# The genuine fix: a real POST /api/auth/logout call, then a redirect to the
# identity provider's post-logout page -- exactly what the issue's
# "Expected" section describes.
_FIXED_ONLOGOUT_BLOCK = """\
function onLogoutClick() {
    document.getElementById('user-menu').classList.add('hidden');
    fetch('/api/auth/logout', { method: 'POST' }).then(function () {
        window.location.href = '/api/auth/post-logout';
    });
}
"""

_DASHBOARD_JS_TAIL = """
function initDashboard() {
    document.getElementById('logout-button').addEventListener('click', onLogoutClick);
}
"""

_RELATIVE_JS_PATH = "src/main/resources/static/js/dashboard.js"


@pytest.fixture
def logout_bug_repo(tmp_path) -> Path:
    """A small, realistic polyglot repo (Java backend + static JS frontend)
    whose JS logout click-handler has the exact bug `github_server.py`'s
    `get_issue()` describes -- see module docstring."""
    root = tmp_path / "logout_bug_repo"
    root.mkdir()

    # newline="" disables platform newline translation -- on Windows,
    # Path.write_text() otherwise silently turns '\n' into '\r\n', which
    # would break edit_file's later EXACT string match of _BUGGY_ONLOGOUT_
    # BLOCK (a plain '\n'-only Python literal) against the on-disk bytes.
    # Kenzo's own write_file/edit_file tools use the same newline="" for
    # exactly this reason (see kenzo.tools.filesystem).
    (root / "pom.xml").write_text(_POM_XML, encoding="utf-8", newline="")

    java_dir = root / "src" / "main" / "java" / "com" / "example" / "auth"
    java_dir.mkdir(parents=True)
    (java_dir / "LogoutController.java").write_text(_LOGOUT_CONTROLLER_JAVA, encoding="utf-8", newline="")
    (java_dir / "AuthService.java").write_text(_AUTH_SERVICE_JAVA, encoding="utf-8", newline="")

    js_dir = root / "src" / "main" / "resources" / "static" / "js"
    js_dir.mkdir(parents=True)
    (js_dir / "dashboard.js").write_text(
        _DASHBOARD_JS_PRELUDE + _BUGGY_ONLOGOUT_BLOCK + _DASHBOARD_JS_TAIL, encoding="utf-8", newline=""
    )

    return root


def _integration_config() -> KenzoConfig:
    return KenzoConfig(
        providers={"nemotron": ProviderConfig(api_key="fake-key", default_model="fake-model")},
        permissions=PermissionsConfig(allow_write=True, allow_shell_exec=False),
        mcp=MCPConfig(
            servers={"github": MCPServerConfig(command=PY, args=[str(FAKE_SERVERS_DIR / "github_server.py")])},
            allow_write=True,
        ),
    )


# ---------------------------------------------------------------------------
# The test.
# ---------------------------------------------------------------------------


def test_fix_issue_431_full_workflow_through_real_controller_and_real_mcp_server(
    logout_bug_repo, monkeypatch
):
    decisions = [
        # Steps 1-2: fetch + understand the GitHub issue.
        _decision(
            ControllerAction.ANALYZE,
            "Fetch and read GitHub issue #431 before touching any code.",
            "431",
        ),
        # Step 3: search the repo for the buggy code the issue points at.
        _decision(
            ControllerAction.SEARCH,
            "Locate the client-side logout handler the issue points at.",
            "onLogoutClick",
        ),
        # Step 4: fix the code.
        _decision(
            ControllerAction.EDIT,
            "Apply the real fix: onLogoutClick must call POST /api/auth/logout and redirect.",
            _RELATIVE_JS_PATH,
        ),
        # Step 5: run tests.
        _decision(
            ControllerAction.TEST,
            "Shell exec is disabled for this scenario -- verify with a careful static read-through.",
        ),
        # Step 6: prepare the PR.
        _decision(
            ControllerAction.EDIT,
            "Local fix is made and verified -- prepare a pull request for it.",
            "",
        ),
        _decision(ControllerAction.FINISH, "Issue #431 is fixed, verified, and a PR has been prepared."),
    ]

    pr_title = "Fix logout button: call /api/auth/logout on click (#431)"
    pr_body = (
        "Fixes #431. onLogoutClick() in dashboard.js now issues POST /api/auth/logout "
        "and redirects to the identity provider's post-logout page once that request "
        "resolves, so an OIDC SSO session is actually terminated when the user clicks "
        "Logout."
    )
    pr_branch = "fix/431-logout-button"

    actions = [
        # ANALYZE (steps 1-2): a REAL github_get_issue call through the real
        # subprocess, then a final summary of what it said.
        _tool_call_response("call-get-issue", "github_get_issue", {"issue_number": 431}),
        _final_text(
            "Issue #431: the Logout button does nothing after an OIDC SSO login -- clicking "
            "it never issues a POST to /api/auth/logout, so the session (and the identity "
            "provider's own session) stays alive and the user just sits on the dashboard. "
            "Local username/password users are unaffected. I need to find the client-side "
            "click handler responsible for this."
        ),
        # SEARCH (step 3): a REAL native search_files call against the fixture.
        _tool_call_response("call-search-onlogout", "search_files", {"pattern": "onLogoutClick"}),
        _final_text(
            f"Found onLogoutClick() defined in {_RELATIVE_JS_PATH} -- it only hides the user "
            "menu locally and never calls POST /api/auth/logout, exactly matching the bug report."
        ),
        # EDIT #1 (step 4): a REAL edit_file call applying the genuine fix.
        _tool_call_response(
            "call-edit-fix",
            "edit_file",
            {"path": _RELATIVE_JS_PATH, "old_text": _BUGGY_ONLOGOUT_BLOCK, "new_text": _FIXED_ONLOGOUT_BLOCK},
        ),
        _final_text(
            "Fixed onLogoutClick() in dashboard.js: it now issues POST /api/auth/logout and, "
            "once that resolves, redirects the browser to the identity provider's post-logout "
            "page -- matching the issue's expected behavior."
        ),
        # TEST (step 5): shell exec is disabled, so a real read_file() re-read
        # backs a static read-through verdict, per the TEST action's own
        # documented fallback instructions.
        _tool_call_response("call-verify-read", "read_file", {"path": _RELATIVE_JS_PATH}),
        _final_text(
            json.dumps(
                {
                    "passed": True,
                    "summary": (
                        "Static read-through confirms onLogoutClick() now calls POST "
                        "/api/auth/logout and redirects to /api/auth/post-logout; shell exec "
                        "is disabled so no live test run was possible."
                    ),
                    "command": "",
                    "raw_output": "",
                }
            )
        ),
        # EDIT #2 (step 6): a REAL github_create_pull_request call through the
        # real subprocess.
        _tool_call_response(
            "call-create-pr",
            "github_create_pull_request",
            {"title": pr_title, "body": pr_body, "branch": pr_branch},
        ),
        _final_text(f"Opened a pull request for the fix on branch '{pr_branch}'; ready for review."),
    ]

    router = _ControllerScriptedRouter(decisions, actions)

    def _scripted_complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        return router.complete(task_type, messages, tools=tools, tool_choice=tool_choice, **kwargs)

    monkeypatch.setattr(ModelRouter, "complete", _scripted_complete)

    config = _integration_config()
    runtime = AgentRuntime(repo_path=str(logout_bug_repo), config=config)

    try:
        # The real github MCP server must have actually connected, and its
        # tools must actually be merged into the real registry, or nothing
        # below would prove anything real.
        assert runtime.mcp_client.connect_results["github"].success is True
        assert "github_get_issue" in runtime.tools.names()
        assert "github_create_pull_request" in runtime.tools.names()

        # Spy on the REAL MCPServerConnection.call_tool bound method -- see
        # module docstring for why this is the correct, minimal boundary to
        # prove a real MCP round trip happened (not just that the scripted
        # router's final text claimed one did). Assigning a plain function
        # directly onto an instance attribute shadows the bound method
        # without going through the descriptor protocol, so `_spy_call_tool`
        # receives exactly (tool_name, arguments), matching the real call
        # site in kenzo.mcp.adapter.build_tool_spec's `_call` closure.
        github_connection = runtime.mcp_client.connections["github"]
        real_call_tool = github_connection.call_tool
        captured_mcp_calls: list[dict] = []

        def _spy_call_tool(tool_name, arguments):
            result = real_call_tool(tool_name, arguments)
            captured_mcp_calls.append({"tool_name": tool_name, "arguments": dict(arguments), "result": result})
            return result

        github_connection.call_tool = _spy_call_tool

        result = runtime.controller.run("Fix issue #431")

        # -- Assertion 1: the real dynamic reasoning loop actually finished,
        # and AgentController's test gate (_build_final_summary) reads it as
        # a genuine success because the most recent scripted TEST verdict
        # passed.
        assert result["finished"] is True
        assert result["success"] is True

        # -- Assertion 2: the fixture's real file was really rewritten with
        # the genuine fix, not just described in scripted prose.
        fixed_content = (logout_bug_repo / _RELATIVE_JS_PATH).read_text(encoding="utf-8")
        assert "fetch('/api/auth/logout', { method: 'POST' })" in fixed_content
        assert "window.location.href = '/api/auth/post-logout';" in fixed_content
        assert "BUG (issue #431)" not in fixed_content  # the old buggy block is genuinely gone

        # -- Assertion 3: a real unified diff was captured for the edit, in
        # task_state.changes (built by AgentController._apply_updates from
        # the real edit_file ToolResult).
        matching_changes = [c for c in result["changes"] if c["file_path"] == _RELATIVE_JS_PATH]
        assert len(matching_changes) == 1
        assert "fetch('/api/auth/logout'" in matching_changes[0]["diff"]

        # -- Assertion 4: SEARCH's real search_files call actually ran against
        # the real fixture and found the real file (proves native repo search
        # genuinely executed, not just the MCP layer).
        assert _RELATIVE_JS_PATH in result["files_seen"]

        # -- Assertion 5: the two GitHub MCP calls really executed through the
        # real subprocess -- proven via the call_tool spy's captured real
        # arguments and real ToolResult, independent of the scripted text.
        get_issue_calls = [c for c in captured_mcp_calls if c["tool_name"] == "get_issue"]
        assert len(get_issue_calls) == 1
        assert get_issue_calls[0]["arguments"] == {"issue_number": 431}
        assert get_issue_calls[0]["result"].success is True
        # The exact real canned text from github_server.py's get_issue().
        assert "Issue #431: Logout button does nothing after OIDC redirect" in get_issue_calls[0]["result"].output
        assert "no outgoing request at all" in get_issue_calls[0]["result"].output

        create_pr_calls = [c for c in captured_mcp_calls if c["tool_name"] == "create_pull_request"]
        assert len(create_pr_calls) == 1
        assert create_pr_calls[0]["arguments"] == {"title": pr_title, "body": pr_body, "branch": pr_branch}
        assert create_pr_calls[0]["result"].success is True
        # The exact real canned confirmation text from github_server.py's
        # create_pull_request() -- proof this came back over the real
        # subprocess, not a hallucinated summary.
        assert f"Created pull request #245 on branch '{pr_branch}'" in create_pr_calls[0]["result"].output
        assert "https://github.com/example/repo/pull/245" in create_pr_calls[0]["result"].output

        # -- Assertion 6: the real PR branch name genuinely reached the
        # controller's own recorded transcript too (task_state.decisions),
        # corroborating that the tool result flowed all the way back into
        # the model turn that answered EDIT #2, not just that our external
        # spy happened to see the subprocess call in isolation.
        assert any(pr_branch in d for d in result["decisions"])

    finally:
        runtime.close()
