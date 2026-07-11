"""Standalone BEFORE vs AFTER benchmark for Phase 9 "Parallel Agent Workers"
(`kenzo.workers.manager.WorkerManager.run_parallel`, wired into
`kenzo.core.controller.AgentController` via the DELEGATE action).

NOT a pytest test (see tests/test_workers_benchmark.py for the fast,
deterministic regression sibling of this script). This is a human-readable
report generator: it builds one synthetic "reasonably large" repository
encoding a plausible "authentication sometimes fails" bug, then drives a
REAL `AgentController` (real tools via `build_default_registry`, real
`RepositoryMemory`, real `KenzoConfig` -- only the model/router layer is
faked, following the exact scripted-router convention established by
tests/test_controller_scenarios.py's `_ControllerScriptedRouter`) through
two scripted scenarios against that SAME ground truth:

  BEFORE -- what the main agent had to do before Phase 9: three SEPARATE,
  SEQUENTIAL SEARCH/READ/ANALYZE actions (one per independent angle: the
  controller/service code, the existing test coverage, and the recent
  config change), each a real `run_tool_loop` call, then EDIT/TEST/FINISH.

  AFTER -- the same investigation collapsed into ONE DELEGATE action: three
  independent read-only worker sub-agents cover the same three angles
  CONCURRENTLY (a real `concurrent.futures.ThreadPoolExecutor`, not a
  simulation), producing equivalent information in one outer-loop iteration
  instead of three, then EDIT/TEST/FINISH identically.

Usage
-----
    C:\\Users\\sahil.r\\Ken\\.venv\\Scripts\\python.exe scripts\\benchmark_workers.py

=============================================================================
METHODOLOGY -- what this DOES and DOES NOT prove
=============================================================================
There is no real LLM API call anywhere in this script (same as the rest of
this project's test suite -- everything is mocked at the `ModelRouter
.complete`/`.complete_text` layer). Every "decision" and every "finding" the
fake router returns is pre-scripted by this file, not reasoned about by a
real model.

This DOES demonstrate, honestly and concretely:
  - the MECHANICAL effect of delegation: investigating 3 independent angles
    takes 3 outer-loop iterations serially (BEFORE) vs 1 outer-loop
    iteration (AFTER), for the exact same information gathered (same files
    discovered, same root cause identified, same fix applied);
  - REAL wall-clock overlap: the artificial per-model-call delay
    (`time.sleep(0.3)`, applied identically to both scenarios so the
    comparison stays apples-to-apples) is genuinely paid concurrently by 3
    OS threads in AFTER, via a real `ThreadPoolExecutor` -- not simulated by
    dividing a number after the fact.

This does NOT demonstrate:
  - anything about a real model's reasoning QUALITY, whether delegation
    produces BETTER investigations, or whether a real model would even
    choose to delegate here -- the "decisions" are hardcoded by this script,
    not chosen by a model;
  - real API latency -- 0.3s/call is an arbitrary stand-in chosen only to
    make wall-clock overlap visible and measurable, not a claim about any
    provider's real response time;
  - anything about token cost, context-window usage, or worker output
    quality/accuracy in a real, non-scripted setting.

In short: this is a controlled, scripted measurement of the SHAPE of the
execution graph (serial vs parallel round-trips) that Phase 9 changes, not a
benchmark of model intelligence.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Allow running as `python scripts/benchmark_workers.py` from the repo root
# without requiring kenzo to already be on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kenzo.config.settings import KenzoConfig, PermissionsConfig  # noqa: E402
from kenzo.core.controller import AgentController, ControllerAction  # noqa: E402
from kenzo.memory.repository import RepositoryMemory  # noqa: E402
from kenzo.tools.registry import build_default_registry  # noqa: E402

# Per-model-call artificial delay. Applied identically inside BOTH scenarios'
# router, so the comparison is apples-to-apples: the only thing that differs
# is whether calls happen serially or with real thread-level overlap.
CALL_DELAY_SECONDS = 0.3


# =============================================================================
# Fake litellm-shaped response scaffolding -- same pattern as
# tests/test_controller_scenarios.py's `_ControllerScriptedRouter`
# (`_FakeMessage`/`_FakeChoice`/`_FakeModelResponse`/`_FakeFunction`/
# `_FakeToolCall`, `_decision`/`_tool_call_response`/`_final_text` helpers).
# =============================================================================


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


# =============================================================================
# Thread-safe scripted router.
#
# Thread-safety note (the same trap that bit tests/test_controller_scenarios
# .py's author -- see this project's kenzo.core.tool_loop.run_tool_loop):
# `run_tool_loop` never passes any caller-identifying info to
# `router.complete(task_type, messages, tools=..., tool_choice=...)` beyond
# `messages`/`tools`/`tool_choice`. A `decide_next_action`-forced call is
# identifiable via `tool_choice`; `AgentController._propose_worker_tasks`
# (DELEGATE -> WorkerTasks) calls `router.complete_text(...)`, which passes
# NO `tools` kwarg at all (`tools is None` there specifically, vs a list for
# every `run_tool_loop`-driven call, worker or not). In the AFTER scenario 3
# workers call this router CONCURRENTLY from real OS threads spawned by a
# `ThreadPoolExecutor` -- there is no strict FIFO order across those 3 calls,
# so worker responses are NOT served from one shared queue; instead each
# call's messages are inspected for a worker-specific marker embedded in
# that worker's own task description (see `_ANGLE_MARKER`), and served from
# a queue private to THAT worker. All shared mutable state (the queues, the
# call counter) is protected by one `threading.Lock`.
# =============================================================================


class _BenchmarkRouter:
    def __init__(
        self,
        decisions: list,
        actions: list,
        delegate_planning: Optional[list] = None,
        worker_scripts: Optional[dict[str, list]] = None,
        call_delay: float = CALL_DELAY_SECONDS,
    ) -> None:
        self._decisions = list(decisions)
        self._actions = list(actions)
        self._delegate_planning = list(delegate_planning or [])
        self._worker_scripts = {k: list(v) for k, v in (worker_scripts or {}).items()}
        self._call_delay = call_delay
        self._lock = threading.Lock()
        self.call_count = 0
        self.max_concurrent_calls = 0
        self._in_flight = 0

    def _worker_key_for(self, combined_content: str) -> str:
        for key in self._worker_scripts:
            if key in combined_content:
                return key
        raise AssertionError(
            f"could not determine which worker this call belongs to from content: {combined_content[:200]!r}"
        )

    def complete(self, task_type, messages, tools=None, tool_choice=None, **kwargs):
        # Artificial per-call latency, deliberately OUTSIDE the lock so
        # concurrent worker calls genuinely overlap in wall-clock time
        # instead of being serialized by this fake -- that overlap is
        # exactly the mechanism under measurement.
        with self._lock:
            self._in_flight += 1
            self.max_concurrent_calls = max(self.max_concurrent_calls, self._in_flight)
        if self._call_delay:
            time.sleep(self._call_delay)
        with self._lock:
            self._in_flight -= 1
            self.call_count += 1

        if _is_decide_call(tool_choice):
            with self._lock:
                if not self._decisions:
                    raise AssertionError("ran out of scripted decisions")
                return self._decisions.pop(0)

        combined = " ".join(str(m.get("content", "")) for m in messages)
        if "Investigation task:" in combined:
            key = self._worker_key_for(combined)
            with self._lock:
                queue = self._worker_scripts.get(key)
                if not queue:
                    raise AssertionError(f"ran out of scripted worker responses for {key!r}")
                return queue.pop(0)

        if tools is None:
            # `AgentController._propose_worker_tasks`'s `complete_text(...)`
            # call -- see the class docstring above.
            with self._lock:
                if not self._delegate_planning:
                    raise AssertionError("ran out of scripted delegate-planning responses")
                return self._delegate_planning.pop(0)

        with self._lock:
            if not self._actions:
                raise AssertionError("ran out of scripted action responses")
            return self._actions.pop(0)

    def complete_text(self, task_type, system_prompt, user_prompt, **kwargs):
        """Mirrors `kenzo.models.router.ModelRouter.complete_text` exactly:
        builds a plain system/user message pair and delegates to `complete`
        with no `tools`/`tool_choice` -- this is the call
        `_propose_worker_tasks` makes."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.complete(task_type, messages, **kwargs)
        content = response.choices[0].message.content
        return content or ""


# =============================================================================
# Synthetic repository: a plausible "authentication sometimes fails" bug.
#
# Ground truth: `auth/oidc_config.py` recently extended TOKEN_EXPIRY_SECONDS
# from 300s to 3600s (2026-06-01, "reduce forced re-logins"). `auth
# /session_manager.py` duplicates that number in its OWN hardcoded
# `_REFRESH_WINDOW_SECONDS`, which was never updated -- so SessionManager
# still enforces the OLD 300s window and incorrectly forces a relogin for
# any session between 300s and 3600s old, even though AuthService (which
# correctly reads the current config) would still consider that session's
# access token perfectly valid. This is genuinely evidenced across the
# controller, the service, the config file, AND an existing test that
# exercises exactly this path -- the three angles the BEFORE/AFTER scenarios
# both investigate.
# =============================================================================

_ANGLE_MARKER = {
    "CODE": "[ANGLE:CODE]",
    "TEST": "[ANGLE:TEST]",
    "CONFIG": "[ANGLE:CONFIG]",
}

_OIDC_CONFIG_PY = '''"""OIDC / SSO configuration for the authentication subsystem."""

# 2026-06-01: extended from 300 -> 3600 seconds to reduce how often active
# users get forced back through a full re-login (see CHANGELOG.md).
TOKEN_EXPIRY_SECONDS = 3600

ISSUER = "https://auth.example.com"
CLIENT_ID = "example-client"
REDIRECT_URI = "https://app.example.com/callback"
'''

_AUTH_SERVICE_PY = '''"""AuthService: issues and validates access tokens."""

from auth.oidc_config import TOKEN_EXPIRY_SECONDS


class AuthService:
    """Talks to the OIDC provider to issue/validate tokens."""

    def __init__(self, user_repository):
        self.user_repository = user_repository

    def authenticate(self, username, password):
        user = self.user_repository.find_by_username(username)
        if user is None:
            return None
        return self._issue_token(user)

    def _issue_token(self, user):
        return {"access_token": f"tok-{user['id']}", "expires_in": TOKEN_EXPIRY_SECONDS}

    def validate_token(self, token, issued_at, now):
        """A token is valid only while now - issued_at < TOKEN_EXPIRY_SECONDS."""
        return (now - issued_at) < TOKEN_EXPIRY_SECONDS
'''

# THE BUG lives here: `_REFRESH_WINDOW_SECONDS` duplicates
# oidc_config.TOKEN_EXPIRY_SECONDS instead of importing it, and was never
# updated when that value was extended from 300 to 3600.
_SESSION_MANAGER_PY_BUGGY = '''"""SessionManager: decides whether an existing session is still eligible for
a silent token refresh, or whether the user must be sent back through the
full login flow.
"""

# BUG: hardcoded to the OLD TOKEN_EXPIRY_SECONDS (300s) from before the
# 2026-06-01 change that extended it to 3600s in auth/oidc_config.py. Never
# updated to match, so SessionManager still enforces the OLD, stricter
# window: sessions between 300s and 3600s old are STILL VALID per
# AuthService.validate_token() (which correctly reads the current config
# value) but SessionManager.is_refreshable() rejects them anyway and forces
# a relogin. This is the intermittent "authentication sometimes fails" bug
# -- any user whose session is refreshed more than 5 minutes (but less than
# 60 minutes) after issuance gets incorrectly logged out.
_REFRESH_WINDOW_SECONDS = 300


class SessionManager:
    """Gatekeeper for silent session refresh, sitting between
    AuthController and AuthService."""

    def __init__(self, auth_service):
        self.auth_service = auth_service

    def is_refreshable(self, session, now):
        return (now - session["issued_at"]) < _REFRESH_WINDOW_SECONDS

    def refresh(self, session, now):
        if not self.is_refreshable(session, now):
            return {"status": "relogin_required"}
        if not self.auth_service.validate_token(
            session["access_token"], session["issued_at"], now
        ):
            return {"status": "refresh_failed_token_expired"}
        return {"status": "refreshed"}
'''

_AUTH_CONTROLLER_PY = '''"""AuthController: HTTP-facing entry points for login/refresh."""

from auth.session_manager import SessionManager


class AuthController:
    def __init__(self, auth_service):
        self.auth_service = auth_service
        self.session_manager = SessionManager(auth_service)

    def login(self, username, password):
        token = self.auth_service.authenticate(username, password)
        if token is None:
            return {"status": "invalid_credentials"}
        return {"status": "ok", "token": token}

    def refresh(self, session, now):
        return self.session_manager.refresh(session, now)
'''

_TEST_SESSION_REFRESH_PY = '''"""Regression coverage for AuthController.refresh(): a session younger than
TOKEN_EXPIRY_SECONDS must refresh successfully, not be forced to relogin.
"""

from auth.auth_controller import AuthController
from auth.auth_service import AuthService


class _FakeUserRepository:
    def find_by_username(self, username):
        return {"id": 1, "username": username}


def test_refresh_succeeds_for_session_within_token_expiry():
    controller = AuthController(AuthService(_FakeUserRepository()))
    session = {"access_token": "tok-1", "issued_at": 0}
    now = 350  # well within TOKEN_EXPIRY_SECONDS (3600s), but past the
    # SessionManager bug's stale 300s window.

    result = controller.refresh(session, now)

    assert result["status"] == "refreshed", (
        "a session younger than TOKEN_EXPIRY_SECONDS must refresh "
        f"successfully, not report {result['status']!r}"
    )
'''

_TEST_AUTH_CONTROLLER_PY = '''from auth.auth_controller import AuthController
from auth.auth_service import AuthService


class _FakeUserRepository:
    def find_by_username(self, username):
        if username == "known-user":
            return {"id": 7, "username": username}
        return None


def test_login_returns_token_for_known_user():
    controller = AuthController(AuthService(_FakeUserRepository()))
    result = controller.login("known-user", "any-password")
    assert result["status"] == "ok"
    assert "token" in result


def test_login_rejects_unknown_user():
    controller = AuthController(AuthService(_FakeUserRepository()))
    result = controller.login("nobody", "any-password")
    assert result["status"] == "invalid_credentials"
'''

_TEST_AUTH_SERVICE_PY = '''from auth.auth_service import AuthService
from auth.oidc_config import TOKEN_EXPIRY_SECONDS


class _FakeUserRepository:
    def find_by_username(self, username):
        return {"id": 1, "username": username}


def test_issue_token_reports_current_expiry():
    service = AuthService(_FakeUserRepository())
    token = service.authenticate("someone", "pw")
    assert token["expires_in"] == TOKEN_EXPIRY_SECONDS


def test_validate_token_true_within_expiry():
    service = AuthService(_FakeUserRepository())
    assert service.validate_token("tok", issued_at=0, now=TOKEN_EXPIRY_SECONDS - 1) is True


def test_validate_token_false_after_expiry():
    service = AuthService(_FakeUserRepository())
    assert service.validate_token("tok", issued_at=0, now=TOKEN_EXPIRY_SECONDS + 1) is False
'''

# Unrelated filler modules -- genuine additional files/layers so this is a
# reasonably large repo, not a relabeled 2-file toy.
_FILLER_FILES: dict[str, str] = {
    "billing/invoice_service.py": (
        '"""InvoiceService: generates and stores customer invoices. Unrelated '
        'to authentication -- repo filler."""\n\n\n'
        "class InvoiceService:\n"
        "    def __init__(self, payment_gateway):\n"
        "        self.payment_gateway = payment_gateway\n\n"
        "    def create_invoice(self, customer_id, line_items):\n"
        "        total = sum(item[\"amount\"] for item in line_items)\n"
        "        return {\"customer_id\": customer_id, \"total\": total, \"status\": \"draft\"}\n\n"
        "    def finalize_invoice(self, invoice):\n"
        "        invoice[\"status\"] = \"finalized\"\n"
        "        return invoice\n"
    ),
    "billing/payment_gateway.py": (
        '"""PaymentGateway: a stub third-party payment processor client. Repo filler."""\n\n\n'
        "class PaymentGateway:\n"
        "    def charge(self, amount, card_token):\n"
        "        if amount <= 0:\n"
        "            raise ValueError(\"amount must be positive\")\n"
        "        return {\"charge_id\": f\"ch_{card_token[-4:]}\", \"amount\": amount, \"status\": \"succeeded\"}\n"
    ),
    "notifications/email_sender.py": (
        '"""EmailSender: repo filler, unrelated to authentication."""\n\n\n'
        "class EmailSender:\n"
        "    def __init__(self, smtp_host):\n"
        "        self.smtp_host = smtp_host\n\n"
        "    def send(self, to_address, subject, body):\n"
        "        return {\"to\": to_address, \"subject\": subject, \"delivered\": True}\n"
    ),
    "notifications/sms_sender.py": (
        '"""SmsSender: repo filler, unrelated to authentication."""\n\n\n'
        "class SmsSender:\n"
        "    def send(self, phone_number, message):\n"
        "        return {\"phone_number\": phone_number, \"delivered\": True, \"length\": len(message)}\n"
    ),
    "catalog/product_catalog.py": (
        '"""ProductCatalog: repo filler, unrelated to authentication."""\n\n\n'
        "class ProductCatalog:\n"
        "    def __init__(self):\n"
        "        self._products = {}\n\n"
        "    def add_product(self, sku, name, price):\n"
        "        self._products[sku] = {\"name\": name, \"price\": price}\n\n"
        "    def get_product(self, sku):\n"
        "        return self._products.get(sku)\n"
    ),
    "catalog/inventory_manager.py": (
        '"""InventoryManager: repo filler, unrelated to authentication."""\n\n\n'
        "class InventoryManager:\n"
        "    def __init__(self):\n"
        "        self._stock = {}\n\n"
        "    def reserve(self, sku, quantity):\n"
        "        available = self._stock.get(sku, 0)\n"
        "        if available < quantity:\n"
        "            return False\n"
        "        self._stock[sku] = available - quantity\n"
        "        return True\n\n"
        "    def restock(self, sku, quantity):\n"
        "        self._stock[sku] = self._stock.get(sku, 0) + quantity\n"
    ),
    "utils/date_utils.py": (
        '"""date_utils: repo filler, unrelated to authentication."""\n\n'
        "from datetime import datetime, timezone\n\n\n"
        "def utc_now_iso() -> str:\n"
        "    return datetime.now(timezone.utc).isoformat()\n\n\n"
        "def seconds_between(a: float, b: float) -> float:\n"
        "    return abs(b - a)\n"
    ),
    "utils/string_utils.py": (
        '"""string_utils: repo filler, unrelated to authentication."""\n\n\n'
        "def slugify(value: str) -> str:\n"
        "    return \"-\".join(value.lower().split())\n\n\n"
        "def truncate(value: str, max_len: int) -> str:\n"
        "    return value if len(value) <= max_len else value[: max_len - 3] + \"...\"\n"
    ),
    "utils/logging_utils.py": (
        '"""logging_utils: repo filler, unrelated to authentication."""\n\n\n'
        "def format_log_line(level: str, message: str) -> str:\n"
        "    return f\"[{level.upper()}] {message}\"\n"
    ),
    "api/health_check.py": (
        '"""health_check: repo filler, unrelated to authentication."""\n\n\n'
        "def health_check() -> dict:\n"
        "    return {\"status\": \"ok\"}\n"
    ),
    "api/middleware.py": (
        '"""middleware: repo filler, unrelated to authentication."""\n\n\n'
        "class RequestLoggingMiddleware:\n"
        "    def __call__(self, request, next_handler):\n"
        "        return next_handler(request)\n"
    ),
    "README.md": (
        "# Example Service\n\n"
        "A small example service used to exercise Kenzo's Phase 9 parallel "
        "worker benchmark. Includes an authentication subsystem (`auth/`), "
        "plus billing, notifications, catalog, and utility modules.\n"
    ),
    "CHANGELOG.md": (
        "# Changelog\n\n"
        "## 2026-06-01\n"
        "- Extended `TOKEN_EXPIRY_SECONDS` in `auth/oidc_config.py` from 300 "
        "to 3600 seconds to reduce how often active users are forced back "
        "through a full re-login.\n"
    ),
    "config/app_config.yaml": ("service_name: example-service\nenvironment: production\nlog_level: info\n"),
}

# Every relative path that gets an empty __init__.py (kept out of the big
# content dict above purely to reduce visual noise).
_INIT_FILES = [
    "auth/__init__.py",
    "tests/__init__.py",
    "tests/auth/__init__.py",
    "billing/__init__.py",
    "notifications/__init__.py",
    "catalog/__init__.py",
    "utils/__init__.py",
    "api/__init__.py",
]

_AUTH_FILES: dict[str, str] = {
    "auth/oidc_config.py": _OIDC_CONFIG_PY,
    "auth/auth_service.py": _AUTH_SERVICE_PY,
    "auth/session_manager.py": _SESSION_MANAGER_PY_BUGGY,
    "auth/auth_controller.py": _AUTH_CONTROLLER_PY,
    "tests/auth/test_session_refresh.py": _TEST_SESSION_REFRESH_PY,
    "tests/auth/test_auth_controller.py": _TEST_AUTH_CONTROLLER_PY,
    "tests/auth/test_auth_service.py": _TEST_AUTH_SERVICE_PY,
}


def build_synthetic_repo(root: Path) -> Path:
    """Build one physical copy of the synthetic "authentication sometimes
    fails" repository under `root`. Called TWICE (once per scenario, into
    two separate tmp directories) rather than shared between BEFORE and
    AFTER: both scenarios apply a real EDIT to the same file, so sharing one
    physical copy would let whichever scenario runs first silently "fix" the
    ground truth out from under the second -- defeating the "same ground
    truth" comparison. Both copies are built from this exact same function/
    content, so they are identical at the moment each scenario starts.
    """
    root.mkdir(parents=True, exist_ok=True)
    all_files: dict[str, str] = {**_AUTH_FILES, **_FILLER_FILES}
    for rel_path, content in all_files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for rel_path in _INIT_FILES:
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    return root


def _count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


# =============================================================================
# Scripted scenarios
# =============================================================================

_FIX_OLD_TEXT = "_REFRESH_WINDOW_SECONDS = 300"
_FIX_NEW_TEXT = "from auth.oidc_config import TOKEN_EXPIRY_SECONDS\n\n_REFRESH_WINDOW_SECONDS = TOKEN_EXPIRY_SECONDS"

_TEST_PASS_JSON = json.dumps(
    {
        "passed": True,
        "summary": "test_refresh_succeeds_for_session_within_token_expiry now passes",
        "command": "pytest tests/auth/test_session_refresh.py",
        "raw_output": "",
    }
)


def _before_scenario(router_kwargs_only: bool = False):
    """Three separate, sequential SEARCH/READ/ANALYZE actions -- one per
    angle -- then EDIT/TEST/FINISH. This is what the main agent had to do
    before Phase 9: every angle is its own full outer-loop iteration."""
    decisions = [
        _decision(
            ControllerAction.SEARCH,
            "Angle 1/3: look at the controller and session-management code that implements refresh.",
            "auth/auth_controller.py, auth/session_manager.py",
        ),
        _decision(
            ControllerAction.READ,
            "Angle 2/3: check the existing test coverage for session refresh.",
            "tests/auth/test_session_refresh.py",
        ),
        _decision(
            ControllerAction.ANALYZE,
            "Angle 3/3: check whether a recent config change is inconsistent with the code.",
            "auth/oidc_config.py",
        ),
        _decision(
            ControllerAction.EDIT,
            "Apply the fix: make SessionManager read TOKEN_EXPIRY_SECONDS from oidc_config "
            "instead of its stale hardcoded refresh window.",
            "auth/session_manager.py",
        ),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(
            ControllerAction.FINISH,
            "Session refresh now matches the current token expiry config; the regression test passes.",
        ),
    ]

    actions = [
        # Angle 1: controller/service code. SEARCH's tool subset (see
        # ACTION_TOOLS[ControllerAction.SEARCH] in kenzo.core.controller)
        # deliberately does NOT include read_file (that's READ's/ANALYZE's
        # job) -- only search_files/find_symbol/etc -- so this angle uses a
        # REAL search_files call against the REAL repo, whose genuine
        # matches (both auth_controller.py and session_manager.py mention
        # "SessionManager") are what actually populate files_seen below,
        # not a scripted path.
        _tool_call_response("call-1a", "search_files", {"pattern": "SessionManager"}),
        _final_text(
            "search_files('SessionManager') surfaced auth/auth_controller.py (constructs a "
            "SessionManager) and auth/session_manager.py (defines it) -- AuthController.refresh() "
            "delegates to SessionManager.refresh(), which checks is_refreshable() using a LOCAL "
            "_REFRESH_WINDOW_SECONDS=300 before ever consulting AuthService's own token validation."
        ),
        # Angle 2: the test file.
        _tool_call_response("call-2a", "read_file", {"path": "tests/auth/test_session_refresh.py"}),
        _final_text(
            "test_refresh_succeeds_for_session_within_token_expiry expects a 350s-old session to "
            "refresh successfully, since 350 < TOKEN_EXPIRY_SECONDS."
        ),
        # Angle 3: the recent config change.
        _tool_call_response("call-3a", "read_file", {"path": "auth/oidc_config.py"}),
        _final_text(
            "auth/oidc_config.py shows TOKEN_EXPIRY_SECONDS was extended from 300 to 3600 on "
            "2026-06-01, but session_manager.py's _REFRESH_WINDOW_SECONDS was never updated to "
            "match -- confirmed root cause."
        ),
        # EDIT.
        _tool_call_response(
            "call-edit",
            "edit_file",
            {"path": "auth/session_manager.py", "old_text": _FIX_OLD_TEXT, "new_text": _FIX_NEW_TEXT},
        ),
        _final_text(
            "Updated SessionManager to read TOKEN_EXPIRY_SECONDS from auth/oidc_config.py instead "
            "of a stale hardcoded window."
        ),
        # TEST.
        _final_text(_TEST_PASS_JSON),
    ]

    return decisions, actions


def _after_scenario():
    """One DELEGATE action covering all 3 angles at once (3 concurrent
    workers), then EDIT/TEST/FINISH identically to the BEFORE scenario."""
    decisions = [
        _decision(
            ControllerAction.DELEGATE,
            "Three independent angles need checking at once: the controller/service code, the "
            "existing test coverage, and whether a recent config change is inconsistent with it.",
            "why session refresh sometimes fails",
        ),
        _decision(
            ControllerAction.EDIT,
            "Apply the fix: make SessionManager read TOKEN_EXPIRY_SECONDS from oidc_config "
            "instead of its stale hardcoded refresh window.",
            "auth/session_manager.py",
        ),
        _decision(ControllerAction.TEST, "Validate the fix."),
        _decision(
            ControllerAction.FINISH,
            "Session refresh now matches the current token expiry config; the regression test passes.",
        ),
    ]

    # The single non-tool-calling `complete_text` call that turns the
    # DELEGATE decision into 3 WorkerTasks (see
    # AgentController._propose_worker_tasks). Each description embeds a
    # unique `[ANGLE:...]` marker so the concurrent worker calls below can
    # be routed correctly regardless of thread interleaving (see
    # `_BenchmarkRouter`).
    delegate_planning = [
        _final_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "worker_type": "code_research",
                            "description": (
                                "Investigate auth/auth_controller.py and auth/session_manager.py to "
                                "trace exactly how session refresh decides whether to allow a silent "
                                f"refresh vs forcing relogin. {_ANGLE_MARKER['CODE']}"
                            ),
                        },
                        {
                            "worker_type": "test",
                            "description": (
                                "Find and read the test(s) covering session refresh behavior under "
                                f"tests/auth/, and report what they assert. {_ANGLE_MARKER['TEST']}"
                            ),
                        },
                        {
                            "worker_type": "bug_analysis",
                            "description": (
                                "Read auth/oidc_config.py and check whether any recently-changed "
                                "configuration constant (e.g. token expiry) is duplicated or "
                                "hardcoded elsewhere in the auth code in a way that could now be "
                                f"inconsistent. {_ANGLE_MARKER['CONFIG']}"
                            ),
                        },
                    ]
                }
            )
        )
    ]

    worker_scripts = {
        _ANGLE_MARKER["CODE"]: [
            _tool_call_response("w-code-1", "read_file", {"path": "auth/auth_controller.py"}),
            _tool_call_response("w-code-2", "read_file", {"path": "auth/session_manager.py"}),
            _final_text(
                json.dumps(
                    {
                        "findings": (
                            "AuthController.refresh() delegates to SessionManager.refresh(), which "
                            "checks is_refreshable() using a LOCAL _REFRESH_WINDOW_SECONDS=300 "
                            "before ever consulting AuthService's own token validation."
                        ),
                        "confidence": 0.8,
                        "files": ["auth/auth_controller.py", "auth/session_manager.py"],
                        "recommendations": [
                            "Compare _REFRESH_WINDOW_SECONDS in session_manager.py against "
                            "auth/oidc_config.py's TOKEN_EXPIRY_SECONDS."
                        ],
                    }
                )
            ),
        ],
        _ANGLE_MARKER["TEST"]: [
            _tool_call_response("w-test-1", "read_file", {"path": "tests/auth/test_session_refresh.py"}),
            _final_text(
                json.dumps(
                    {
                        "findings": (
                            "test_refresh_succeeds_for_session_within_token_expiry expects a "
                            "350s-old session to refresh successfully, since that is within "
                            "TOKEN_EXPIRY_SECONDS."
                        ),
                        "confidence": 0.9,
                        "files": ["tests/auth/test_session_refresh.py"],
                        "recommendations": ["Run this test after the fix to confirm the windows match."],
                    }
                )
            ),
        ],
        _ANGLE_MARKER["CONFIG"]: [
            _tool_call_response("w-config-1", "read_file", {"path": "auth/oidc_config.py"}),
            _final_text(
                json.dumps(
                    {
                        "findings": (
                            "auth/oidc_config.py shows TOKEN_EXPIRY_SECONDS was extended from 300 "
                            "to 3600 on 2026-06-01. session_manager.py was not updated to match."
                        ),
                        "confidence": 0.85,
                        "files": ["auth/oidc_config.py"],
                        "recommendations": [
                            "Update SessionManager to read TOKEN_EXPIRY_SECONDS from oidc_config.py "
                            "instead of a hardcoded duplicate."
                        ],
                    }
                )
            ),
        ],
    }

    # Post-DELEGATE actions: identical EDIT/TEST scripting to BEFORE.
    actions = [
        _tool_call_response(
            "call-edit",
            "edit_file",
            {"path": "auth/session_manager.py", "old_text": _FIX_OLD_TEXT, "new_text": _FIX_NEW_TEXT},
        ),
        _final_text(
            "Updated SessionManager to read TOKEN_EXPIRY_SECONDS from auth/oidc_config.py instead "
            "of a stale hardcoded window."
        ),
        _final_text(_TEST_PASS_JSON),
    ]

    return decisions, actions, delegate_planning, worker_scripts


# =============================================================================
# Runner
# =============================================================================


def _real_controller(repo_root: Path, router: Any) -> AgentController:
    permissions = PermissionsConfig(allow_write=True)
    tools = build_default_registry(str(repo_root), permissions)
    memory = RepositoryMemory(str(repo_root))
    config = KenzoConfig()
    return AgentController(router, tools, memory, config)


def run_before(repo_root: Path) -> dict:
    decisions, actions = _before_scenario()
    router = _BenchmarkRouter(decisions, actions)
    controller = _real_controller(repo_root, router)

    start = time.monotonic()
    result = controller.run("Find why authentication sometimes fails")
    elapsed = time.monotonic() - start

    result["_elapsed_seconds"] = elapsed
    result["_router_call_count"] = router.call_count
    result["_max_concurrent_calls"] = router.max_concurrent_calls
    return result


def run_after(repo_root: Path) -> dict:
    decisions, actions, delegate_planning, worker_scripts = _after_scenario()
    router = _BenchmarkRouter(decisions, actions, delegate_planning, worker_scripts)
    controller = _real_controller(repo_root, router)

    start = time.monotonic()
    result = controller.run("Find why authentication sometimes fails")
    elapsed = time.monotonic() - start

    result["_elapsed_seconds"] = elapsed
    result["_router_call_count"] = router.call_count
    result["_max_concurrent_calls"] = router.max_concurrent_calls
    return result


def _iterations(result: dict) -> int:
    return result.get("iteration", len(result["task_state"]["decisions"]))


def _print_report(before: dict, after: dict, file_count: int) -> None:
    b_files = len(before["files_seen"])
    a_files = len(after["files_seen"])
    b_iter = _iterations(before)
    a_iter = _iterations(after)
    b_time = before["_elapsed_seconds"]
    a_time = after["_elapsed_seconds"]

    print("=" * 78)
    print('Kenzo Phase 9 "Parallel Agent Workers" -- BEFORE vs AFTER benchmark')
    print("=" * 78)
    print('Task: "Find why authentication sometimes fails"')
    print(f"Synthetic repo: {file_count} files across auth/billing/notifications/catalog/utils/api")
    print()
    print(f"{'Scenario':<28}{'Files discovered':<20}{'Iterations':<14}{'Wall-clock (s)':<16}")
    print("-" * 78)
    print(f"{'BEFORE (sequential)':<28}{b_files:<20}{b_iter:<14}{b_time:<16.2f}")
    print(f"{'AFTER (delegated)':<28}{a_files:<20}{a_iter:<14}{a_time:<16.2f}")
    print("-" * 78)

    iter_pct = (1 - a_iter / b_iter) * 100 if b_iter else 0.0
    time_pct = (1 - a_time / b_time) * 100 if b_time else 0.0
    print(f"Iteration reduction:   {b_iter} -> {a_iter}   ({iter_pct:.1f}% fewer outer-loop iterations)")
    print(f"Wall-clock reduction:  {b_time:.2f}s -> {a_time:.2f}s   ({time_pct:.1f}% faster)")
    print(
        f"Router calls made:     BEFORE={before['_router_call_count']} (max {before['_max_concurrent_calls']} "
        f"concurrent)   AFTER={after['_router_call_count']} (max {after['_max_concurrent_calls']} concurrent)"
    )
    print()
    print(f"Files discovered (BEFORE): {sorted(before['files_seen'])}")
    print(f"Files discovered (AFTER):  {sorted(after['files_seen'])}")
    print(f"Files modified (BEFORE):   {before['files_modified']}")
    print(f"Files modified (AFTER):    {after['files_modified']}")
    print(f"Test passed (BEFORE):      {before['success']}")
    print(f"Test passed (AFTER):       {after['success']}")
    print()
    print("-" * 78)
    print("METHODOLOGY -- what this DOES and DOES NOT prove")
    print("-" * 78)
    print(
        "No real LLM API call is made anywhere above -- every decision and every worker\n"
        "finding is pre-scripted (see this script's module docstring and\n"
        "tests/test_controller_scenarios.py's established fake-router convention, which\n"
        "this script follows exactly). This DOES demonstrate the MECHANICAL effect of\n"
        "delegation under a controlled scenario: 3 independent investigation angles took\n"
        f"{b_iter} serial outer-loop iterations before Phase 9 vs {a_iter} now (1 DELEGATE\n"
        "covering all 3 concurrently via a real ThreadPoolExecutor), and the wall-clock\n"
        "time above reflects REAL thread-level overlap of the identical artificial\n"
        f"per-call delay ({CALL_DELAY_SECONDS}s), not a simulated/divided number. This does NOT\n"
        "prove anything about a real model's reasoning quality, whether a real model would\n"
        "choose to delegate here, or real API latency -- there is no real model in this run."
    )


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="kenzo_workers_bench_"))
    print(f"Fixture directory: {tmp_root}")
    try:
        # Two independent physical copies of the SAME synthetic repo/ground
        # truth -- see build_synthetic_repo's docstring for why BEFORE and
        # AFTER cannot safely share one copy (both apply a real EDIT).
        before_root = build_synthetic_repo(tmp_root / "repo_before")
        after_root = build_synthetic_repo(tmp_root / "repo_after")
        file_count = _count_files(before_root)
        print(f"Built synthetic repo: {file_count} files (identical copy for each scenario).\n")

        print("Running BEFORE scenario (3 sequential SEARCH/READ/ANALYZE actions)...")
        before = run_before(before_root)
        print(f"  done in {before['_elapsed_seconds']:.2f}s, {_iterations(before)} iteration(s).\n")

        print("Running AFTER scenario (1 DELEGATE action, 3 concurrent workers)...")
        after = run_after(after_root)
        print(f"  done in {after['_elapsed_seconds']:.2f}s, {_iterations(after)} iteration(s).\n")

        _print_report(before, after, file_count)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\nCleaned up fixture directory: {tmp_root}")


if __name__ == "__main__":
    main()
