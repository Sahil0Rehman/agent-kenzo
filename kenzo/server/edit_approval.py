"""`EditApprovalToolRegistry`: Phase 7 IDE-mode edit gating.

CLI mode's `write_file`/`edit_file` (see `kenzo.tools.filesystem`) write
immediately and report a diff afterward -- correct and desired for a
terminal session where the diff IS the review step. IDE mode must not do
that: "Agent proposes edit -> VS Code diff viewer -> user accepts/rejects...
Do NOT directly overwrite."

This wrapper intercepts `write_file`/`edit_file` calls, computes the SAME
diff those tools would produce -- WITHOUT writing anything -- sends it to
`on_proposal` (wired to a WebSocket send in `kenzo.server.session`), and
BLOCKS the calling (synchronous agent) thread until a decision arrives via
`resolve_proposal` (called from the WebSocket receiver when an
`edit_decision` message comes in) or `approval_timeout` elapses. Only the
REAL `write_file`/`edit_file` tool -- called after acceptance, through the
full existing wrapping chain (`CheckpointingToolRegistry`/
`ReindexingToolRegistry`/`CachedToolRegistry`, all still layered underneath,
see `AgentController.__init__`) -- ever writes to disk on Kenzo's side; if
the client reports it already applied the edit itself (via VS Code's
`WorkspaceEdit`, the correct mechanism per the Phase 7 spec, so the change
integrates with VS Code's own undo/dirty-file/save handling), Kenzo trusts
that as the actual on-disk result and does not write a second time.

Preview computation necessarily duplicates a small amount of `write_file`/
`edit_file`'s own validation logic (path-traversal resolution, binary/UTF-8
checks, `edit_file`'s exact-unique-match rule) -- this is a deliberate,
documented tradeoff: `kenzo.tools.filesystem` has no dry-run mode, and
adding one there would touch Phase 1 code this project has otherwise kept
stable across every phase. The real tool's own validation still runs again
at actual-write time (after acceptance), so a file that changed between
preview and acceptance is caught then, not silently misapplied.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Optional

from kenzo.config.settings import PermissionsConfig
from kenzo.core.schemas import ToolResult
from kenzo.tools import filesystem as fs

DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0

OnProposal = Callable[[dict], None]


class _PendingProposal:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.decision: str = "reject"  # safe default if never explicitly set
        self.applied_by_client = False


class EditApprovalToolRegistry:
    """Wraps a tool registry so `write_file`/`edit_file` propose-then-wait
    instead of writing immediately. `on_proposal` is a plain public
    attribute (not fixed at construction) -- `kenzo.server.session
    .AgentSession` sets it to a session-specific callback before starting a
    task and clears it (`None`) afterward, exactly mirroring how
    `AgentController.on_event` is mutated per-task rather than fixed once at
    `AgentRuntime` construction (see that class's docstring for the
    single-runtime-per-repo rationale this relies on).

    `on_proposal is None` (the default -- true for every CLI-mode
    `AgentController`, since CLI never sets `ide_mode=True` and this class is
    never even constructed there) means IDE mode is not actually wired up
    yet; rather than hang forever or silently auto-apply, `write_file`/
    `edit_file` fail closed with a clear error in that state.
    """

    _GATED = {"write_file", "edit_file"}

    def __init__(
        self,
        wrapped: Any,
        repo_root: str,
        permissions: PermissionsConfig,
        approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        self._wrapped = wrapped
        self._repo_root = repo_root
        self._permissions = permissions
        self._approval_timeout = approval_timeout
        self.on_proposal: Optional[OnProposal] = None
        self._pending: dict[str, _PendingProposal] = {}
        self._lock = threading.Lock()

    def execute(self, name: str, arguments: dict) -> Any:
        if name not in self._GATED:
            return self._wrapped.execute(name, arguments)

        if not self._permissions.allow_write:
            # Mirror the real tool's own permission-denied ToolResult exactly
            # -- no point proposing an edit the user could never approve.
            return self._wrapped.execute(name, arguments)

        if self.on_proposal is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=(
                    "IDE edit-approval mode is active but no client is currently connected to "
                    "approve edits -- the file was NOT modified."
                ),
            )

        preview = self._preview(name, arguments)
        if isinstance(preview, ToolResult):
            return preview  # validation failed (not found, ambiguous match, binary, ...)

        proposal_id = str(uuid.uuid4())
        pending = _PendingProposal()
        with self._lock:
            self._pending[proposal_id] = pending

        self.on_proposal(
            {
                "proposal_id": proposal_id,
                "path": preview["path"],
                "action": preview["action"],
                "diff": preview["diff"],
                "new_content": preview["new_content"],
            }
        )

        answered = pending.event.wait(timeout=self._approval_timeout)
        with self._lock:
            self._pending.pop(proposal_id, None)

        if not answered:
            return ToolResult(
                tool_name=name,
                success=False,
                error=(
                    f"Edit proposal for '{preview['path']}' timed out after "
                    f"{self._approval_timeout:.0f}s waiting for user approval -- the file was NOT modified."
                ),
            )

        if pending.decision != "accept":
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"User rejected the proposed edit to '{preview['path']}' -- the file was NOT modified.",
            )

        if pending.applied_by_client:
            # The client already applied it (VS Code WorkspaceEdit) -- trust
            # that as the real on-disk result rather than writing again; see
            # module docstring.
            return ToolResult(
                tool_name=name,
                success=True,
                output=f"Edit to {preview['path']} applied by the connected client.",
                data={
                    "path": preview["path"],
                    "diff": preview["diff"],
                    "new_content": preview["new_content"],
                    "action": preview["action"],
                },
            )

        # Accepted, but the client wants Kenzo to perform the actual write
        # (e.g. a non-VS-Code client) -- go through the REAL tool, and thus
        # the full existing checkpoint/reindex/cache-invalidation chain still
        # layered underneath this wrapper.
        return self._wrapped.execute(name, arguments)

    def resolve_proposal(self, proposal_id: str, decision: str, applied_by_client: bool = False) -> bool:
        """Called from `kenzo.server`'s WebSocket receiver when an
        `edit_decision` message arrives. Returns `False` if `proposal_id` is
        unknown (already timed out, already resolved, or never existed -- a
        stale/duplicate client message), which the caller should treat as a
        harmless no-op, not an error."""
        with self._lock:
            pending = self._pending.get(proposal_id)
        if pending is None:
            return False
        pending.decision = decision
        pending.applied_by_client = applied_by_client
        pending.event.set()
        return True

    # ------------------------------------------------------------------
    # Preview computation (see module docstring for the duplication tradeoff)
    # ------------------------------------------------------------------

    def _preview(self, name: str, arguments: dict) -> dict[str, str] | ToolResult:
        if name == "write_file":
            return self._preview_write(arguments.get("path", ""), arguments.get("content", ""))
        return self._preview_edit(
            arguments.get("path", ""), arguments.get("old_text", ""), arguments.get("new_text", "")
        )

    def _preview_write(self, path: str, content: str) -> dict[str, str] | ToolResult:
        try:
            resolved = fs._resolve_within_repo(self._repo_root, path)
        except fs.PathTraversalError as exc:
            return ToolResult(tool_name="write_file", success=False, error=str(exc))
        if resolved.exists() and resolved.is_dir():
            return ToolResult(tool_name="write_file", success=False, error=f"'{path}' is a directory, not a file")

        old_content: Optional[str] = None
        created = not resolved.exists()
        if not created:
            try:
                old_raw = resolved.read_bytes()
                if not fs._is_binary(old_raw):
                    old_content, _decode_error = fs._decode_text_or_result(old_raw, path, "write_file")
            except OSError:
                old_content = None

        diff = fs._unified_diff(old_content or "", content, path)
        return {"path": path, "diff": diff, "new_content": content, "action": "create" if created else "edit"}

    def _preview_edit(self, path: str, old_text: str, new_text: str) -> dict[str, str] | ToolResult:
        try:
            resolved = fs._resolve_within_repo(self._repo_root, path)
        except fs.PathTraversalError as exc:
            return ToolResult(tool_name="edit_file", success=False, error=str(exc))
        if not resolved.exists():
            return ToolResult(tool_name="edit_file", success=False, error=f"file not found: {path}")
        if resolved.is_dir():
            return ToolResult(tool_name="edit_file", success=False, error=f"'{path}' is a directory, not a file")

        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            return ToolResult(tool_name="edit_file", success=False, error=f"could not read '{path}': {exc}")
        if fs._is_binary(raw):
            return ToolResult(
                tool_name="edit_file", success=False, error=f"'{path}' looks like a binary file and cannot be edited as text"
            )

        text, decode_error = fs._decode_text_or_result(raw, path, "edit_file")
        if decode_error is not None:
            return decode_error

        occurrences = text.count(old_text)
        if occurrences == 0:
            return ToolResult(tool_name="edit_file", success=False, error=f"old_text not found in '{path}'; no edit was made")
        if occurrences > 1:
            return ToolResult(
                tool_name="edit_file",
                success=False,
                error=(
                    f"old_text found {occurrences} times in '{path}'; edit_file requires an exact, "
                    "unique match. Include more surrounding context to disambiguate."
                ),
            )

        new_content = text.replace(old_text, new_text, 1)
        diff = fs._unified_diff(text, new_content, path)
        return {"path": path, "diff": diff, "new_content": new_content, "action": "edit"}

    # ------------------------------------------------------------------
    # Pass-through delegation, so this class duck-types as a ToolRegistry.
    # ------------------------------------------------------------------

    def to_openai_tools(self) -> list[dict]:
        return self._wrapped.to_openai_tools()

    def names(self) -> list[str]:
        return self._wrapped.names()

    def get(self, name: str):
        return self._wrapped.get(name)

    def list_specs(self):
        return self._wrapped.list_specs()

    def register(self, spec) -> None:
        self._wrapped.register(spec)

    def register_many(self, specs) -> None:
        self._wrapped.register_many(specs)

    def subset(self, names: list[str]) -> "EditApprovalToolRegistry":
        clone = EditApprovalToolRegistry(
            self._wrapped.subset(names), self._repo_root, self._permissions, self._approval_timeout
        )
        clone.on_proposal = self.on_proposal
        # Subsets share the SAME pending-proposal bookkeeping/lock as the
        # parent -- a proposal_id issued by one scoped view must still be
        # resolvable via whichever view's resolve_proposal is called (in
        # practice always the top-level AgentController.resolve_edit_proposal,
        # but this keeps the invariant correct regardless of caller).
        clone._pending = self._pending
        clone._lock = self._lock
        return clone
