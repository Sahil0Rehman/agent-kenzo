"""Multi-repo workspace: lets a single Kenzo run see (read-only) and,
if explicitly told, edit more than one repository at once.

Built for a concrete real-world shape this project's own dogfooding
surfaced: the same logical service checked out multiple times across
sprint/release branches (e.g. `Sprint53/CIDSaas/CIDSaas` and
`june-release-2026/CIDSaas/CIDSaas`), plus a build-time dependency on an
entirely different service/branch (e.g. entity classes living in
`Sprint52/CIDBLayer/CIDBLayerSvc`). The workflow this exists to support:
"fix this in the new branch, but check how the old branch handled it first"
and, less often, "port this same fix into both branches."

Deliberately NOT a persistent config file (`~/.kenzo/workspaces/*.yaml` or
similar) -- the repos in play are named fresh on the command line for each
invocation (`--see`/`--also-edit`); Kenzo reasons about how they relate from
the request and what it reads, the same way it already reasons about a
single repo, rather than requiring a workspace definition to be maintained
up front.

Safety model:
- The PRIMARY repo (`--repo`, unchanged from every prior phase) is always
  writable, exactly as today.
- `--see` repos are readable but NEVER writable, full stop -- there is no
  code path that can make one writable short of also being named in
  `--also-edit`.
- `--also-edit` repos are both readable and writable, and (critically) each
  gets its OWN `CheckpointManager`, so `kenzo rollback <that repo's path>`
  can undo a bad edit there exactly like it can for the primary repo -- see
  `kenzo.core.runtime.AgentRuntime`'s workspace wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class WorkspaceError(ValueError):
    """Raised for a malformed --see/--also-edit spec or an unresolvable path."""


@dataclass
class WorkspaceRepo:
    alias: str
    path: str  # absolute, resolved
    writable: bool


@dataclass
class Workspace:
    """`primary` is always writable and is NOT included in `others` -- tools
    for the primary repo are the existing, unchanged `read_file`/`write_file`
    /etc.; `others` are reachable only through the new `*_other_repo_*` tools
    (see `kenzo.tools.cross_repo`), each addressed by its `alias`.
    """

    primary_path: str
    others: list[WorkspaceRepo] = field(default_factory=list)

    def find(self, alias: str) -> Optional[WorkspaceRepo]:
        for repo in self.others:
            if repo.alias == alias:
                return repo
        return None

    def readable_aliases(self) -> list[str]:
        return [repo.alias for repo in self.others]

    def writable_aliases(self) -> list[str]:
        return [repo.alias for repo in self.others if repo.writable]

    def has_any_other_repo(self) -> bool:
        return bool(self.others)

    def has_writable_other_repo(self) -> bool:
        return any(repo.writable for repo in self.others)

    def add_reference(self, path: str, alias: Optional[str] = None) -> WorkspaceRepo:
        """Register a new repo MID-SESSION, ALWAYS read-only -- backs the
        `add_reference_repo` tool (see `kenzo.tools.cross_repo`), which lets
        the model discover a repo a user's request mentions inline (e.g. "also
        check C:\\CrossID-Support\\Sprint52\\CIDSaas") without it having been
        named via --see/--also-edit up front. Deliberately has NO writable
        counterpart -- write access is a human decision (--also-edit at
        startup), never something the model grants itself by inferring intent
        from prose; a task that genuinely needs to edit a repo discovered this
        way must tell the user to re-run with --also-edit for it (see
        write_other_repo_file/edit_other_repo_file's refusal message, which
        says exactly this).

        Raises `WorkspaceError` for a nonexistent path or one that resolves to
        the primary repo. If `path` is already registered (by resolved path,
        not alias), returns the EXISTING entry unchanged rather than adding a
        duplicate -- idempotent, since the model may mention the same repo
        again in a later turn.
        """
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"'{path}' does not exist or is not a directory")
        if str(resolved) == self.primary_path:
            raise WorkspaceError(f"'{path}' is already the primary repo you're working in")

        for existing in self.others:
            if existing.path == str(resolved):
                return existing

        taken = {repo.alias for repo in self.others}
        if alias and alias not in taken:
            resolved_alias = alias
        else:
            resolved_alias = _default_alias(str(resolved), taken)

        entry = WorkspaceRepo(alias=resolved_alias, path=str(resolved), writable=False)
        self.others.append(entry)
        return entry


def _default_alias(path: str, taken: set[str]) -> str:
    """Auto-derive a short alias from a path's folder name, deduplicating
    with a numeric suffix on collision (e.g. two repos both named "CIDSaas"
    become "CIDSaas" and "CIDSaas-2"). Callers who want a stable, meaningful
    alias for a genuinely ambiguous set of same-named repos should supply
    one explicitly via `alias=path` instead of relying on this fallback.
    """
    base = Path(path).name or "repo"
    if base not in taken:
        return base
    suffix = 2
    while f"{base}-{suffix}" in taken:
        suffix += 1
    return f"{base}-{suffix}"


def _parse_repo_spec(spec: str) -> tuple[Optional[str], str]:
    """Parse one `--see`/`--also-edit` value: either `alias=path` or just
    `path` (alias auto-derived by the caller). `=` is only treated as an
    alias separator if it appears before any path separator, so a Windows
    drive-letter-only path like `C:\\...` is never misparsed (`:` is not
    `=`, so this is actually unambiguous, but a caller passing a path that
    itself legitimately contains `=` should use the explicit form and can't
    hit this edge case in practice)."""
    if "=" in spec:
        alias, _, path = spec.partition("=")
        alias = alias.strip()
        path = path.strip()
        if alias and path:
            return alias, path
    return None, spec.strip()


def build_workspace(
    primary_repo: str,
    see: Optional[list[str]] = None,
    also_edit: Optional[list[str]] = None,
) -> Workspace:
    """Build a `Workspace` from CLI-style `--see`/`--also-edit` values.

    Raises `WorkspaceError` for a path that doesn't exist or isn't a
    directory, or an alias collision between explicitly-named aliases (an
    auto-derived alias colliding with another auto-derived one is instead
    resolved with a numeric suffix -- see `_default_alias` -- since that's
    a naming coincidence, not a user mistake worth failing on).
    """
    primary_resolved = str(Path(primary_repo).resolve())
    others: list[WorkspaceRepo] = []
    taken_aliases: set[str] = set()
    explicit_aliases: set[str] = set()

    def _add(spec: str, writable: bool) -> None:
        explicit_alias, raw_path = _parse_repo_spec(spec)
        resolved = Path(raw_path).resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"workspace repo path does not exist or is not a directory: '{raw_path}'")
        if str(resolved) == primary_resolved:
            raise WorkspaceError(f"'{raw_path}' is already the primary repo (--repo) -- do not also list it in --see/--also-edit")

        if explicit_alias:
            if explicit_alias in explicit_aliases:
                raise WorkspaceError(f"duplicate workspace alias '{explicit_alias}' -- each --see/--also-edit alias must be unique")
            explicit_aliases.add(explicit_alias)
            alias = explicit_alias
        else:
            alias = _default_alias(str(resolved), taken_aliases)
        taken_aliases.add(alias)

        others.append(WorkspaceRepo(alias=alias, path=str(resolved), writable=writable))

    for spec in see or []:
        _add(spec, writable=False)
    for spec in also_edit or []:
        _add(spec, writable=True)

    return Workspace(primary_path=primary_resolved, others=others)
