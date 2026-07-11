"""Memory hygiene: scoring how worth-persisting a candidate memory entry is.

Every memory-producing path in Kenzo (session memory, `ArchitectureMemory`
records, anything a future retrieval layer writes back) generates far more
candidate entries than are actually worth keeping around forever -- most of
what an agent does moment-to-moment is exactly the kind of noise a long-lived
memory store should never accumulate: raw tool stdout, a failed shell command,
a scratch value printed while debugging. Left unfiltered, that noise drowns
out the few entries that genuinely matter later: architecture discoveries,
decisions the user or agent made and committed to, fixes that were confirmed
to work, and reusable patterns.

This module is the single gate every one of those write paths should call
before persisting anything long-term. It is a pure, stateless scoring
function plus a threshold -- no I/O, no state, trivially unit-testable, and
safe to call from anywhere without wiring up a class instance.

`kind` is a free-text label describing what an entry represents. The
vocabulary this module actively recognizes mirrors
`kenzo.memory.architecture_memory.MemoryEntry.kind` ("bug_fix" | "decision" |
"pattern" | "module_note"), plus a handful of common synonyms (see the
`_HIGH_VALUE_*`/`_LOW_VALUE_KINDS` sets below) and the "architecture_discovery"
/ "failed_attempt" / "tool_output" / "temporary" terms named directly in the
memory-hygiene requirement this module implements. An unrecognized `kind` is
scored cautiously (below `IMPORTANCE_THRESHOLD` by default) rather than
guessed generously -- the requirement is "store only X", not "store
everything unless proven noise", so the safe default for an unknown label is
to lean toward NOT persisting it.
"""

from __future__ import annotations

# Kinds representing durable knowledge that isn't really an "attempt" that
# can succeed or fail -- a decision was made, or a pattern was identified,
# independent of whether the immediate task around it panned out. These are
# scored high (and stay high) regardless of `succeeded`, per the explicit
# carve-out in the memory-hygiene requirement: "a design decision is worth
# remembering whether or not the immediate attempt succeeded".
_HIGH_VALUE_SUCCESS_INDEPENDENT_KINDS = frozenset(
    {
        "decision",
        "user_decision",
        "design_decision",
        "pattern",
        "design_pattern",
    }
)

# Kinds representing an outcome-bearing attempt -- a "fix" or a "discovery"
# is only actually a fix/discovery if it worked; if `succeeded=False` these
# are much closer in spirit to a failed attempt than to confirmed knowledge,
# so (unlike the kinds above) they ARE penalized hard when unsuccessful.
_HIGH_VALUE_SUCCESS_SENSITIVE_KINDS = frozenset(
    {
        "bug_fix",
        "fix",
        "successful_fix",
        "architecture_discovery",
        "discovery",
        "module_note",
        # A completed task's outcome (kenzo.memory.session_memory.SessionMemory's
        # record() calls this with kind="task_outcome") is the same shape as a
        # bug_fix/discovery: only actually worth keeping if it succeeded.
        "task_outcome",
        # Phase 8: a fix triggered by a screenshot (kenzo.core.runtime
        # .AgentRuntime._record_completed_task's vision-aware branch) -- same
        # shape as bug_fix, just sourced from an image instead of a typed
        # request; only worth remembering if the fix actually worked.
        "vision_fix",
    }
)

# Ephemeral/noisy content the memory-hygiene requirement explicitly says NOT
# to store: raw tool output, failed attempts, scratch/temporary values. Scored
# low even when `succeeded=True` -- a successful `ls` is still not something
# worth remembering forever -- and lower still when `succeeded=False`.
_LOW_VALUE_KINDS = frozenset(
    {
        "tool_output",
        "raw_output",
        "command_output",
        "debug_output",
        "log",
        "failed_attempt",
        "temporary",
        "temp",
        "scratch",
    }
)

# There is rarely a genuinely useful memory this short (e.g. "fixed it",
# "done", "ok") -- content below this length is capped low regardless of how
# valuable its `kind` label claims to be.
_MIN_CONTENT_LENGTH = 15
_SHORT_CONTENT_SCORE_CAP = 0.1

# Chosen as the midpoint of the 0..1 range, positioned so that every kind
# this module actively recognizes lands unambiguously on one side of it:
# high-value kinds clear it comfortably even at their lowest (~0.85+ when
# success-independent, ~0.85 when success-sensitive and succeeded), while
# low-value kinds and failed success-sensitive attempts fall clearly below it
# (<=0.3). An unrecognized `kind` defaults to just under the threshold, so
# ambiguous/unlabeled content must earn its way in rather than being stored
# by default -- matching the requirement's "store only X" framing rather than
# "store everything that isn't obviously Y".
IMPORTANCE_THRESHOLD = 0.5


def _normalize_kind(kind: str) -> str:
    return kind.strip().lower().replace("-", "_").replace(" ", "_")


def score_importance(kind: str, content: str, *, succeeded: bool = True) -> float:
    """Returns a 0..1 importance score. Higher = more worth persisting long-term.

    Direction of every rule (see module docstring and the `_HIGH_VALUE_*`/
    `_LOW_VALUE_KINDS` sets for the exact vocabulary each bucket recognizes):

    - Architecture discoveries, decisions, successful fixes, and patterns
      score high, especially when `succeeded=True`.
    - Decisions and patterns stay high even when `succeeded=False`, since
      they represent knowledge rather than an outcome (see
      `_HIGH_VALUE_SUCCESS_INDEPENDENT_KINDS`).
    - Tool output, raw output, failed attempts, and temporary/scratch content
      score low, and lower still when `succeeded=False`.
    - An unrecognized `kind` gets a cautious, mildly `succeeded`-sensitive
      score that defaults to just below `IMPORTANCE_THRESHOLD`.
    - Very short content (under `_MIN_CONTENT_LENGTH` characters once
      stripped) is capped low no matter what `kind` claims, and empty/
      whitespace-only content scores 0.0 outright.
    """
    stripped_content = (content or "").strip()
    if not stripped_content:
        return 0.0

    normalized_kind = _normalize_kind(kind)

    if normalized_kind in _HIGH_VALUE_SUCCESS_INDEPENDENT_KINDS:
        score = 0.9 if succeeded else 0.85
    elif normalized_kind in _HIGH_VALUE_SUCCESS_SENSITIVE_KINDS:
        score = 0.85 if succeeded else 0.3
    elif normalized_kind in _LOW_VALUE_KINDS:
        score = 0.2 if succeeded else 0.1
    else:
        score = 0.45 if succeeded else 0.25

    if len(stripped_content) < _MIN_CONTENT_LENGTH:
        score = min(score, _SHORT_CONTENT_SCORE_CAP)

    return round(max(0.0, min(1.0, score)), 3)


def is_worth_remembering(kind: str, content: str, *, succeeded: bool = True) -> bool:
    """Gate deciding whether an entry clears the bar for long-term persistence."""
    return score_importance(kind, content, succeeded=succeeded) >= IMPORTANCE_THRESHOLD
