/**
 * Hand-written TypeScript mirror of `kenzo/server/protocol.py` -- the
 * WebSocket JSON message contract between `kenzo server` and this extension.
 *
 * This file MUST be kept in sync BY HAND with the Python source of truth
 * (there is no code-generation step in this project). Every message is a
 * JSON object with a `type` field used to dispatch it.
 */

// ---------------------------------------------------------------------------
// Shared shapes
// ---------------------------------------------------------------------------

export type DiagnosticSeverityString = "error" | "warning" | "info" | "hint";

export interface DiagnosticInfo {
  file: string;
  line: number;
  message: string;
  severity: DiagnosticSeverityString;
}

export interface IdeContext {
  current_file?: string;
  selection?: string;
  selection_start_line?: number;
  selection_end_line?: number;
  open_tabs: string[];
  diagnostics: DiagnosticInfo[];
  workspace_path?: string;
  git_status?: string;
}

/** `AgentController.run()`'s full result dict, as sent in `task_complete`. */
export interface TaskResult {
  success: boolean;
  finished: boolean;
  final_response: string;
  changes: unknown[];
  files_modified: string[];
  files_seen: string[];
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Client -> Server
// ---------------------------------------------------------------------------

export interface StartTaskMessage {
  type: "start_task";
  repo_path: string;
  request: string;
  task_id?: string;
  ide_context?: IdeContext;
  /** Base64-encoded raw image bytes (no `data:` URI prefix). Optional --
   * a client may send only an attached screenshot with no typed `request`. */
  image_data?: string;
  /** One of "png" / "jpg" / "jpeg" / "webp". Only meaningful alongside
   * `image_data`. */
  image_format?: string;
}

export interface CancelMessage {
  type: "cancel";
  repo_path: string;
}

export interface EditDecisionMessage {
  type: "edit_decision";
  proposal_id: string;
  decision: "accept" | "reject";
  applied: boolean;
}

export interface IdeContextMessage {
  type: "ide_context";
  repo_path: string;
  context: IdeContext;
}

export type ClientMessage =
  | StartTaskMessage
  | CancelMessage
  | EditDecisionMessage
  | IdeContextMessage;

// ---------------------------------------------------------------------------
// Server -> Client
// ---------------------------------------------------------------------------

export interface TaskStartedEvent {
  type: "task_started";
  task_id: string;
  repo_path: string;
}

/** `event_type` mirrors `kenzo.core.events.AgentEventType`; kept as `string`
 * here (not a literal union) so an extension build never hard-fails if the
 * server's taxonomy grows a value this client doesn't know about yet -- an
 * unrecognized `event_type` should just render generically, not be dropped. */
export interface AgentEventMessage {
  type: "event";
  task_id: string;
  event_type: string;
  stage: string;
  message: string;
  data: Record<string, unknown>;
}

export interface EditProposalEvent {
  type: "edit_proposal";
  task_id: string;
  proposal_id: string;
  path: string;
  action: "create" | "edit";
  diff: string;
  new_content: string;
}

export interface TaskCompleteEvent {
  type: "task_complete";
  task_id: string;
  result: TaskResult;
}

export interface ServerErrorEvent {
  type: "server_error";
  task_id?: string;
  message: string;
}

export type ServerMessage =
  | TaskStartedEvent
  | AgentEventMessage
  | EditProposalEvent
  | TaskCompleteEvent
  | ServerErrorEvent;

// ---------------------------------------------------------------------------
// Runtime validation
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((v) => typeof v === "string");
}

function isValidSeverity(value: unknown): value is DiagnosticSeverityString {
  return value === "error" || value === "warning" || value === "info" || value === "hint";
}

function isDiagnosticInfo(value: unknown): value is DiagnosticInfo {
  if (!isRecord(value)) return false;
  return (
    typeof value.file === "string" &&
    typeof value.line === "number" &&
    typeof value.message === "string" &&
    isValidSeverity(value.severity)
  );
}

function isIdeContext(value: unknown): value is IdeContext {
  if (!isRecord(value)) return false;
  if (value.current_file !== undefined && typeof value.current_file !== "string") return false;
  if (value.selection !== undefined && typeof value.selection !== "string") return false;
  if (
    value.selection_start_line !== undefined &&
    typeof value.selection_start_line !== "number"
  ) {
    return false;
  }
  if (value.selection_end_line !== undefined && typeof value.selection_end_line !== "number") {
    return false;
  }
  if (!isStringArray(value.open_tabs)) return false;
  if (!Array.isArray(value.diagnostics) || !value.diagnostics.every(isDiagnosticInfo)) {
    return false;
  }
  if (value.workspace_path !== undefined && typeof value.workspace_path !== "string") {
    return false;
  }
  if (value.git_status !== undefined && typeof value.git_status !== "string") return false;
  return true;
}

/**
 * Validate + narrow a raw decoded JSON object into a `ServerMessage`.
 * Returns `null` (rather than throwing) for anything malformed or of an
 * unrecognized `type` -- callers should log and ignore, never crash the
 * extension host over a bad/forward-incompatible message from the server.
 */
export function parseServerMessage(raw: unknown): ServerMessage | null {
  if (!isRecord(raw)) return null;
  const type = raw.type;

  switch (type) {
    case "task_started": {
      if (typeof raw.task_id === "string" && typeof raw.repo_path === "string") {
        return { type: "task_started", task_id: raw.task_id, repo_path: raw.repo_path };
      }
      return null;
    }
    case "event": {
      if (
        typeof raw.task_id === "string" &&
        typeof raw.event_type === "string" &&
        typeof raw.stage === "string" &&
        typeof raw.message === "string"
      ) {
        return {
          type: "event",
          task_id: raw.task_id,
          event_type: raw.event_type,
          stage: raw.stage,
          message: raw.message,
          data: isRecord(raw.data) ? raw.data : {},
        };
      }
      return null;
    }
    case "edit_proposal": {
      if (
        typeof raw.task_id === "string" &&
        typeof raw.proposal_id === "string" &&
        typeof raw.path === "string" &&
        (raw.action === "create" || raw.action === "edit") &&
        typeof raw.diff === "string" &&
        typeof raw.new_content === "string"
      ) {
        return {
          type: "edit_proposal",
          task_id: raw.task_id,
          proposal_id: raw.proposal_id,
          path: raw.path,
          action: raw.action,
          diff: raw.diff,
          new_content: raw.new_content,
        };
      }
      return null;
    }
    case "task_complete": {
      if (typeof raw.task_id === "string" && isRecord(raw.result)) {
        const result = raw.result as Record<string, unknown>;
        return {
          type: "task_complete",
          task_id: raw.task_id,
          result: {
            success: Boolean(result.success),
            finished: Boolean(result.finished),
            final_response:
              typeof result.final_response === "string" ? result.final_response : "",
            changes: Array.isArray(result.changes) ? result.changes : [],
            files_modified: isStringArray(result.files_modified) ? result.files_modified : [],
            files_seen: isStringArray(result.files_seen) ? result.files_seen : [],
            ...result,
          },
        };
      }
      return null;
    }
    case "server_error": {
      if (typeof raw.message === "string") {
        return {
          type: "server_error",
          task_id: typeof raw.task_id === "string" ? raw.task_id : undefined,
          message: raw.message,
        };
      }
      return null;
    }
    default:
      return null;
  }
}

/** Exported for reuse/testing of the shared IdeContext shape check. */
export { isIdeContext };
