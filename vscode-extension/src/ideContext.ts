/**
 * Builds the `IdeContext` wire object (see protocol.ts) from real VS Code
 * state. The VS Code API surface actually touched is expressed as the
 * `VscodeLike` structural interface below rather than imported directly, so
 * `buildIdeContext`/`collectDiagnostics`/`collectOpenTabs` can be unit
 * tested with a plain fake object -- no real `vscode` module (which only
 * exists inside the extension host) is required to exercise this file's
 * logic. `extension.ts` passes the real `vscode` module in, which
 * structurally satisfies this interface.
 */

import { exec as execCallback } from "child_process";
import { promisify } from "util";
import { DiagnosticInfo, IdeContext } from "./protocol";

export type ExecFn = (
  command: string,
  options: { cwd: string }
) => Promise<{ stdout: string; stderr: string }>;

const defaultExec: ExecFn = promisify(execCallback) as unknown as ExecFn;

// -- structural subset of the real `vscode` module used here ----------------

export interface VscodePosition {
  line: number;
  character: number;
}

export interface VscodeRange {
  start: VscodePosition;
  end: VscodePosition;
}

export interface VscodeSelection extends VscodeRange {
  isEmpty: boolean;
}

export interface VscodeUri {
  fsPath: string;
  toString(): string;
}

export interface VscodeTextDocument {
  uri: VscodeUri;
  getText(range?: VscodeRange): string;
}

export interface VscodeTextEditor {
  document: VscodeTextDocument;
  selection: VscodeSelection;
}

export interface VscodeTabInput {
  uri?: VscodeUri;
}

export interface VscodeTab {
  input?: VscodeTabInput | unknown;
}

export interface VscodeTabGroup {
  tabs: VscodeTab[];
}

export interface VscodeDiagnostic {
  range: VscodeRange;
  message: string;
  severity: number;
}

export interface VscodeWorkspaceFolder {
  uri: VscodeUri;
}

export interface VscodeDisposable {
  dispose(): unknown;
}

export interface VscodeLike {
  window: {
    activeTextEditor?: VscodeTextEditor;
    tabGroups: { all: VscodeTabGroup[] };
    onDidChangeActiveTextEditor?(listener: (...args: unknown[]) => void): VscodeDisposable;
    onDidChangeTextEditorSelection?(listener: (...args: unknown[]) => void): VscodeDisposable;
  };
  workspace: {
    workspaceFolders?: VscodeWorkspaceFolder[];
  };
  languages: {
    getDiagnostics(): Array<[VscodeUri, VscodeDiagnostic[]]>;
  };
  DiagnosticSeverity: {
    Error: number;
    Warning: number;
    Information: number;
    Hint: number;
  };
}

function severityToString(vscodeApi: VscodeLike, severity: number): DiagnosticInfo["severity"] {
  const S = vscodeApi.DiagnosticSeverity;
  if (severity === S.Error) return "error";
  if (severity === S.Warning) return "warning";
  if (severity === S.Information) return "info";
  return "hint";
}

export function collectDiagnostics(vscodeApi: VscodeLike): DiagnosticInfo[] {
  const out: DiagnosticInfo[] = [];
  for (const [uri, diagnostics] of vscodeApi.languages.getDiagnostics()) {
    for (const diagnostic of diagnostics) {
      out.push({
        file: uri.fsPath,
        line: diagnostic.range.start.line,
        message: diagnostic.message,
        severity: severityToString(vscodeApi, diagnostic.severity),
      });
    }
  }
  return out;
}

export function collectOpenTabs(vscodeApi: VscodeLike): string[] {
  const tabs: string[] = [];
  for (const group of vscodeApi.window.tabGroups.all) {
    for (const tab of group.tabs) {
      const input = tab.input as VscodeTabInput | undefined;
      if (input && input.uri && input.uri.fsPath) {
        tabs.push(input.uri.fsPath);
      }
    }
  }
  return tabs;
}

/** Real, non-stub `git status --porcelain` shell-out (via `child_process`).
 * Returns `undefined` (rather than throwing) if `cwd` isn't a git repo, or
 * `git` isn't on PATH -- git status is best-effort context, not required. */
export async function getGitStatus(
  cwd: string,
  execFn: ExecFn = defaultExec
): Promise<string | undefined> {
  try {
    const { stdout } = await execFn("git status --porcelain", { cwd });
    return stdout.trim();
  } catch {
    return undefined;
  }
}

export async function buildIdeContext(
  vscodeApi: VscodeLike,
  execFn: ExecFn = defaultExec
): Promise<IdeContext> {
  const editor = vscodeApi.window.activeTextEditor;
  const workspaceFolder = vscodeApi.workspace.workspaceFolders?.[0];

  const context: IdeContext = {
    open_tabs: collectOpenTabs(vscodeApi),
    diagnostics: collectDiagnostics(vscodeApi),
  };

  if (editor) {
    context.current_file = editor.document.uri.fsPath;
    if (!editor.selection.isEmpty) {
      context.selection = editor.document.getText(editor.selection);
      context.selection_start_line = editor.selection.start.line;
      context.selection_end_line = editor.selection.end.line;
    }
  }

  if (workspaceFolder) {
    context.workspace_path = workspaceFolder.uri.fsPath;
    context.git_status = await getGitStatus(workspaceFolder.uri.fsPath, execFn);
  }

  return context;
}

/**
 * Wires `onDidChangeActiveTextEditor` / `onDidChangeTextEditorSelection` to
 * push a freshly-built `IdeContext` to `onContextChange`, debounced so a
 * flurry of selection changes doesn't flood the WebSocket.
 */
export function registerIdeContextWatchers(
  vscodeApi: VscodeLike,
  onContextChange: (context: IdeContext) => void,
  debounceMs = 300,
  execFn: ExecFn = defaultExec
): VscodeDisposable {
  let timer: ReturnType<typeof setTimeout> | undefined;

  const trigger = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      buildIdeContext(vscodeApi, execFn)
        .then(onContextChange)
        .catch(() => {
          /* best-effort; a failed context build must never crash the extension */
        });
    }, debounceMs);
  };

  const disposables: VscodeDisposable[] = [];
  if (vscodeApi.window.onDidChangeActiveTextEditor) {
    disposables.push(vscodeApi.window.onDidChangeActiveTextEditor(trigger));
  }
  if (vscodeApi.window.onDidChangeTextEditorSelection) {
    disposables.push(vscodeApi.window.onDidChangeTextEditorSelection(trigger));
  }

  return {
    dispose() {
      if (timer) clearTimeout(timer);
      for (const d of disposables) d.dispose();
    },
  };
}
