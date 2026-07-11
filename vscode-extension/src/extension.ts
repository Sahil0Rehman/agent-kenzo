/**
 * Extension activation: wires the WebSocket client (kenzoClient.ts) to the
 * three activity-bar views, registers the four+one commands, the IDE
 * context provider, the edit-proposal/diff/WorkspaceEdit flow, and the
 * Kenzo terminal.
 */

import * as vscode from "vscode";
import { KenzoClient } from "./kenzoClient";
import * as protocol from "./protocol";
import { buildIdeContext, registerIdeContextWatchers, VscodeLike } from "./ideContext";
import {
  BEFORE_SCHEME,
  KenzoVirtualContentProvider,
  PROPOSAL_SCHEME,
} from "./diffContentProvider";
import { ChangesStore } from "./changesStore";
import { ChangesViewProvider } from "./changesViewProvider";
import { MemoryViewProvider } from "./memoryViewProvider";
import { KenzoChatViewProvider } from "./chatViewProvider";
import { TerminalManager } from "./terminalManager";
import { EditApplyVscodeLike, handleEditProposal } from "./editProposalHandler";

function getWorkspaceRepoPath(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function getConfiguredClient(): KenzoClient {
  const config = vscode.workspace.getConfiguration("kenzo");
  const host = config.get<string>("serverHost", "127.0.0.1");
  const port = config.get<number>("serverPort", 8756);
  return new KenzoClient({ host, port });
}

function relativePath(fsPath: string): string {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) return fsPath;
  return vscode.workspace.asRelativePath(fsPath, false) || fsPath;
}

function severityToWireString(severity: vscode.DiagnosticSeverity): protocol.DiagnosticSeverityString {
  switch (severity) {
    case vscode.DiagnosticSeverity.Error:
      return "error";
    case vscode.DiagnosticSeverity.Warning:
      return "warning";
    case vscode.DiagnosticSeverity.Information:
      return "info";
    default:
      return "hint";
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const client = getConfiguredClient();
  const contentProvider = new KenzoVirtualContentProvider();
  const changesStore = new ChangesStore();
  const changesViewProvider = new ChangesViewProvider(changesStore, contentProvider);
  const memoryViewProvider = new MemoryViewProvider();
  const terminalManager = new TerminalManager();

  const getRepoPath = (): string | undefined => getWorkspaceRepoPath();
  const getIdeContext = (): Promise<protocol.IdeContext | undefined> => {
    const repoPath = getRepoPath();
    if (!repoPath) return Promise.resolve(undefined);
    return buildIdeContext(vscode as unknown as VscodeLike);
  };

  const chatViewProvider = new KenzoChatViewProvider(
    context.extensionUri,
    client,
    getRepoPath,
    getIdeContext
  );

  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(PROPOSAL_SCHEME, contentProvider),
    vscode.workspace.registerTextDocumentContentProvider(BEFORE_SCHEME, contentProvider),
    vscode.window.registerWebviewViewProvider(KenzoChatViewProvider.viewType, chatViewProvider),
    vscode.window.registerTreeDataProvider("kenzo.changesView", changesViewProvider),
    vscode.window.registerTreeDataProvider("kenzo.memoryView", memoryViewProvider)
  );

  // -- wire client events to the views / terminal / changes store ---------

  client.on("task_started", (msg: protocol.TaskStartedEvent) => {
    chatViewProvider.onTaskStarted(msg.task_id);
  });

  client.on("event", (msg: protocol.AgentEventMessage) => {
    chatViewProvider.onAgentEvent(msg);
    terminalManager.handleAgentEvent(msg);
  });

  client.on("edit_proposal", (proposal: protocol.EditProposalEvent) => {
    chatViewProvider.onEditProposalPending(proposal);
    void handleEditProposal(
      proposal,
      vscode as unknown as EditApplyVscodeLike,
      client,
      {
        proposedUri: (p) => contentProvider.setContent(PROPOSAL_SCHEME, p.proposal_id, p.new_content),
        beforeUri: (p, before) => contentProvider.setContent(BEFORE_SCHEME, p.proposal_id, before),
      }
    ).then((outcome) => {
      chatViewProvider.onEditProposalDecided(outcome.path, outcome.decision);
      if (outcome.decision === "accept") {
        changesStore.addAcceptedProposal({
          id: proposal.proposal_id,
          path: outcome.path,
          action: outcome.action,
          beforeContent: outcome.beforeContent,
          afterContent: outcome.afterContent,
        });
      }
    });
  });

  client.on("task_complete", (msg: protocol.TaskCompleteEvent) => {
    chatViewProvider.onTaskComplete(msg);
    memoryViewProvider.update(msg.result);
    changesStore.mergeTaskComplete(msg.result.files_modified ?? []);
  });

  client.on("server_error", (msg: protocol.ServerErrorEvent) => {
    chatViewProvider.onServerError(msg);
    vscode.window.showErrorMessage(`Kenzo: ${msg.message}`);
  });

  client.on("socket_error", () => {
    /* swallow -- reconnect-with-backoff in KenzoClient handles retrying;
       surfacing every transient connection error would be noisy. */
  });

  client.connect();
  context.subscriptions.push({ dispose: () => client.disconnect() });

  // -- IDE context watchers -------------------------------------------------

  const ideContextWatcher = registerIdeContextWatchers(vscode as unknown as VscodeLike, (ctx) => {
    const repoPath = getRepoPath();
    if (repoPath) client.sendIdeContext(repoPath, ctx);
  });
  context.subscriptions.push({ dispose: () => ideContextWatcher.dispose() });

  // -- commands --------------------------------------------------------------

  context.subscriptions.push(
    vscode.commands.registerCommand("kenzo.chat", () => {
      chatViewProvider.reveal();
    }),

    vscode.commands.registerCommand("kenzo.explainFile", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("Kenzo: no active file to explain.");
        return;
      }
      const rel = relativePath(editor.document.uri.fsPath);
      chatViewProvider.reveal();
      const ideContext = await getIdeContext();
      const repoPath = getRepoPath();
      if (!repoPath) return;
      client.startTask(repoPath, `Explain what ${rel} does`, undefined, ideContext);
    }),

    vscode.commands.registerCommand("kenzo.fixError", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("Kenzo: no active file.");
        return;
      }
      const position = editor.selection.active;
      const diagnostics = vscode.languages
        .getDiagnostics(editor.document.uri)
        .filter((d) => d.range.contains(position));
      if (diagnostics.length === 0) {
        vscode.window.showInformationMessage("Kenzo: no diagnostic at the cursor.");
        return;
      }
      const diagnostic = diagnostics[0];
      const rel = relativePath(editor.document.uri.fsPath);
      const line = diagnostic.range.start.line + 1;
      chatViewProvider.reveal();
      const ideContext = await getIdeContext();
      if (ideContext) {
        ideContext.diagnostics = vscode.languages.getDiagnostics(editor.document.uri).map((d) => ({
          file: rel,
          line: d.range.start.line,
          message: d.message,
          severity: severityToWireString(d.severity),
        }));
      }
      const repoPath = getRepoPath();
      if (!repoPath) return;
      client.startTask(
        repoPath,
        `Fix this error: ${diagnostic.message} at ${rel}:${line}`,
        undefined,
        ideContext
      );
    }),

    vscode.commands.registerCommand("kenzo.reviewSelection", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.selection.isEmpty) {
        vscode.window.showWarningMessage("Kenzo: select some code first.");
        return;
      }
      const selectedText = editor.document.getText(editor.selection);
      chatViewProvider.reveal();
      const ideContext = await getIdeContext();
      if (ideContext) {
        ideContext.selection = selectedText;
        ideContext.selection_start_line = editor.selection.start.line;
        ideContext.selection_end_line = editor.selection.end.line;
      }
      const repoPath = getRepoPath();
      if (!repoPath) return;
      client.startTask(repoPath, `Review this code:\n${selectedText}`, undefined, ideContext);
    }),

    vscode.commands.registerCommand("kenzo.showTerminal", () => {
      terminalManager.reveal();
    }),

    vscode.commands.registerCommand("kenzo.openChangeDiff", (recordId: string) => {
      void changesViewProvider.openDiff(recordId);
    })
  );

  // -- Fix Error as a quick-fix CodeActionProvider (nice-to-have) -----------

  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      { pattern: "**/*" },
      {
        provideCodeActions(document, range, ctx) {
          const errorDiagnostics = ctx.diagnostics.filter(
            (d) => d.severity === vscode.DiagnosticSeverity.Error
          );
          if (errorDiagnostics.length === 0) return [];
          const action = new vscode.CodeAction(
            "Kenzo: Fix Error",
            vscode.CodeActionKind.QuickFix
          );
          action.command = { command: "kenzo.fixError", title: "Kenzo: Fix Error" };
          action.diagnostics = errorDiagnostics;
          return [action];
        },
      },
      { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }
    )
  );
}

export function deactivate(): void {
  // Disconnection is handled by the disposable pushed in activate().
}
