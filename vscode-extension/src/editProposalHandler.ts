/**
 * Handles one `edit_proposal` event end to end:
 *   1. Show a real VS Code diff view (current file content vs. proposed
 *      `new_content`).
 *   2. Ask the user to accept/reject.
 *   3. On accept: apply the change via a real `vscode.WorkspaceEdit` (never
 *      a raw `fs.writeFile`), then send `edit_decision` with `applied:
 *      true`. On reject: send `edit_decision` with `applied: false` and
 *      touch nothing.
 *
 * The VS Code API surface touched is expressed as `EditApplyVscodeLike`
 * (a structural subset) so this logic is unit-testable with a fake object,
 * without needing the real `vscode` module.
 */

import { EditProposalEvent } from "./protocol";

export interface VscodeUriLike {
  fsPath: string;
  toString(): string;
}

export interface VscodePositionLike {
  line: number;
  character: number;
}

export interface VscodeRangeLike {
  start: VscodePositionLike;
  end: VscodePositionLike;
}

export interface VscodeWorkspaceEditLike {
  replace(uri: VscodeUriLike, range: VscodeRangeLike, newText: string): void;
  createFile(uri: VscodeUriLike, options?: { overwrite?: boolean; ignoreIfExists?: boolean }): void;
  insert(uri: VscodeUriLike, position: VscodePositionLike, text: string): void;
}

export interface VscodeTextDocumentLike {
  getText(): string;
  uri: VscodeUriLike;
}

export interface EditApplyVscodeLike {
  Uri: {
    file(path: string): VscodeUriLike;
  };
  workspace: {
    openTextDocument(uri: VscodeUriLike): Thenable<VscodeTextDocumentLike>;
    applyEdit(edit: VscodeWorkspaceEditLike): Thenable<boolean>;
    fs: {
      stat(uri: VscodeUriLike): Thenable<unknown>;
    };
  };
  WorkspaceEdit: new () => VscodeWorkspaceEditLike;
  Position: new (line: number, character: number) => VscodePositionLike;
  Range: new (
    startLine: number,
    startChar: number,
    endLine: number,
    endChar: number
  ) => VscodeRangeLike;
  window: {
    showInformationMessage(message: string, ...items: string[]): Thenable<string | undefined>;
  };
  commands: {
    executeCommand(command: string, ...args: unknown[]): Thenable<unknown>;
  };
}

export interface KenzoClientLike {
  resolveEditProposal(proposalId: string, decision: "accept" | "reject", applied: boolean): void;
}

/** Supplies the two virtual diff URIs (proposed content, pre-edit snapshot)
 * -- backed by `KenzoVirtualContentProvider` in the real extension. */
export interface DiffUriProvider {
  proposedUri(proposal: EditProposalEvent): VscodeUriLike;
  beforeUri(proposal: EditProposalEvent, beforeContent: string): VscodeUriLike;
}

export interface EditProposalOutcome {
  decision: "accept" | "reject";
  path: string;
  beforeContent: string;
  afterContent: string;
  action: "create" | "edit";
}

async function readExistingContent(
  vscodeApi: EditApplyVscodeLike,
  uri: VscodeUriLike
): Promise<string> {
  try {
    const doc = await vscodeApi.workspace.openTextDocument(uri);
    return doc.getText();
  } catch {
    return "";
  }
}

function fullDocumentRange(vscodeApi: EditApplyVscodeLike, text: string): VscodeRangeLike {
  const lines = text.split("\n");
  const lastLine = Math.max(lines.length - 1, 0);
  const lastLineLength = lines[lastLine]?.length ?? 0;
  return new vscodeApi.Range(0, 0, lastLine, lastLineLength);
}

/**
 * Runs the full show-diff -> prompt -> (apply + edit_decision) flow for one
 * proposal. Returns the outcome so callers (e.g. the Changes view store)
 * can record it. Never writes via `fs` -- accept always goes through
 * `vscode.WorkspaceEdit` + `vscode.workspace.applyEdit`.
 */
export async function handleEditProposal(
  proposal: EditProposalEvent,
  vscodeApi: EditApplyVscodeLike,
  client: KenzoClientLike,
  diffProvider: DiffUriProvider
): Promise<EditProposalOutcome> {
  const fileUri = vscodeApi.Uri.file(proposal.path);
  const beforeContent =
    proposal.action === "create" ? "" : await readExistingContent(vscodeApi, fileUri);

  const leftUri = diffProvider.beforeUri(proposal, beforeContent);
  const rightUri = diffProvider.proposedUri(proposal);
  const title = `Kenzo Proposal: ${proposal.path} (${proposal.action})`;

  try {
    await vscodeApi.commands.executeCommand("vscode.diff", leftUri, rightUri, title);
  } catch {
    // Diff view failed to open (e.g. an unusual URI scheme in a test/host
    // edge case) -- fall through to the accept/reject prompt regardless;
    // showing the diff is a nicety, not a precondition for the decision.
  }

  const choice = await vscodeApi.window.showInformationMessage(
    `Kenzo wants to ${proposal.action === "create" ? "create" : "edit"} ${proposal.path}. Apply this change?`,
    "Accept",
    "Reject"
  );

  if (choice === "Accept") {
    const edit = new vscodeApi.WorkspaceEdit();
    if (proposal.action === "create") {
      edit.createFile(fileUri, { overwrite: true, ignoreIfExists: false });
      edit.insert(fileUri, new vscodeApi.Position(0, 0), proposal.new_content);
    } else {
      const range = fullDocumentRange(vscodeApi, beforeContent);
      edit.replace(fileUri, range, proposal.new_content);
    }
    await vscodeApi.workspace.applyEdit(edit);
    client.resolveEditProposal(proposal.proposal_id, "accept", true);
    return {
      decision: "accept",
      path: proposal.path,
      beforeContent,
      afterContent: proposal.new_content,
      action: proposal.action,
    };
  }

  client.resolveEditProposal(proposal.proposal_id, "reject", false);
  return {
    decision: "reject",
    path: proposal.path,
    beforeContent,
    afterContent: proposal.new_content,
    action: proposal.action,
  };
}
