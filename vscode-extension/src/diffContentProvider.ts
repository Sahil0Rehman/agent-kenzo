/**
 * A real `vscode.TextDocumentContentProvider` -- the standard VS Code
 * pattern for showing read-only "virtual" content (here: an edit proposal's
 * pre-edit snapshot and its proposed new content) in a diff view without
 * writing anything to a temp file.
 *
 * Two schemes are served by the same provider instance:
 *   kenzo-proposal:/<proposalId>  -- the proposed new content
 *   kenzo-before:/<proposalId>    -- the pre-edit snapshot
 */

import * as vscode from "vscode";

export const PROPOSAL_SCHEME = "kenzo-proposal";
export const BEFORE_SCHEME = "kenzo-before";

export class KenzoVirtualContentProvider implements vscode.TextDocumentContentProvider {
  private readonly contents = new Map<string, string>();
  private readonly onDidChangeEmitter = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.onDidChangeEmitter.event;

  private keyFor(scheme: string, id: string): string {
    return `${scheme}:${id}`;
  }

  setContent(scheme: string, id: string, content: string): vscode.Uri {
    this.contents.set(this.keyFor(scheme, id), content);
    const uri = vscode.Uri.parse(`${scheme}:/${encodeURIComponent(id)}`);
    this.onDidChangeEmitter.fire(uri);
    return uri;
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    const id = decodeURIComponent(uri.path.replace(/^\//, ""));
    return this.contents.get(this.keyFor(uri.scheme, id)) ?? "";
  }

  clear(scheme: string, id: string): void {
    this.contents.delete(this.keyFor(scheme, id));
  }

  dispose(): void {
    this.onDidChangeEmitter.dispose();
    this.contents.clear();
  }
}
