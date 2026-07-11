/**
 * "Kenzo Changes" activity-bar view: a `vscode.TreeDataProvider` listing
 * files touched by the current/most recent task (from `ChangesStore`).
 * Each entry is clickable and opens a real VS Code diff view against the
 * pre-edit content captured at accept-time (via `KenzoVirtualContentProvider`),
 * or -- for fallback entries with no captured snapshot -- just opens the
 * file.
 */

import * as vscode from "vscode";
import { ChangeRecord, ChangesStore } from "./changesStore";
import { BEFORE_SCHEME, KenzoVirtualContentProvider, PROPOSAL_SCHEME } from "./diffContentProvider";

export class ChangeTreeItem extends vscode.TreeItem {
  constructor(public readonly record: ChangeRecord) {
    super(record.path, vscode.TreeItemCollapsibleState.None);
    this.description = record.action;
    this.tooltip = record.path;
    this.contextValue = "kenzoChange";
    this.command = {
      command: "kenzo.openChangeDiff",
      title: "Open Diff",
      arguments: [record.id],
    };
    this.iconPath = new vscode.ThemeIcon(
      record.action === "create" ? "diff-added" : "diff-modified"
    );
  }
}

export class ChangesViewProvider implements vscode.TreeDataProvider<ChangeTreeItem> {
  private readonly onDidChangeTreeDataEmitter = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.onDidChangeTreeDataEmitter.event;

  constructor(
    private readonly store: ChangesStore,
    private readonly contentProvider: KenzoVirtualContentProvider
  ) {
    this.store.onDidChange(() => this.onDidChangeTreeDataEmitter.fire());
  }

  getTreeItem(element: ChangeTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): ChangeTreeItem[] {
    return this.store.getAll().map((record) => new ChangeTreeItem(record));
  }

  async openDiff(recordId: string): Promise<void> {
    const record = this.store.getById(recordId);
    if (!record) {
      vscode.window.showWarningMessage("Kenzo: this change is no longer available.");
      return;
    }
    if (record.beforeContent === undefined || record.afterContent === undefined) {
      // Fallback entry with no captured snapshot -- just open the file.
      const uri = vscode.Uri.file(record.path);
      await vscode.window.showTextDocument(uri);
      return;
    }
    const leftUri = this.contentProvider.setContent(BEFORE_SCHEME, record.id, record.beforeContent);
    const rightUri = this.contentProvider.setContent(
      PROPOSAL_SCHEME,
      record.id,
      record.afterContent
    );
    await vscode.commands.executeCommand(
      "vscode.diff",
      leftUri,
      rightUri,
      `Kenzo Change: ${record.path}`
    );
  }
}
