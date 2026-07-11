/**
 * "Kenzo Memory" activity-bar view: a READ-ONLY `vscode.TreeDataProvider`.
 *
 * There is no dedicated "get memory" WebSocket message in the protocol
 * (see kenzo/server/protocol.py's ClientMessage union), so this renders
 * from the most recent `task_complete` result plus a placeholder note --
 * it does NOT invent a new protocol message the server doesn't implement.
 */

import * as vscode from "vscode";
import { TaskResult } from "./protocol";

class MemoryTreeItem extends vscode.TreeItem {
  constructor(label: string, description?: string, tooltip?: string) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = description;
    this.tooltip = tooltip ?? description;
  }
}

export class MemoryViewProvider implements vscode.TreeDataProvider<MemoryTreeItem> {
  private readonly onDidChangeTreeDataEmitter = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.onDidChangeTreeDataEmitter.event;

  private lastResult: TaskResult | undefined;

  update(result: TaskResult): void {
    this.lastResult = result;
    this.onDidChangeTreeDataEmitter.fire();
  }

  getTreeItem(element: MemoryTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): MemoryTreeItem[] {
    const items: MemoryTreeItem[] = [];
    if (!this.lastResult) {
      items.push(new MemoryTreeItem("No task has completed yet"));
    } else {
      items.push(new MemoryTreeItem("Success", String(this.lastResult.success)));
      items.push(new MemoryTreeItem("Finished", String(this.lastResult.finished)));
      items.push(
        new MemoryTreeItem(
          "Files seen",
          String(this.lastResult.files_seen?.length ?? 0),
          (this.lastResult.files_seen ?? []).join("\n")
        )
      );
      items.push(
        new MemoryTreeItem(
          "Files modified",
          String(this.lastResult.files_modified?.length ?? 0),
          (this.lastResult.files_modified ?? []).join("\n")
        )
      );
      items.push(
        new MemoryTreeItem(
          "Final response",
          undefined,
          this.lastResult.final_response
        )
      );
    }
    items.push(
      new MemoryTreeItem(
        "Live project-memory browsing arrives in a future update",
        undefined,
        "The kenzo server WebSocket protocol has no get-memory message yet -- this panel only reflects the last completed task's result."
      )
    );
    return items;
  }
}
