/**
 * In-memory record of files touched by the current/most recent task, used
 * by the "Kenzo Changes" view. Populated two ways:
 *   1. `addAcceptedProposal` -- every ACCEPTED edit_proposal, capturing the
 *      pre-edit snapshot so the Changes view can open a real diff.
 *   2. `mergeTaskComplete` -- `task_complete`'s `result.files_modified`, as
 *      a fallback so the list still reflects reality if a file was touched
 *      through some path that didn't produce a matching proposal record.
 */

export interface ChangeRecord {
  id: string;
  path: string;
  action: "create" | "edit";
  beforeContent: string | undefined;
  afterContent: string | undefined;
  timestamp: number;
}

type Listener = () => void;

export class ChangesStore {
  private records: ChangeRecord[] = [];
  private listeners: Listener[] = [];

  onDidChange(listener: Listener): { dispose(): void } {
    this.listeners.push(listener);
    return {
      dispose: () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      },
    };
  }

  private fire(): void {
    for (const l of this.listeners) l();
  }

  addAcceptedProposal(record: Omit<ChangeRecord, "timestamp">): void {
    this.records = this.records.filter((r) => r.path !== record.path);
    this.records.push({ ...record, timestamp: Date.now() });
    this.fire();
  }

  mergeTaskComplete(filesModified: string[]): void {
    let changed = false;
    for (const path of filesModified) {
      if (!this.records.some((r) => r.path === path)) {
        this.records.push({
          id: `fallback:${path}`,
          path,
          action: "edit",
          beforeContent: undefined,
          afterContent: undefined,
          timestamp: Date.now(),
        });
        changed = true;
      }
    }
    if (changed) this.fire();
  }

  getAll(): ChangeRecord[] {
    return [...this.records].sort((a, b) => b.timestamp - a.timestamp);
  }

  getById(id: string): ChangeRecord | undefined {
    return this.records.find((r) => r.id === id);
  }

  clear(): void {
    this.records = [];
    this.fire();
  }
}
