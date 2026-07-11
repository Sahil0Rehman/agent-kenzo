/**
 * A "Kenzo" `vscode.Terminal` used purely for VISIBILITY of shell commands
 * the agent already ran server-side -- `terminal.sendText(cmd, false)` so
 * the command text is displayed without being re-executed in this
 * terminal (the real execution already happened in `kenzo server`'s
 * process, via `kenzo.tools.terminal.run_command`).
 *
 * Grounded in the real event shapes emitted by
 * `kenzo.core.controller.AgentController._on_tool_start`:
 *   - `event_type: "test_running"` carries `data.command` -- the actual
 *     shell command string. This is the one we surface.
 *   - `event_type: "tool_start"` for `run_command` carries
 *     `data.arguments.command` (the same command, one event earlier) --
 *     kept as a fallback in case a future server build only emits
 *     `tool_start` for some commands.
 */

import * as vscode from "vscode";
import { AgentEventMessage } from "./protocol";

export interface VscodeTerminalLike {
  sendText(text: string, addNewLine?: boolean): void;
  show(preserveFocus?: boolean): void;
}

export interface VscodeTerminalFactory {
  createTerminal(name: string): VscodeTerminalLike;
}

function extractCommand(event: AgentEventMessage): string | undefined {
  if (event.event_type === "test_running") {
    const command = event.data?.command;
    if (typeof command === "string" && command.length > 0) return command;
  }
  if (event.event_type === "tool_start") {
    const tool = event.data?.tool;
    const args = event.data?.arguments as Record<string, unknown> | undefined;
    if (tool === "run_command" && args && typeof args.command === "string") {
      return args.command;
    }
  }
  return undefined;
}

export class TerminalManager {
  private terminal: VscodeTerminalLike | undefined;

  constructor(private readonly factory: VscodeTerminalFactory = vscode.window) {}

  private ensureTerminal(): VscodeTerminalLike {
    if (!this.terminal) {
      this.terminal = this.factory.createTerminal("Kenzo");
    }
    return this.terminal;
  }

  /** Called for every incoming `event` message; no-ops for events that
   * don't carry a shell command. */
  handleAgentEvent(event: AgentEventMessage): void {
    const command = extractCommand(event);
    if (!command) return;
    const terminal = this.ensureTerminal();
    terminal.sendText(`# Kenzo ran: ${command}`, false);
  }

  reveal(): void {
    this.ensureTerminal().show(true);
  }

  /** Reset so a fresh terminal is created next time (e.g. if the user
   * closed it). Not currently wired to `onDidCloseTerminal` to keep scope
   * small, but exposed for tests / future use. */
  reset(): void {
    this.terminal = undefined;
  }
}
