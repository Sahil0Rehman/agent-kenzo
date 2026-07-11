/**
 * WebSocket client for `kenzo server`'s `/ws` endpoint. Wraps the `ws`
 * package connection and exposes an EventEmitter-style API
 * (`connected` / `disconnected` / `message` / `task_started` / `event` /
 * `edit_proposal` / `task_complete` / `server_error` / `parse_error` /
 * `socket_error` / `send_failed`) that other parts of the extension
 * subscribe to.
 */

import { EventEmitter } from "events";
import WebSocket from "ws";
import * as protocol from "./protocol";

export interface KenzoClientOptions {
  host?: string;
  port?: number;
  /** Base delay (ms) for the first reconnect attempt; doubles each retry,
   * capped at `maxReconnectDelayMs`. Exposed for tests. */
  initialReconnectDelayMs?: number;
  maxReconnectDelayMs?: number;
}

export class KenzoClient extends EventEmitter {
  private readonly url: string;
  private readonly initialReconnectDelayMs: number;
  private readonly maxReconnectDelayMs: number;

  private ws: WebSocket | undefined;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  private manuallyDisconnected = true;

  constructor(options: KenzoClientOptions = {}) {
    super();
    const host = options.host ?? "127.0.0.1";
    const port = options.port ?? 8756;
    this.url = `ws://${host}:${port}/ws`;
    this.initialReconnectDelayMs = options.initialReconnectDelayMs ?? 1000;
    this.maxReconnectDelayMs = options.maxReconnectDelayMs ?? 30000;
  }

  get isConnected(): boolean {
    return this.ws !== undefined && this.ws.readyState === WebSocket.OPEN;
  }

  connect(): void {
    this.manuallyDisconnected = false;
    this.clearReconnectTimer();
    this.openSocket();
  }

  disconnect(): void {
    this.manuallyDisconnected = true;
    this.clearReconnectTimer();
    this.ws?.removeAllListeners();
    this.ws?.close();
    this.ws = undefined;
  }

  // -- outgoing messages ---------------------------------------------------

  startTask(
    repoPath: string,
    request: string,
    taskId?: string,
    ideContext?: protocol.IdeContext,
    image?: { data: string; format: string }
  ): void {
    this.send({
      type: "start_task",
      repo_path: repoPath,
      request,
      task_id: taskId,
      ide_context: ideContext,
      image_data: image?.data,
      image_format: image?.format,
    });
  }

  cancel(repoPath: string): void {
    this.send({ type: "cancel", repo_path: repoPath });
  }

  resolveEditProposal(proposalId: string, decision: "accept" | "reject", applied: boolean): void {
    this.send({ type: "edit_decision", proposal_id: proposalId, decision, applied });
  }

  sendIdeContext(repoPath: string, context: protocol.IdeContext): void {
    this.send({ type: "ide_context", repo_path: repoPath, context });
  }

  // -- internals ------------------------------------------------------------

  private openSocket(): void {
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch (err) {
      this.emit("socket_error", err);
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.on("open", () => {
      this.reconnectAttempts = 0;
      this.emit("connected");
    });
    ws.on("message", (data: WebSocket.RawData) => {
      this.handleRawMessage(data.toString());
    });
    ws.on("close", () => {
      this.emit("disconnected");
      if (!this.manuallyDisconnected) {
        this.scheduleReconnect();
      }
    });
    ws.on("error", (err: Error) => {
      this.emit("socket_error", err);
    });
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer();
    const delay = Math.min(
      this.maxReconnectDelayMs,
      this.initialReconnectDelayMs * 2 ** this.reconnectAttempts
    );
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => {
      if (!this.manuallyDisconnected) {
        this.openSocket();
      }
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
  }

  private handleRawMessage(raw: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      this.emit("parse_error", raw, err);
      return;
    }
    const msg = protocol.parseServerMessage(parsed);
    if (!msg) {
      this.emit("parse_error", raw);
      return;
    }
    this.emit("message", msg);
    this.emit(msg.type, msg);
  }

  private send(msg: protocol.ClientMessage): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.emit("send_failed", msg);
      return;
    }
    try {
      this.ws.send(JSON.stringify(msg));
    } catch (err) {
      this.emit("send_failed", msg, err);
    }
  }
}
