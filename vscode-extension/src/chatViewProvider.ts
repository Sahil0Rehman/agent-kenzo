/**
 * "Kenzo Chat" activity-bar view: a `vscode.WebviewViewProvider` rendering
 * a scrolling message list of streamed `event` messages (styled distinctly
 * by `event_type`), an input box that starts a task, and status lines for
 * edit-proposal prompts (the actual accept/reject decision is taken via
 * `vscode.window.showInformationMessage`, per the spec's explicitly-allowed
 * simpler option -- see editProposalHandler.ts; this view just narrates
 * that a decision is pending/was made).
 */

import * as vscode from "vscode";
import { KenzoClient } from "./kenzoClient";
import { AgentEventMessage, EditProposalEvent, ServerErrorEvent, TaskCompleteEvent } from "./protocol";

export class KenzoChatViewProvider implements vscode.WebviewViewProvider {
  static readonly viewType = "kenzo.chatView";

  private view: vscode.WebviewView | undefined;
  private currentTaskId: string | undefined;
  private currentRepoPath: string | undefined;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly client: KenzoClient,
    private readonly getRepoPath: () => string | undefined,
    private readonly getIdeContext: () => Promise<import("./protocol").IdeContext | undefined>
  ) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true, localResourceRoots: [this.extensionUri] };
    webviewView.webview.html = this.renderHtml();

    webviewView.webview.onDidReceiveMessage(
      async (msg: {
        type: string;
        text?: string;
        image?: { data: string; format: string; name?: string };
      }) => {
        const text = (msg.text ?? "").trim();
        if (msg.type === "submit" && (text.length > 0 || msg.image)) {
          await this.startTask(text, msg.image);
        }
      }
    );
  }

  reveal(): void {
    this.view?.show?.(true);
  }

  /** `request` may be empty as long as `image` is set -- mirrors the wire
   * protocol's `request` field being optional-with-an-image. */
  async startTask(
    request: string,
    image?: { data: string; format: string; name?: string }
  ): Promise<void> {
    if (!request.trim() && !image) return;
    const repoPath = this.getRepoPath();
    if (!repoPath) {
      vscode.window.showWarningMessage("Kenzo: open a folder/workspace before starting a task.");
      return;
    }
    this.currentRepoPath = repoPath;
    this.postToWebview({
      type: "user_message",
      text: request.trim() ? request : "(screenshot attached)",
    });
    const ideContext = await this.getIdeContext();
    this.client.startTask(
      repoPath,
      request,
      this.currentTaskId,
      ideContext,
      image ? { data: image.data, format: image.format } : undefined
    );
  }

  onTaskStarted(taskId: string): void {
    this.currentTaskId = taskId;
    this.postToWebview({ type: "task_started", taskId });
  }

  onAgentEvent(event: AgentEventMessage): void {
    if (this.currentTaskId && event.task_id !== this.currentTaskId) return;
    this.postToWebview({
      type: "event",
      eventType: event.event_type,
      stage: event.stage,
      message: event.message,
    });
  }

  onEditProposalPending(proposal: EditProposalEvent): void {
    if (this.currentTaskId && proposal.task_id !== this.currentTaskId) return;
    this.postToWebview({
      type: "edit_proposal_pending",
      path: proposal.path,
      action: proposal.action,
    });
  }

  onEditProposalDecided(path: string, decision: "accept" | "reject"): void {
    this.postToWebview({ type: "edit_proposal_decided", path, decision });
  }

  onTaskComplete(event: TaskCompleteEvent): void {
    if (this.currentTaskId && event.task_id !== this.currentTaskId) return;
    this.postToWebview({
      type: "task_complete",
      success: event.result.success,
      finalResponse: event.result.final_response,
      filesModified: event.result.files_modified,
    });
  }

  onServerError(event: ServerErrorEvent): void {
    this.postToWebview({ type: "server_error", message: event.message });
  }

  private postToWebview(message: unknown): void {
    this.view?.webview.postMessage(message);
  }

  private renderHtml(): string {
    return /* html */ `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
<style>
  body { font-family: var(--vscode-font-family); font-size: var(--vscode-font-size); color: var(--vscode-foreground); padding: 0; margin: 0; display: flex; flex-direction: column; height: 100vh; }
  #messages { flex: 1; overflow-y: auto; padding: 8px; }
  .msg { margin-bottom: 10px; padding: 6px 8px; border-radius: 4px; white-space: pre-wrap; word-break: break-word; }
  .msg.user { background: var(--vscode-input-background); }
  .msg.thinking { color: var(--vscode-descriptionForeground); font-style: italic; }
  .msg.tool_start { color: var(--vscode-charts-blue); }
  .msg.tool_result { color: var(--vscode-charts-green); }
  .msg.file_changed { color: var(--vscode-charts-yellow); font-weight: 600; }
  .msg.test_running { color: var(--vscode-charts-purple); font-family: var(--vscode-editor-font-family); }
  .msg.error { color: var(--vscode-errorForeground); font-weight: 600; }
  .msg.complete { color: var(--vscode-charts-green); font-weight: 600; }
  .msg.proposal { border: 1px solid var(--vscode-panel-border); }
  .msg.server_error { color: var(--vscode-errorForeground); background: var(--vscode-inputValidation-errorBackground); }
  .msg.vision { color: var(--vscode-charts-orange); border-left: 3px solid var(--vscode-charts-orange); padding-left: 6px; }
  .msg.delegate { color: var(--vscode-charts-blue); border-left: 3px solid var(--vscode-charts-blue); padding-left: 6px; white-space: pre-wrap; }
  #attachChip { display: none; align-items: center; gap: 6px; margin: 0 8px; padding: 4px 8px; font-size: 0.9em; background: var(--vscode-input-background); border-radius: 4px; width: fit-content; }
  #attachChip button { background: none; border: none; color: var(--vscode-descriptionForeground); cursor: pointer; font-size: inherit; padding: 0; }
  #inputRow { display: flex; gap: 4px; padding: 8px; border-top: 1px solid var(--vscode-panel-border); align-items: flex-end; }
  #inputBox { flex: 1; }
  #attachBtn { cursor: pointer; }
  textarea, button { font-family: inherit; font-size: inherit; }
</style>
</head>
<body>
  <div id="messages"></div>
  <div id="attachChip">
    <span id="attachChipLabel"></span>
    <button id="attachRemoveBtn" title="Remove attachment">✕</button>
  </div>
  <div id="inputRow">
    <input type="file" id="fileInput" accept="image/png,image/jpeg,image/webp" style="display:none" />
    <button id="attachBtn" title="Attach screenshot">📎</button>
    <textarea id="inputBox" rows="2" placeholder="Ask Kenzo..."></textarea>
    <button id="sendBtn">Send</button>
  </div>
<script>
  const vscodeApi = acquireVsCodeApi();
  const messagesEl = document.getElementById("messages");
  const inputBox = document.getElementById("inputBox");
  const sendBtn = document.getElementById("sendBtn");
  const fileInput = document.getElementById("fileInput");
  const attachBtn = document.getElementById("attachBtn");
  const attachChip = document.getElementById("attachChip");
  const attachChipLabel = document.getElementById("attachChipLabel");
  const attachRemoveBtn = document.getElementById("attachRemoveBtn");

  /** @type {{ data: string, format: string, name: string } | null} */
  let pendingImage = null;

  function normalizeFormat(file) {
    const mime = (file.type || "").toLowerCase();
    if (mime === "image/png") return "png";
    if (mime === "image/jpeg") return "jpg";
    if (mime === "image/webp") return "webp";
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (ext === "png" || ext === "jpg" || ext === "jpeg" || ext === "webp") return ext;
    return "png";
  }

  function showAttachChip(name) {
    attachChipLabel.textContent = "📎 " + name;
    attachChip.style.display = "flex";
  }

  function clearAttachment() {
    pendingImage = null;
    fileInput.value = "";
    attachChip.style.display = "none";
  }

  function appendMessage(cls, text) {
    const el = document.createElement("div");
    el.className = "msg " + cls;
    el.textContent = text;
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function submit() {
    const text = inputBox.value || "";
    if (!text.trim() && !pendingImage) return;
    const payload = { type: "submit", text };
    if (pendingImage) payload.image = pendingImage;
    vscodeApi.postMessage(payload);
    inputBox.value = "";
    clearAttachment();
  }

  sendBtn.addEventListener("click", submit);
  inputBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  attachBtn.addEventListener("click", () => fileInput.click());
  attachRemoveBtn.addEventListener("click", clearAttachment);

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const commaIdx = result.indexOf(",");
      const base64 = commaIdx >= 0 ? result.slice(commaIdx + 1) : result;
      pendingImage = { data: base64, format: normalizeFormat(file), name: file.name };
      showAttachChip(file.name);
    };
    reader.readAsDataURL(file);
  });

  window.addEventListener("message", (event) => {
    const msg = event.data;
    switch (msg.type) {
      case "user_message":
        appendMessage("user", "You: " + msg.text);
        break;
      case "task_started":
        appendMessage("thinking", "Kenzo task started (" + msg.taskId + ")");
        break;
      case "event": {
        const cls = msg.stage === "vision" ? "vision" : msg.stage === "delegate" ? "delegate" : (msg.eventType || "thinking");
        appendMessage(cls, "[" + msg.stage + "] " + msg.message);
        break;
      }
      case "edit_proposal_pending":
        appendMessage("proposal", "Kenzo proposes to " + msg.action + " " + msg.path + " -- respond to the prompt to accept/reject.");
        break;
      case "edit_proposal_decided":
        appendMessage("proposal", msg.path + ": " + msg.decision.toUpperCase());
        break;
      case "task_complete":
        appendMessage("complete", (msg.success ? "Done: " : "Finished (unsuccessful): ") + msg.finalResponse);
        if (msg.filesModified && msg.filesModified.length) {
          appendMessage("complete", "Files modified: " + msg.filesModified.join(", "));
        }
        break;
      case "server_error":
        appendMessage("server_error", "Server error: " + msg.message);
        break;
    }
  });
</script>
</body>
</html>`;
  }
}
