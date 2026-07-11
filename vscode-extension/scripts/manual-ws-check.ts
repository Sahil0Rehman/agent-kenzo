/**
 * Standalone, non-test manual verification aid: connects a real `ws`
 * WebSocket client to a REAL running `kenzo server` instance and exercises
 * start_task -> events -> task_complete/server_error end-to-end.
 *
 * Not part of the mocha suite -- run directly with:
 *   node out/scripts/manual-ws-check.js [repoPath] [host] [port]
 *
 * Expects `kenzo server` already running (see vscode-extension/README.md).
 * A real LLM call will fail in this environment (no API key configured) --
 * that's fine, this script only proves the WebSocket plumbing (connect,
 * start_task, task_started ack, and either progress events or a clean
 * server_error/task_complete reflecting the missing-key failure).
 */

import WebSocket from "ws";
import * as protocol from "../src/protocol";

const repoPath = process.argv[2] ?? process.cwd();
const host = process.argv[3] ?? "127.0.0.1";
const port = Number(process.argv[4] ?? 8756);
const url = `ws://${host}:${port}/ws`;

console.log(`[manual-ws-check] connecting to ${url}`);
console.log(`[manual-ws-check] repo_path=${repoPath}`);

const ws = new WebSocket(url);

let sawTaskStarted = false;
let sawTerminalMessage = false;

const hardTimeout = setTimeout(() => {
  console.error("[manual-ws-check] TIMEOUT waiting for a terminal message (task_complete/server_error).");
  process.exit(1);
}, 30000);

ws.on("open", () => {
  console.log("[manual-ws-check] OPEN -- socket connected.");
  const startMsg: protocol.StartTaskMessage = {
    type: "start_task",
    repo_path: repoPath,
    request: "Say hello (this is a plumbing check, not a real coding task).",
  };
  ws.send(JSON.stringify(startMsg));
  console.log("[manual-ws-check] SENT start_task:", JSON.stringify(startMsg));
});

ws.on("message", (raw) => {
  const text = raw.toString();
  console.log("[manual-ws-check] RECEIVED raw:", text);

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    console.error("[manual-ws-check] FAILED to JSON.parse server message:", err);
    return;
  }

  const msg = protocol.parseServerMessage(parsed);
  if (!msg) {
    console.error("[manual-ws-check] FAILED protocol.parseServerMessage -- unrecognized shape.");
    return;
  }

  switch (msg.type) {
    case "task_started":
      sawTaskStarted = true;
      console.log(`[manual-ws-check] task_started ack: task_id=${msg.task_id} repo_path=${msg.repo_path}`);
      break;
    case "event":
      console.log(`[manual-ws-check] event: event_type=${msg.event_type} stage=${msg.stage} message=${msg.message}`);
      break;
    case "edit_proposal":
      console.log(`[manual-ws-check] edit_proposal: path=${msg.path} action=${msg.action} -- auto-rejecting (plumbing check only).`);
      ws.send(
        JSON.stringify({
          type: "edit_decision",
          proposal_id: msg.proposal_id,
          decision: "reject",
          applied: false,
        } satisfies protocol.EditDecisionMessage)
      );
      break;
    case "task_complete":
      sawTerminalMessage = true;
      console.log(`[manual-ws-check] task_complete: success=${msg.result.success} finished=${msg.result.finished}`);
      console.log(`[manual-ws-check] final_response: ${msg.result.final_response}`);
      finish();
      break;
    case "server_error":
      sawTerminalMessage = true;
      console.log(`[manual-ws-check] server_error: ${msg.message}`);
      finish();
      break;
  }
});

ws.on("error", (err) => {
  console.error("[manual-ws-check] SOCKET ERROR:", err.message);
  clearTimeout(hardTimeout);
  process.exit(1);
});

ws.on("close", (code, reason) => {
  console.log(`[manual-ws-check] CLOSE code=${code} reason=${reason.toString()}`);
});

function finish(): void {
  clearTimeout(hardTimeout);
  console.log("");
  console.log("[manual-ws-check] SUMMARY");
  console.log(`  connected: true`);
  console.log(`  saw task_started ack: ${sawTaskStarted}`);
  console.log(`  saw a terminal message (task_complete or server_error): ${sawTerminalMessage}`);
  console.log(
    sawTaskStarted && sawTerminalMessage
      ? "[manual-ws-check] RESULT: PASS -- protocol.ts/kenzoClient.ts wire format interoperates with the real Python server."
      : "[manual-ws-check] RESULT: INCOMPLETE -- did not observe the expected message sequence."
  );
  ws.close();
  setTimeout(() => process.exit(sawTaskStarted && sawTerminalMessage ? 0 : 1), 200);
}

void sawTerminalMessage;
void sawTaskStarted;
