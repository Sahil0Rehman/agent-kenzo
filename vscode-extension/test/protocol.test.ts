import assert from "node:assert/strict";
import { parseServerMessage } from "../src/protocol";

describe("protocol.parseServerMessage", () => {
  it("parses a valid task_started message", () => {
    const msg = parseServerMessage({ type: "task_started", task_id: "t1", repo_path: "/repo" });
    assert.deepEqual(msg, { type: "task_started", task_id: "t1", repo_path: "/repo" });
  });

  it("parses a valid event message, defaulting missing data to {}", () => {
    const msg = parseServerMessage({
      type: "event",
      task_id: "t1",
      event_type: "tool_start",
      stage: "tool",
      message: "Calling run_command...",
    });
    assert.deepEqual(msg, {
      type: "event",
      task_id: "t1",
      event_type: "tool_start",
      stage: "tool",
      message: "Calling run_command...",
      data: {},
    });
  });

  it("parses a valid edit_proposal message", () => {
    const raw = {
      type: "edit_proposal",
      task_id: "t1",
      proposal_id: "p1",
      path: "src/foo.ts",
      action: "edit",
      diff: "--- a\n+++ b\n",
      new_content: "new content",
    };
    const msg = parseServerMessage(raw);
    assert.deepEqual(msg, raw);
  });

  it("rejects an edit_proposal with an invalid action", () => {
    const msg = parseServerMessage({
      type: "edit_proposal",
      task_id: "t1",
      proposal_id: "p1",
      path: "src/foo.ts",
      action: "delete",
      diff: "",
      new_content: "",
    });
    assert.equal(msg, null);
  });

  it("parses task_complete and fills in defaults for missing result fields", () => {
    const msg = parseServerMessage({
      type: "task_complete",
      task_id: "t1",
      result: { success: true, finished: true },
    });
    assert.equal(msg?.type, "task_complete");
    if (msg?.type === "task_complete") {
      assert.equal(msg.result.success, true);
      assert.equal(msg.result.finished, true);
      assert.deepEqual(msg.result.files_modified, []);
      assert.deepEqual(msg.result.files_seen, []);
      assert.equal(msg.result.final_response, "");
    }
  });

  it("parses server_error with optional task_id omitted", () => {
    const msg = parseServerMessage({ type: "server_error", message: "boom" });
    assert.deepEqual(msg, { type: "server_error", task_id: undefined, message: "boom" });
  });

  it("returns null for an unknown type", () => {
    assert.equal(parseServerMessage({ type: "not_a_real_type" }), null);
  });

  it("returns null for non-object input", () => {
    assert.equal(parseServerMessage("just a string"), null);
    assert.equal(parseServerMessage(null), null);
    assert.equal(parseServerMessage(42), null);
    assert.equal(parseServerMessage(["array"]), null);
  });

  it("returns null when required fields are missing", () => {
    assert.equal(parseServerMessage({ type: "task_started", task_id: "t1" }), null);
    assert.equal(parseServerMessage({ type: "server_error" }), null);
  });
});
