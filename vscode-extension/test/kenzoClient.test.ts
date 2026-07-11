import assert from "node:assert/strict";
import { WebSocketServer, WebSocket as WsClient } from "ws";
import { AddressInfo } from "node:net";
import { KenzoClient } from "../src/kenzoClient";
import * as protocol from "../src/protocol";

function startServer(): Promise<{ wss: WebSocketServer; port: number }> {
  return new Promise((resolve) => {
    const wss = new WebSocketServer({ port: 0, path: "/ws" });
    wss.on("listening", () => {
      const { port } = wss.address() as AddressInfo;
      resolve({ wss, port });
    });
  });
}

describe("KenzoClient (against a real in-process ws server)", () => {
  let wss: WebSocketServer;
  let port: number;

  beforeEach(async () => {
    const started = await startServer();
    wss = started.wss;
    port = started.port;
  });

  afterEach((done) => {
    wss.close(() => done());
  });

  it("connects, sends start_task, and receives task_started + event + task_complete", (done) => {
    wss.on("connection", (socket: WsClient) => {
      socket.on("message", (raw) => {
        const msg = JSON.parse(raw.toString()) as protocol.ClientMessage;
        assert.equal(msg.type, "start_task");
        if (msg.type !== "start_task") return;
        assert.equal(msg.repo_path, "/repo");
        assert.equal(msg.request, "do the thing");

        socket.send(JSON.stringify({ type: "task_started", task_id: "t1", repo_path: "/repo" }));
        socket.send(
          JSON.stringify({
            type: "event",
            task_id: "t1",
            event_type: "thinking",
            stage: "controller",
            message: "Thinking...",
            data: {},
          })
        );
        socket.send(
          JSON.stringify({
            type: "task_complete",
            task_id: "t1",
            result: {
              success: true,
              finished: true,
              final_response: "done",
              changes: [],
              files_modified: [],
              files_seen: [],
            },
          })
        );
      });
    });

    const client = new KenzoClient({ host: "127.0.0.1", port });
    const received: string[] = [];

    client.on("task_started", (msg: protocol.TaskStartedEvent) => {
      received.push("task_started");
      assert.equal(msg.task_id, "t1");
    });
    client.on("event", (msg: protocol.AgentEventMessage) => {
      received.push("event");
      assert.equal(msg.message, "Thinking...");
    });
    client.on("task_complete", (msg: protocol.TaskCompleteEvent) => {
      received.push("task_complete");
      assert.equal(msg.result.success, true);
      assert.deepEqual(received, ["task_started", "event", "task_complete"]);
      client.disconnect();
      done();
    });
    client.on("connected", () => {
      client.startTask("/repo", "do the thing");
    });

    client.connect();
  });

  it("round-trips edit_decision and ide_context messages", (done) => {
    const serverReceived: protocol.ClientMessage[] = [];
    wss.on("connection", (socket: WsClient) => {
      socket.on("message", (raw) => {
        serverReceived.push(JSON.parse(raw.toString()));
        if (serverReceived.length === 2) {
          assert.equal(serverReceived[0].type, "edit_decision");
          assert.equal(serverReceived[1].type, "ide_context");
          client.disconnect();
          done();
        }
      });
    });

    const client = new KenzoClient({ host: "127.0.0.1", port });
    client.on("connected", () => {
      client.resolveEditProposal("p1", "accept", true);
      client.sendIdeContext("/repo", {
        open_tabs: ["a.ts"],
        diagnostics: [],
        workspace_path: "/repo",
      });
    });
    client.connect();
  });

  it("emits parse_error and does not crash on malformed JSON from the server", (done) => {
    wss.on("connection", (socket: WsClient) => {
      socket.send("not json{{{");
    });
    const client = new KenzoClient({ host: "127.0.0.1", port });
    client.on("parse_error", () => {
      client.disconnect();
      done();
    });
    client.connect();
  });

  it("emits send_failed instead of throwing when sending while disconnected", () => {
    const client = new KenzoClient({ host: "127.0.0.1", port: port + 1 });
    let failed = false;
    client.on("send_failed", () => {
      failed = true;
    });
    client.startTask("/repo", "hello");
    assert.equal(failed, true);
  });

  it("includes image_data/image_format in start_task when an image is passed", (done) => {
    wss.on("connection", (socket: WsClient) => {
      socket.on("message", (raw) => {
        const msg = JSON.parse(raw.toString()) as protocol.ClientMessage;
        assert.equal(msg.type, "start_task");
        if (msg.type !== "start_task") return;
        assert.equal(msg.repo_path, "/repo");
        assert.equal(msg.request, "");
        assert.equal(msg.image_data, "ZmFrZWJhc2U2NA==");
        assert.equal(msg.image_format, "png");
        client.disconnect();
        done();
      });
    });

    const client = new KenzoClient({ host: "127.0.0.1", port });
    client.on("connected", () => {
      client.startTask("/repo", "", undefined, undefined, {
        data: "ZmFrZWJhc2U2NA==",
        format: "png",
      });
    });
    client.connect();
  });

  it("omits image_data/image_format from start_task when no image is passed (matching task_id/ide_context)", (done) => {
    wss.on("connection", (socket: WsClient) => {
      socket.on("message", (raw) => {
        const parsed = JSON.parse(raw.toString()) as Record<string, unknown>;
        assert.equal(parsed.type, "start_task");
        assert.equal("image_data" in parsed, false);
        assert.equal("image_format" in parsed, false);
        assert.equal("task_id" in parsed, false);
        assert.equal("ide_context" in parsed, false);
        client.disconnect();
        done();
      });
    });

    const client = new KenzoClient({ host: "127.0.0.1", port });
    client.on("connected", () => {
      client.startTask("/repo", "do the thing");
    });
    client.connect();
  });
});
