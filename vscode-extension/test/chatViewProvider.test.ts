import assert from "node:assert/strict";
import sinon from "sinon";
import { installFakeVscodeModule, uninstallFakeVscodeModule } from "./mockVscodeModule";
import { createFakeVscode, FakeUri } from "./fakeVscode";

/**
 * Minimal fake of the `vscode.WebviewView` surface that
 * `KenzoChatViewProvider.resolveWebviewView` touches: `webview.options`,
 * `webview.html`, `webview.onDidReceiveMessage`, `webview.postMessage`.
 * The webview's own in-page JS (FileReader/DOM logic in `renderHtml()`'s
 * `<script>`) is NOT exercised here -- there is no real webview DOM under
 * mocha/node, so that piece is covered only by manual reasoning (see the
 * task report).
 */
function createFakeWebviewView() {
  let receivedHandler: ((msg: unknown) => unknown) | undefined;
  const posted: Array<Record<string, unknown>> = [];
  const webview = {
    options: undefined as unknown,
    html: "",
    onDidReceiveMessage: (cb: (msg: unknown) => unknown) => {
      receivedHandler = cb;
      return { dispose: () => undefined };
    },
    postMessage: (msg: Record<string, unknown>) => {
      posted.push(msg);
      return Promise.resolve(true);
    },
  };
  const view = { webview, show: sinon.spy() };
  return {
    view,
    posted,
    trigger: (msg: unknown) => receivedHandler?.(msg),
  };
}

describe("KenzoChatViewProvider (against a fully faked `vscode` module)", () => {
  let fake: ReturnType<typeof createFakeVscode>;

  beforeEach(() => {
    fake = createFakeVscode({ workspaceFolders: [{ uri: FakeUri.file("/repo") }] });
    installFakeVscodeModule(fake.fakeVscode);
    delete require.cache[require.resolve("../src/chatViewProvider")];
  });

  afterEach(() => {
    uninstallFakeVscodeModule();
    delete require.cache[require.resolve("../src/chatViewProvider")];
  });

  function makeProvider(getRepoPath: () => string | undefined = () => "/repo") {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { KenzoChatViewProvider } = require("../src/chatViewProvider");
    const startTaskCalls: unknown[][] = [];
    const fakeClient = {
      startTask: (...args: unknown[]) => startTaskCalls.push(args),
    };
    const provider = new KenzoChatViewProvider(
      FakeUri.file("/ext"),
      fakeClient,
      getRepoPath,
      async () => undefined
    );
    return { provider, startTaskCalls, fakeClient };
  }

  it("passes an image through to KenzoClient.startTask with an empty request string", async () => {
    const { provider, startTaskCalls } = makeProvider();
    const fakeView = createFakeWebviewView();
    provider.resolveWebviewView(fakeView.view);

    await fakeView.trigger({ type: "submit", text: "", image: { data: "abc123", format: "png" } });

    assert.equal(startTaskCalls.length, 1);
    const [repoPath, request, taskId, ideContext, image] = startTaskCalls[0];
    assert.equal(repoPath, "/repo");
    assert.equal(request, "");
    assert.equal(taskId, undefined);
    assert.equal(ideContext, undefined);
    assert.deepEqual(image, { data: "abc123", format: "png" });
  });

  it("does not gate sending on non-empty text when an image is attached (whitespace-only text + image)", async () => {
    const { provider, startTaskCalls } = makeProvider();
    const fakeView = createFakeWebviewView();
    provider.resolveWebviewView(fakeView.view);

    await fakeView.trigger({ type: "submit", text: "   ", image: { data: "xyz", format: "jpg" } });

    assert.equal(startTaskCalls.length, 1);
  });

  it("posts a '(screenshot attached)' placeholder to the webview transcript when text is empty but an image is attached", async () => {
    const { provider } = makeProvider();
    const fakeView = createFakeWebviewView();
    provider.resolveWebviewView(fakeView.view);

    await fakeView.trigger({ type: "submit", text: "", image: { data: "abc123", format: "png" } });

    const userMessage = fakeView.posted.find((m) => m.type === "user_message");
    assert.ok(userMessage, "expected a user_message to be posted to the webview");
    assert.equal(userMessage!.text, "(screenshot attached)");
  });

  it("does nothing when both text and image are empty/absent", async () => {
    const { provider, startTaskCalls } = makeProvider();
    const fakeView = createFakeWebviewView();
    provider.resolveWebviewView(fakeView.view);

    await fakeView.trigger({ type: "submit", text: "   " });

    assert.equal(startTaskCalls.length, 0);
    assert.equal(fakeView.posted.length, 0);
  });

  it("still warns and does not call startTask when there's no workspace folder, even with an image attached", async () => {
    const { provider, startTaskCalls } = makeProvider(() => undefined);
    const fakeView = createFakeWebviewView();
    provider.resolveWebviewView(fakeView.view);

    await fakeView.trigger({ type: "submit", text: "", image: { data: "abc", format: "jpg" } });

    assert.equal(startTaskCalls.length, 0);
    assert.equal(fake.spies.showWarningMessage.callCount, 1);
  });

  it("omits the image argument to KenzoClient.startTask when no image is attached (unchanged text-only path)", async () => {
    const { provider, startTaskCalls } = makeProvider();
    const fakeView = createFakeWebviewView();
    provider.resolveWebviewView(fakeView.view);

    await fakeView.trigger({ type: "submit", text: "hello" });

    assert.equal(startTaskCalls.length, 1);
    assert.equal(startTaskCalls[0][1], "hello");
    assert.equal(startTaskCalls[0][4], undefined);
  });
});
