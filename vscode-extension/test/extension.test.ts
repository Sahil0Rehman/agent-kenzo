import assert from "node:assert/strict";
import { installFakeVscodeModule, uninstallFakeVscodeModule } from "./mockVscodeModule";
import { createFakeExtensionContext, createFakeVscode, FakeUri } from "./fakeVscode";

describe("extension.activate (against a fully faked `vscode` module)", () => {
  let context: ReturnType<typeof createFakeExtensionContext>;
  let fake: ReturnType<typeof createFakeVscode>;

  afterEach(() => {
    // Disconnects KenzoClient (clears any pending reconnect timer / closes
    // the socket) and disposes every other registered disposable.
    for (const d of context.subscriptions) {
      try {
        d.dispose();
      } catch {
        /* best-effort cleanup */
      }
    }
    uninstallFakeVscodeModule();
    // Force a fresh module instance (with a fresh `vscode` binding) next test.
    delete require.cache[require.resolve("../src/extension")];
    delete require.cache[require.resolve("../src/kenzoClient")];
    delete require.cache[require.resolve("../src/chatViewProvider")];
    delete require.cache[require.resolve("../src/changesViewProvider")];
    delete require.cache[require.resolve("../src/memoryViewProvider")];
    delete require.cache[require.resolve("../src/diffContentProvider")];
    delete require.cache[require.resolve("../src/terminalManager")];
    delete require.cache[require.resolve("../src/editProposalHandler")];
    delete require.cache[require.resolve("../src/ideContext")];
  });

  it("registers all 4 spec-required commands plus the show-terminal/open-diff helper commands", () => {
    fake = createFakeVscode({
      workspaceFolders: [{ uri: FakeUri.file("/repo") }],
    });
    installFakeVscodeModule(fake.fakeVscode);
    context = createFakeExtensionContext();

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const extensionModule = require("../src/extension");
    extensionModule.activate(context);

    const registeredIds = fake.spies.registerCommand.getCalls().map((c) => c.args[0]);
    for (const expected of [
      "kenzo.chat",
      "kenzo.explainFile",
      "kenzo.fixError",
      "kenzo.reviewSelection",
      "kenzo.showTerminal",
    ]) {
      assert.ok(registeredIds.includes(expected), `expected ${expected} to be registered`);
    }
  });

  it("registers the three activity-bar views (chat webview, changes tree, memory tree)", () => {
    fake = createFakeVscode({ workspaceFolders: [{ uri: FakeUri.file("/repo") }] });
    installFakeVscodeModule(fake.fakeVscode);
    context = createFakeExtensionContext();

    const extensionModule = require("../src/extension");
    extensionModule.activate(context);

    assert.equal(fake.spies.registerWebviewViewProvider.callCount, 1);
    assert.equal(fake.spies.registerWebviewViewProvider.getCall(0).args[0], "kenzo.chatView");

    const treeIds = fake.spies.registerTreeDataProvider.getCalls().map((c) => c.args[0]);
    assert.deepEqual(treeIds.sort(), ["kenzo.changesView", "kenzo.memoryView"]);
  });

  it("registers both virtual-content-provider schemes used by the diff flow", () => {
    fake = createFakeVscode({ workspaceFolders: [{ uri: FakeUri.file("/repo") }] });
    installFakeVscodeModule(fake.fakeVscode);
    context = createFakeExtensionContext();

    const extensionModule = require("../src/extension");
    extensionModule.activate(context);

    const schemes = fake.spies.registerTextDocumentContentProvider.getCalls().map((c) => c.args[0]);
    assert.deepEqual(schemes.sort(), ["kenzo-before", "kenzo-proposal"]);
  });

  it("kenzo.explainFile shows a warning instead of starting a task when there is no active editor", async () => {
    fake = createFakeVscode({ workspaceFolders: [{ uri: FakeUri.file("/repo") }] });
    installFakeVscodeModule(fake.fakeVscode);
    context = createFakeExtensionContext();

    const extensionModule = require("../src/extension");
    extensionModule.activate(context);

    await fake.spies.executeCommand("kenzo.explainFile");
    assert.equal(fake.spies.showWarningMessage.callCount, 1);
  });
});
