import assert from "node:assert/strict";
import {
  buildIdeContext,
  collectDiagnostics,
  collectOpenTabs,
  registerIdeContextWatchers,
  VscodeLike,
} from "../src/ideContext";

function makeFakeVscode(overrides: Partial<VscodeLike> = {}): VscodeLike {
  const base: VscodeLike = {
    window: {
      activeTextEditor: undefined,
      tabGroups: { all: [] },
    },
    workspace: {
      workspaceFolders: undefined,
    },
    languages: {
      getDiagnostics: () => [],
    },
    DiagnosticSeverity: { Error: 0, Warning: 1, Information: 2, Hint: 3 },
  };
  return { ...base, ...overrides };
}

describe("ideContext.collectDiagnostics", () => {
  it("maps severities to wire strings and flattens per-uri diagnostics", () => {
    const vscodeApi = makeFakeVscode({
      languages: {
        getDiagnostics: () => [
          [
            { fsPath: "/repo/a.ts", toString: () => "file:/repo/a.ts" },
            [
              { range: { start: { line: 3, character: 0 }, end: { line: 3, character: 5 } }, message: "err", severity: 0 },
              { range: { start: { line: 5, character: 0 }, end: { line: 5, character: 1 } }, message: "warn", severity: 1 },
            ],
          ],
        ],
      },
    });
    const result = collectDiagnostics(vscodeApi);
    assert.deepEqual(result, [
      { file: "/repo/a.ts", line: 3, message: "err", severity: "error" },
      { file: "/repo/a.ts", line: 5, message: "warn", severity: "warning" },
    ]);
  });
});

describe("ideContext.collectOpenTabs", () => {
  it("flattens tab groups and skips tabs with no uri", () => {
    const vscodeApi = makeFakeVscode({
      window: {
        activeTextEditor: undefined,
        tabGroups: {
          all: [
            { tabs: [{ input: { uri: { fsPath: "/repo/a.ts", toString: () => "" } } }, { input: {} }] },
            { tabs: [{ input: { uri: { fsPath: "/repo/b.ts", toString: () => "" } } }] },
          ],
        },
      },
    });
    assert.deepEqual(collectOpenTabs(vscodeApi), ["/repo/a.ts", "/repo/b.ts"]);
  });
});

describe("ideContext.buildIdeContext", () => {
  it("produces the right shape with an active editor, a selection, and a workspace", async () => {
    const document = {
      uri: { fsPath: "/repo/a.ts", toString: () => "" },
      getText: (range?: unknown) => (range ? "selected text" : "full text"),
    };
    const vscodeApi = makeFakeVscode({
      window: {
        activeTextEditor: {
          document,
          selection: {
            isEmpty: false,
            start: { line: 2, character: 0 },
            end: { line: 4, character: 1 },
          },
        },
        tabGroups: { all: [] },
      },
      workspace: {
        workspaceFolders: [{ uri: { fsPath: "/repo", toString: () => "" } }],
      },
    });

    const execFn = async (_cmd: string, _opts: { cwd: string }) => ({
      stdout: " M src/a.ts\n",
      stderr: "",
    });

    const context = await buildIdeContext(vscodeApi, execFn);

    assert.equal(context.current_file, "/repo/a.ts");
    assert.equal(context.selection, "selected text");
    assert.equal(context.selection_start_line, 2);
    assert.equal(context.selection_end_line, 4);
    assert.equal(context.workspace_path, "/repo");
    assert.equal(context.git_status, "M src/a.ts");
    assert.deepEqual(context.open_tabs, []);
    assert.deepEqual(context.diagnostics, []);
  });

  it("omits selection fields when the selection is empty, and omits git_status on failure", async () => {
    const document = {
      uri: { fsPath: "/repo/a.ts", toString: () => "" },
      getText: () => "",
    };
    const vscodeApi = makeFakeVscode({
      window: {
        activeTextEditor: {
          document,
          selection: { isEmpty: true, start: { line: 0, character: 0 }, end: { line: 0, character: 0 } },
        },
        tabGroups: { all: [] },
      },
      workspace: {
        workspaceFolders: [{ uri: { fsPath: "/repo", toString: () => "" } }],
      },
    });

    const execFn = async () => {
      throw new Error("not a git repo");
    };

    const context = await buildIdeContext(vscodeApi, execFn);
    assert.equal(context.selection, undefined);
    assert.equal(context.selection_start_line, undefined);
    assert.equal(context.git_status, undefined);
  });

  it("omits workspace fields entirely when there is no workspace folder", async () => {
    const vscodeApi = makeFakeVscode();
    const context = await buildIdeContext(vscodeApi, async () => ({ stdout: "", stderr: "" }));
    assert.equal(context.workspace_path, undefined);
    assert.equal(context.git_status, undefined);
    assert.equal(context.current_file, undefined);
  });
});

describe("ideContext.registerIdeContextWatchers", () => {
  it("debounces rapid selection changes into a single buildIdeContext call", (done) => {
    let selectionListener: (() => void) | undefined;
    const vscodeApi = makeFakeVscode({
      window: {
        activeTextEditor: undefined,
        tabGroups: { all: [] },
        onDidChangeActiveTextEditor: () => ({ dispose: () => undefined }),
        onDidChangeTextEditorSelection: (listener: () => void) => {
          selectionListener = listener;
          return { dispose: () => undefined };
        },
      },
    });

    let callCount = 0;
    const disposable = registerIdeContextWatchers(
      vscodeApi,
      () => {
        callCount += 1;
      },
      20,
      async () => ({ stdout: "", stderr: "" })
    );

    assert.ok(selectionListener, "listener should have been registered");
    selectionListener!();
    selectionListener!();
    selectionListener!();

    setTimeout(() => {
      assert.equal(callCount, 1);
      disposable.dispose();
      done();
    }, 60);
  });
});
