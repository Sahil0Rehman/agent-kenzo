/**
 * A minimal, structurally-faithful fake of the real `vscode` module --
 * just enough surface for `src/extension.ts` (and everything it wires up)
 * to run `activate()` outside the real VS Code/Electron host. Used both
 * directly (DI-style, in ideContext/editProposalHandler tests) and via
 * `mockVscodeModule.ts`'s `require("vscode")` interception (in
 * extension.test.ts).
 */

import sinon from "sinon";

export class FakeEventEmitter<T> {
  private listeners: Array<(e: T) => void> = [];
  event = (listener: (e: T) => void) => {
    this.listeners.push(listener);
    return {
      dispose: () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      },
    };
  };
  fire(e: T): void {
    for (const l of [...this.listeners]) l(e);
  }
  dispose(): void {
    this.listeners = [];
  }
}

export const FakeDiagnosticSeverity = { Error: 0, Warning: 1, Information: 2, Hint: 3 } as const;
export const FakeTreeItemCollapsibleState = { None: 0, Collapsed: 1, Expanded: 2 } as const;
export const FakeCodeActionKind = { QuickFix: "quickfix" } as const;
export const FakeViewColumn = { Active: -1 } as const;

export class FakeUri {
  private constructor(
    public scheme: string,
    public path: string,
    public fsPath: string
  ) {}
  static file(p: string): FakeUri {
    return new FakeUri("file", p, p);
  }
  static parse(s: string): FakeUri {
    const idx = s.indexOf(":");
    const scheme = s.slice(0, idx);
    const path = s.slice(idx + 1);
    return new FakeUri(scheme, path, path);
  }
  toString(): string {
    return `${this.scheme}:${this.path}`;
  }
}

export class FakePosition {
  constructor(
    public line: number,
    public character: number
  ) {}
}

function comparePositions(a: FakePosition, b: FakePosition): number {
  if (a.line !== b.line) return a.line - b.line;
  return a.character - b.character;
}

export class FakeRange {
  start: FakePosition;
  end: FakePosition;
  constructor(startLine: number, startChar: number, endLine: number, endChar: number) {
    this.start = new FakePosition(startLine, startChar);
    this.end = new FakePosition(endLine, endChar);
  }
  contains(pos: FakePosition): boolean {
    return comparePositions(this.start, pos) <= 0 && comparePositions(pos, this.end) <= 0;
  }
}

export class FakeSelection extends FakeRange {
  isEmpty: boolean;
  constructor(startLine: number, startChar: number, endLine: number, endChar: number) {
    super(startLine, startChar, endLine, endChar);
    this.isEmpty = startLine === endLine && startChar === endChar;
  }
  get active(): FakePosition {
    return this.end;
  }
}

export class FakeWorkspaceEdit {
  ops: Array<{ op: string; [k: string]: unknown }> = [];
  replace(uri: FakeUri, range: FakeRange, text: string): void {
    this.ops.push({ op: "replace", uri, range, text });
  }
  createFile(uri: FakeUri, options?: unknown): void {
    this.ops.push({ op: "createFile", uri, options });
  }
  insert(uri: FakeUri, position: FakePosition, text: string): void {
    this.ops.push({ op: "insert", uri, position, text });
  }
}

export class FakeTreeItem {
  description?: string;
  tooltip?: string;
  contextValue?: string;
  command?: unknown;
  iconPath?: unknown;
  constructor(
    public label: string,
    public collapsibleState: number
  ) {}
}

export class FakeThemeIcon {
  constructor(public id: string) {}
}

export class FakeCodeAction {
  command?: unknown;
  diagnostics?: unknown;
  constructor(
    public title: string,
    public kind: string
  ) {}
}

export interface FakeTextDocument {
  uri: FakeUri;
  getText(range?: FakeRange): string;
}

export interface FakeTextEditor {
  document: FakeTextDocument;
  selection: FakeSelection;
}

export interface FakeExtensionContext {
  extensionUri: FakeUri;
  subscriptions: Array<{ dispose(): unknown }>;
}

export interface CreateFakeVscodeOptions {
  workspaceFolders?: Array<{ uri: FakeUri }>;
  activeTextEditor?: FakeTextEditor;
  diagnostics?: Array<[FakeUri, Array<{ range: FakeRange; message: string; severity: number }>]>;
  showInformationMessageResult?: string | undefined;
}

export function createFakeVscode(options: CreateFakeVscodeOptions = {}) {
  const registeredCommands = new Map<string, (...args: unknown[]) => unknown>();
  const executeCommand = sinon.stub().callsFake(async (command: string, ...args: unknown[]) => {
    const handler = registeredCommands.get(command);
    if (handler) return handler(...args);
    return undefined;
  });

  const registerWebviewViewProvider = sinon.spy();
  const registerTreeDataProvider = sinon.spy();
  const registerTextDocumentContentProvider = sinon.spy();
  const registerCodeActionsProvider = sinon.spy();
  const showWarningMessage = sinon.spy();
  const showErrorMessage = sinon.spy();
  const showInformationMessage = sinon
    .stub()
    .resolves(options.showInformationMessageResult);
  const showTextDocument = sinon.stub().resolves(undefined);
  const createTerminal = sinon.stub().returns({ sendText: sinon.spy(), show: sinon.spy() });

  const registerCommand = sinon.spy((id: string, handler: (...args: unknown[]) => unknown) => {
    registeredCommands.set(id, handler);
    return { dispose: () => registeredCommands.delete(id) };
  });

  const fakeVscode = {
    EventEmitter: FakeEventEmitter,
    Uri: FakeUri,
    Position: FakePosition,
    Range: FakeRange,
    Selection: FakeSelection,
    WorkspaceEdit: FakeWorkspaceEdit,
    TreeItem: FakeTreeItem,
    ThemeIcon: FakeThemeIcon,
    CodeAction: FakeCodeAction,
    CodeActionKind: FakeCodeActionKind,
    TreeItemCollapsibleState: FakeTreeItemCollapsibleState,
    DiagnosticSeverity: FakeDiagnosticSeverity,
    ViewColumn: FakeViewColumn,
    window: {
      activeTextEditor: options.activeTextEditor,
      tabGroups: { all: [] as Array<{ tabs: Array<{ input?: { uri?: FakeUri } }> }> },
      onDidChangeActiveTextEditor: sinon.stub().returns({ dispose: () => undefined }),
      onDidChangeTextEditorSelection: sinon.stub().returns({ dispose: () => undefined }),
      registerWebviewViewProvider,
      registerTreeDataProvider,
      showWarningMessage,
      showErrorMessage,
      showInformationMessage,
      showTextDocument,
      createTerminal,
    },
    workspace: {
      workspaceFolders: options.workspaceFolders,
      getConfiguration: sinon.stub().returns({
        get: (_key: string, fallback: unknown) => fallback,
      }),
      asRelativePath: (fsPath: string) => fsPath,
      registerTextDocumentContentProvider,
      openTextDocument: sinon.stub().rejects(new Error("not found")),
      applyEdit: sinon.stub().resolves(true),
      fs: { stat: sinon.stub().rejects(new Error("not found")) },
    },
    languages: {
      getDiagnostics: sinon.stub().returns(options.diagnostics ?? []),
      registerCodeActionsProvider,
    },
    commands: {
      registerCommand,
      executeCommand,
    },
  };

  return { fakeVscode, registeredCommands, spies: { registerWebviewViewProvider, registerTreeDataProvider, registerTextDocumentContentProvider, registerCodeActionsProvider, registerCommand, showWarningMessage, showErrorMessage, showInformationMessage, showTextDocument, createTerminal, executeCommand } };
}

export function createFakeExtensionContext(): FakeExtensionContext {
  return {
    extensionUri: FakeUri.file("/fake/extension"),
    subscriptions: [],
  };
}
