/**
 * Test-only utility: intercepts Node's module resolution so that
 * `require("vscode")` (what `import * as vscode from "vscode"` compiles
 * to under CommonJS) resolves to a fake object instead of throwing --
 * there is no real `vscode` module outside the VS Code extension host.
 * This lets `out/src/extension.ts` (and anything else that imports
 * `vscode`) be `require()`'d directly by plain Mocha/Node, so
 * `activate()` itself can be exercised in tests.
 */

import Module from "module";

const FAKE_ID = "\0kenzo-fake-vscode";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ModuleAny = Module as any;
const originalResolveFilename = ModuleAny._resolveFilename;

let installed = false;

export function installFakeVscodeModule(fakeVscode: unknown): void {
  if (!installed) {
    ModuleAny._resolveFilename = function (request: string, ...rest: unknown[]) {
      if (request === "vscode") return FAKE_ID;
      return originalResolveFilename.call(this, request, ...rest);
    };
    installed = true;
  }
  ModuleAny._cache[FAKE_ID] = {
    id: FAKE_ID,
    filename: FAKE_ID,
    loaded: true,
    exports: fakeVscode,
  };
}

export function uninstallFakeVscodeModule(): void {
  if (installed) {
    ModuleAny._resolveFilename = originalResolveFilename;
    installed = false;
  }
  delete ModuleAny._cache[FAKE_ID];
}
