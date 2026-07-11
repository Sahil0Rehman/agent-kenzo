import assert from "node:assert/strict";
import {
  EditApplyVscodeLike,
  handleEditProposal,
  KenzoClientLike,
  VscodePositionLike,
  VscodeRangeLike,
  VscodeUriLike,
  VscodeWorkspaceEditLike,
} from "../src/editProposalHandler";
import { EditProposalEvent } from "../src/protocol";

class FakeWorkspaceEdit implements VscodeWorkspaceEditLike {
  ops: Array<{ op: string; [k: string]: unknown }> = [];
  replace(uri: VscodeUriLike, range: VscodeRangeLike, newText: string): void {
    this.ops.push({ op: "replace", uri, range, newText });
  }
  createFile(uri: VscodeUriLike, options?: { overwrite?: boolean; ignoreIfExists?: boolean }): void {
    this.ops.push({ op: "createFile", uri, options });
  }
  insert(uri: VscodeUriLike, position: VscodePositionLike, text: string): void {
    this.ops.push({ op: "insert", uri, position, text });
  }
}

function makeFakeVscodeApi(opts: {
  choice: "Accept" | "Reject" | undefined;
  existingContent?: string;
}): {
  api: EditApplyVscodeLike;
  createdEdits: FakeWorkspaceEdit[];
  appliedEdits: FakeWorkspaceEdit[];
  executedCommands: unknown[][];
} {
  const createdEdits: FakeWorkspaceEdit[] = [];
  const appliedEdits: FakeWorkspaceEdit[] = [];
  const executedCommands: unknown[][] = [];

  class TrackedWorkspaceEdit extends FakeWorkspaceEdit {
    constructor() {
      super();
      createdEdits.push(this);
    }
  }

  const api: EditApplyVscodeLike = {
    Uri: {
      file: (p: string) => ({ fsPath: p, toString: () => `file:${p}` }),
    },
    workspace: {
      openTextDocument: async (uri: VscodeUriLike) => {
        if (opts.existingContent === undefined) {
          throw new Error("file does not exist");
        }
        return { uri, getText: () => opts.existingContent as string };
      },
      applyEdit: async (edit: VscodeWorkspaceEditLike) => {
        appliedEdits.push(edit as FakeWorkspaceEdit);
        return true;
      },
      fs: { stat: async () => ({}) },
    },
    WorkspaceEdit: TrackedWorkspaceEdit,
    Position: class {
      constructor(
        public line: number,
        public character: number
      ) {}
    },
    Range: class {
      start: VscodePositionLike;
      end: VscodePositionLike;
      constructor(sl: number, sc: number, el: number, ec: number) {
        this.start = { line: sl, character: sc };
        this.end = { line: el, character: ec };
      }
    },
    window: {
      showInformationMessage: async () => opts.choice,
    },
    commands: {
      executeCommand: async (...args: unknown[]) => {
        executedCommands.push(args);
        return undefined;
      },
    },
  };

  return { api, createdEdits, appliedEdits, executedCommands };
}

class FakeKenzoClient implements KenzoClientLike {
  calls: Array<{ proposalId: string; decision: "accept" | "reject"; applied: boolean }> = [];
  resolveEditProposal(proposalId: string, decision: "accept" | "reject", applied: boolean): void {
    this.calls.push({ proposalId, decision, applied });
  }
}

const diffProvider = {
  proposedUri: (p: EditProposalEvent) => ({ fsPath: `proposed:${p.proposal_id}`, toString: () => `proposed:${p.proposal_id}` }),
  beforeUri: (p: EditProposalEvent) => ({ fsPath: `before:${p.proposal_id}`, toString: () => `before:${p.proposal_id}` }),
};

const editProposal: EditProposalEvent = {
  type: "edit_proposal",
  task_id: "t1",
  proposal_id: "p1",
  path: "src/foo.ts",
  action: "edit",
  diff: "--- a\n+++ b\n",
  new_content: "new content here",
};

describe("editProposalHandler.handleEditProposal", () => {
  it("on accept: shows the diff, applies via WorkspaceEdit with the new content, and sends edit_decision(applied:true)", async () => {
    const { api, createdEdits, appliedEdits, executedCommands } = makeFakeVscodeApi({
      choice: "Accept",
      existingContent: "old content",
    });
    const client = new FakeKenzoClient();

    const outcome = await handleEditProposal(editProposal, api, client, diffProvider);

    assert.equal(outcome.decision, "accept");
    assert.equal(executedCommands.length, 1);
    assert.equal(executedCommands[0][0], "vscode.diff");

    assert.equal(createdEdits.length, 1);
    assert.equal(appliedEdits.length, 1);
    assert.equal(appliedEdits[0], createdEdits[0]);
    const replaceOp = createdEdits[0].ops.find((o) => o.op === "replace");
    assert.ok(replaceOp, "expected a replace op for an edit action");
    assert.equal(replaceOp!.newText, "new content here");

    assert.deepEqual(client.calls, [{ proposalId: "p1", decision: "accept", applied: true }]);
  });

  it("on accept for a create action: uses createFile + insert, not replace", async () => {
    const { api, createdEdits } = makeFakeVscodeApi({ choice: "Accept" });
    const client = new FakeKenzoClient();
    const createProposal: EditProposalEvent = { ...editProposal, action: "create" };

    await handleEditProposal(createProposal, api, client, diffProvider);

    const ops = createdEdits[0].ops.map((o) => o.op);
    assert.deepEqual(ops, ["createFile", "insert"]);
  });

  it("on reject: does NOT call WorkspaceEdit/applyEdit, and sends edit_decision(applied:false)", async () => {
    const { api, createdEdits, appliedEdits } = makeFakeVscodeApi({
      choice: "Reject",
      existingContent: "old content",
    });
    const client = new FakeKenzoClient();

    const outcome = await handleEditProposal(editProposal, api, client, diffProvider);

    assert.equal(outcome.decision, "reject");
    assert.equal(createdEdits.length, 0);
    assert.equal(appliedEdits.length, 0);
    assert.deepEqual(client.calls, [{ proposalId: "p1", decision: "reject", applied: false }]);
  });

  it("treats a dismissed prompt (undefined choice) as a rejection, not an accept", async () => {
    const { api, appliedEdits } = makeFakeVscodeApi({ choice: undefined, existingContent: "old" });
    const client = new FakeKenzoClient();

    const outcome = await handleEditProposal(editProposal, api, client, diffProvider);

    assert.equal(outcome.decision, "reject");
    assert.equal(appliedEdits.length, 0);
    assert.deepEqual(client.calls, [{ proposalId: "p1", decision: "reject", applied: false }]);
  });
});
