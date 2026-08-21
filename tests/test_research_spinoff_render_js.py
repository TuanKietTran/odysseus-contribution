"""Deep Research "Discuss" spin-off primer must render as a system chip, not a chat bubble.

`POST /api/research/spinoff/<id>` (routes/research/research_routes.py) seeds the
new session with a `role="system"` message carrying the research report as
priming context, tagged `metadata={"research_spinoff_from": session_id}`. Before
this fix, `chatRenderer.addMessage` had no branch for that case, so it fell
through to the standard single-bubble path and rendered the primer as a normal
`msg-ai` chat bubble — indistinguishable from something the model said
(issue #6061).

`addMessage` itself is impractical to drive end-to-end in a bare Node
sandbox: it reaches into ~10 sibling modules (ui.js, markdown.js, tts-ai.js,
providers.js, settings.js, spinner.js, escMenuStack.js, panels.js,
model/matchKey.js, appConfig.js) for things like model-color palettes and
markdown rendering that have nothing to do with this branch. The existing
tests for this exact file (test_chat_tool_screenshot_xss.py,
test_agent_round_model_provenance_ui.py) hit the same wall and fall back to
source-derived checks for the same reason.

This test takes the middle ground allowed by TESTING_STANDARD.md's
"behavioral-first" exception: it extracts *only* the new branch (a
self-contained block with a narrow, stubbable dependency surface — just
`document.createElement`/`appendChild` and `markdownModule.processWithThinking`)
and actually executes it under Node with a minimal DOM stub, rather than
asserting on the surrounding source text.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


_RENDERER_PATH = Path(__file__).resolve().parents[1] / "static" / "js" / "chatRenderer.js"
_RENDERER_SOURCE = _RENDERER_PATH.read_text(encoding="utf-8")
_ROUTES_SOURCE = (
    Path(__file__).resolve().parents[1] / "routes" / "research" / "research_routes.py"
).read_text(encoding="utf-8")
_HAS_NODE = shutil.which("node") is not None

_START_MARKER = "// --- Research spin-off primer (system message) ---"
_END_MARKER = "// --- Standard single-bubble message ---"


def _spinoff_branch_source():
    start = _RENDERER_SOURCE.index(_START_MARKER)
    end = _RENDERER_SOURCE.index(_END_MARKER, start)
    assert start < end
    return _RENDERER_SOURCE[start:end]


def _run_node(source):
    proc = subprocess.run(
        ["node", "--input-type=module"],
        input=source,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_backend_and_frontend_agree_on_the_metadata_key():
    """Both sides of the contract must use the exact same tag."""
    assert 'metadata={"research_spinoff_from": session_id}' in _ROUTES_SOURCE
    assert 'role="system"' in _ROUTES_SOURCE
    assert "metadata?.research_spinoff_from" in _spinoff_branch_source()


def test_branch_precedes_the_standard_bubble_path():
    """The spin-off check must short-circuit before the msg-ai bubble is built."""
    start = _RENDERER_SOURCE.index(_START_MARKER)
    end = _RENDERER_SOURCE.index(_END_MARKER)
    assert start < end
    branch = _spinoff_branch_source()
    assert "return sysWrap;" in branch


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_spinoff_primer_renders_as_a_system_chip_not_an_ai_bubble():
    """Actually run the branch: role=system + research_spinoff_from must
    produce a '.msg.msg-system' element (the existing slash-command/toast
    chip style), never 'msg msg-ai'.
    """
    branch = _spinoff_branch_source()
    harness = "\n".join([
        "class FakeNode {",
        "  constructor(tag) { this.tag = tag; this.className = ''; this.textContent = ''; this._html = ''; this.children = []; }",
        "  appendChild(child) { this.children.push(child); return child; }",
        "  set innerHTML(v) { this._html = v; }",
        "  get innerHTML() { return this._html; }",
        "}",
        "function createElement(tag) { return new FakeNode(tag); }",
        "const document = { createElement };",
        "const markdownModule = { processWithThinking: (t) => '<p>' + t + '</p>' };",
        "const box = new FakeNode('div');",
        "",
        "function run(role, textRaw, metadata) {",
        branch,
        "  return null;",
        "}",
        "",
        "const primer = '[Research context — 2026-08-21]\\n\\nUse the report below.';",
        "const result = run('system', primer, { research_spinoff_from: 'rp-123' });",
        "console.log(JSON.stringify({",
        "  rendered: result !== null,",
        "  className: result ? result.className : null,",
        "  appendedToBox: box.children.includes(result),",
        "  summaryText: result ? result.children[0].children[0].children[0].textContent : null,",
        "}));",
    ])
    out = _run_node(harness)
    assert out["rendered"] is True
    assert out["className"] == "msg msg-system"
    assert out["appendedToBox"] is True
    assert out["summaryText"] == "Research context — 2026-08-21"


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_non_spinoff_system_messages_fall_through_untouched():
    """A plain assistant/user message, or a system message without the
    spin-off tag, must not be swallowed by this branch — it must fall
    through so the standard bubble path still handles it.
    """
    branch = _spinoff_branch_source()
    harness = "\n".join([
        "class FakeNode {",
        "  constructor(tag) { this.tag = tag; this.className = ''; this.children = []; }",
        "  appendChild(child) { this.children.push(child); return child; }",
        "}",
        "function createElement(tag) { return new FakeNode(tag); }",
        "const document = { createElement };",
        "const markdownModule = { processWithThinking: (t) => t };",
        "const box = new FakeNode('div');",
        "",
        "function run(role, textRaw, metadata) {",
        branch,
        "  return 'fell-through';",
        "}",
        "",
        "const results = {",
        "  assistant: run('assistant', 'hello', {}),",
        "  plainSystem: run('system', 'hello', {}),",
        "  userWithFlag: run('user', 'hello', { research_spinoff_from: 'rp-1' }),",
        "};",
        "console.log(JSON.stringify(results));",
    ])
    out = _run_node(harness)
    assert out == {
        "assistant": "fell-through",
        "plainSystem": "fell-through",
        "userWithFlag": "fell-through",
    }
