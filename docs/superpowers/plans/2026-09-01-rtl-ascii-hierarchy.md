# RTL ASCII Hierarchy Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise, annotated ASCII RTL instance hierarchy to the README and the authoritative design document.

**Architecture:** Keep the existing Mermaid hierarchy in `docs/heatvit.md` as the graphical source, and add the requested text tree immediately after it. Add the same tree to the README next to the repository/RTL overview so both entry points use identical module names and line-end `#` comments.

**Tech Stack:** Markdown, SystemVerilog RTL hierarchy, Git.

**Spec:** User request in this conversation; `docs/heatvit.md` §1.1 is the current as-built hierarchy source.

## Global Constraints

- Modify only documentation required for the requested ASCII hierarchy.
- Every tree item has a trailing `#` comment on the same line.
- Preserve existing Mermaid diagram and repository documentation structure.

---

### Task 1: Document the annotated hierarchy

**Files:**
- Modify: `README.md`
- Modify: `docs/heatvit.md`

**Interfaces:**
- Consumes: instantiated RTL hierarchy in `rtl/top/`, `rtl/compute/`, `rtl/selector/`, `rtl/common/`, and `rtl/memory/`.
- Produces: matching Markdown fenced `text` diagrams with `#` line-end annotations.

- [x] **Step 1: Add the tree to the README**

Place a `## RTL 设计层次（简图）` section after `## 仓库结构`, using the exact module names from the as-built hierarchy and one concise trailing `#` annotation per line.

- [x] **Step 2: Add the tree to the authoritative document**

Place a `#### ASCII 文本层次图（含简注）` subsection immediately after the existing Mermaid block in `docs/heatvit.md` §1.1. Retain the Mermaid block and use the same text tree as the README.

- [x] **Step 3: Verify documentation structure**

Run: `rg -n "RTL 设计层次|ASCII 文本层次|heatvit_tensor_executor.*#" README.md docs/heatvit.md`

Expected: both files contain the requested hierarchy headings and annotated executor line.

- [ ] **Step 4: Commit and push**

Run: `git add README.md docs/heatvit.md docs/superpowers/plans/2026-09-01-rtl-ascii-hierarchy.md; git commit -m "docs: add annotated RTL hierarchy"; git push origin main`

Expected: a clean working tree after a successful push.
