# Contributing Guide

## Branching model

- `main` is **protected** — nobody pushes directly to it, including you. All changes go
  through a pull request (PR) and need at least one review before merging.
- One branch per task/issue, named `<type>/<short-description>`, e.g.:
  - `feat/notes-extraction`
  - `fix/cxr-timestamp-bug`
  - `exp/fusemoe-baseline-repro`
- Branch off the latest `main`, keep branches small and focused on one issue — don't bundle
  unrelated changes into one PR, it makes review slower and harder to catch mistakes.

## Task assignment workflow

1. Every task lives as a **GitHub Issue**, created from the templates in
   `.github/ISSUE_TEMPLATE/`. Use the milestone checklist in `PROJECT_CONTEXT.md` §7 as the
   source list of tasks to turn into issues.
2. Issues go on the **Project board** (Kanban: Backlog → In Progress → In Review → Done).
   Assign yourself (or get assigned) before starting — this avoids two people accidentally
   working on the same thing.
3. Reference the issue number in your branch/PR (e.g. `feat/notes-extraction (closes #12)`).

## Pull request checklist (fill this in every PR description)

- [ ] What does this PR do, in one or two sentences?
- [ ] Which issue does it close?
- [ ] How did you test it? (unit test, ran on a small sample cohort, sanity-checked output shape, etc.)
- [ ] Did you update `PROJECT_CONTEXT.md` §7 status if this completes a milestone item?
- [ ] Any assumptions or TODOs left for the reviewer to know about?

## Review process

- I (repo owner) review every PR before merge. Expect comments — this is normal, not a sign
  something's wrong.
- Please respond to every comment (even just "done" or "disagree because X") rather than
  silently pushing a fix — makes re-review much faster.
- Don't merge your own PR, even if you think it's ready — wait for approval.

## Before you start coding: read `PROJECT_CONTEXT.md`

Every file you touch should already have a header comment explaining its purpose and what's
left to do (see the placeholder files already in the repo for the format). If a file you need
doesn't have one yet, add one as part of your PR — this keeps the "explain the whole project
to an AI assistant" cost low for everyone, including future you.

## Using AI coding assistants on this repo

Paste the full contents of `PROJECT_CONTEXT.md` into your assistant's context before asking
for help. It has the gap, the solution, the task definition, and the repo map — this alone
resolves 90% of "wait, what is this project actually doing" confusion. If a file's header
comment references something not covered in `PROJECT_CONTEXT.md`, flag it in your PR so we
can update the context file — it should always be sufficient on its own.
