# Agent Workflow

## Mission
Transform **Wan2GP** into a **lean, headless, command-line video generator**.  
All actions should move the codebase toward minimalism, reproducibility, and deterministic CLI operation.

### Core Directives
- Remove **Gradio** and all GUI-related dependencies; no legacy UI layers should remain.
- Retire **Docker-specific tooling** and documentation in favor of direct CLI workflows.
- Adopt **deliberate removals**: stage proposals and rationales in `PROJECT_PLAN_LIVE.md` before execution.

---

## Daily Operation
- At session start, read `PROJECT_PLAN_LIVE.md`:
  - Review `docs/CONTEXT.md` and `docs/WORK_HISTORY.md` for detailed context and recent work.
  - Review `## Immediate Next Actions` for current tasks.
  - Explore relevant parts of the codebase to rebuild working context.

---

## Environment Setup
- The **human operator** prepares and activates the project `venv` before execution.
- Assume all required dependencies are installed; do **not** install or upgrade packages yourself.
- If you encounter a missing or incompatible dependency, **notify the operator** instead of attempting installation.

---

## Code & Development Standards
- Decouple runtime orchestration from presentation remnants:
  - Move queue management, scheduling, and I/O into CLI modules (`cli/`).
  - Delete Gradio helpers (`shared/gradio`, plugins, web assets`).
  - Replace UI calls with structured logging, progress output, or exceptions.
- Keep model weights, presets, and defaults in their current directories, but **expose them via CLI flags** rather than UI configs.
- Prefer **pure-Python adapters**; guard GPU-accelerated ops behind capability checks.
- Comment concisely — explain **intent**, not syntax. Focus on non-obvious control flow (scheduler handoffs, precision transitions, etc.).
- Ensure all modules run cleanly **without** environment variables or assumptions from the legacy UI layer.
- Maintain idempotent entry points (`cli/generate.py`) for testing and automation.
- Discuss design decisions (architecture, dependency pruning, model coverage) before major edits.
- When refactoring or removing code, preserve debuggability through logging or CLI flags.

---

## Validation
- Follow the live validation guidance documented in `PROJECT_PLAN_LIVE.md` (see `## Validation Expectations`) and record outcomes in **`docs/WORK_HISTORY.md`**.
- Add minimal validation scripts only when necessary; remove them once higher-level automation exists.

---

## Documentation & Communication
- Treat `PROJECT_PLAN_LIVE.md` as the authoritative session log; use `README.md` and `docs/*` (e.g. `docs/CLI.md`, `docs/APPENDIX_HEADLESS.md`) as the current headless CLI references and update them when workflows change.
- Use Python chunked reads for files over 10240B to avoid CLI truncation.
- Record new insights or architectural notes in `docs/CONTEXT.md`, use `docs/WORK_HISTORY.md` for detailed session logs, and keep `PROJECT_PLAN_LIVE.md` focused on current objectives, roadmap, and task lists.
- Follow `## Handoff Protocol` in `PROJECT_PLAN_LIVE.md`; ensure `## Immediate Next Actions` always lists the next concrete tasks before you stop.
- Keep sensitive artifacts (weights, media, tokens) **out of version control**; document required paths and environment variables explicitly.
