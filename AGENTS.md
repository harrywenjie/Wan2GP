# Agent Workflow

## Core Directives
- Treat `PROJECT_PLAN_LIVE.md` as the authoritative live session log, There are 4 sections in this file:
  - `## Objective`, this is high level directives serves as the project goal.
  - `## Project Roadmap`, this is the current project roadmap.
  - `## Validation Expectations`, this is the current validation instructions.
  - `## Immediate Next Actions`, this is a concise list of current tasks.  
- Use `README.md` and `docs/*` as references and update them as needed.
- Always follow `## Handoff Protocol` below when carrying out tasks.
- Ensure `## Immediate Next Actions` always lists the next concrete tasks before you stop.

---

## Daily Operation
- At session start, read `PROJECT_PLAN_LIVE.md`:
  - Review `docs/CONTEXT.md` and `docs/WORK_HISTORY.md` for detailed context and recent work.
  - Review `## Immediate Next Actions` for current tasks.
  - Explore relevant parts of the codebase to rebuild working context.

---

## Environment Setup
- The **human operator** prepares and activates the project `venv` before execution.
- Assume all required dependencies are installed; do **not** install or upgrade packages manually.
- If you encounter a missing or incompatible dependency, **notify the operator** instead of attempting installation.

---

## Coding Style
- Comment concisely — explain **intent**, not syntax. 
- Indent with 4 spaces, keep implementations lean, and document public entry points with concise docstrings.
- Add type hints wherever feasible to clarify interfaces and enable static checking.
- Use a unified logger infrastructure and route runtime output through it rather than direct prints.
- Use lowercase, underscore-separated filenames; reserve CamelCase for classes and snake_case for functions, tasks, and configuration keys.
- Favor relative paths so tooling behaves consistently across development environments.

---

## Validation
- Follow the live validation guidance documented in `## Validation Expectations`.
- Add minimal validation scripts only when necessary, and remove them once higher-level automation exists.

---

## Documentation & Communication
- Use Python chunked reads for files over 10240B to avoid CLI truncation.
- When a timestamp is needed, use the `date` command to get the system time.
- Keep sensitive artifacts (weights, media, tokens) **out of version control**.
- Document required paths and environment variables explicitly.

---

## Handoff Protocol
- Execute a coherent batch of related tasks from `## Immediate Next Actions` — enough for measurable progress but with a **focused** scope.
- After completion, replace the existing `docs/WORK_HISTORY.md` content with a concise (~800 token) summary of prior sessions, then append today’s detailed entry covering code changes.
- Remove finished tasks from `## Immediate Next Actions`.
- Add follow-up tasks or new tasks derived from the roadmap/objective to `## Immediate Next Actions` in proper order; 
- Keep `## Immediate Next Actions` as a concise, timeless checklist.
- Audit and prune `## Project Roadmap`, cleaning up stale milestones.
- Review `## Validation Expectations` and update as needed.
- Update `docs/APPENDIX_HEADLESS.md` and `docs/CLI.md` for changes.
- Record new insights or architectural notes in `docs/CONTEXT.md`; prune stale entries.
- Keep `PROJECT_PLAN_LIVE.md` clear, consistent, and easy for future agents to follow.