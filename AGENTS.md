# Agent Workflow

## Core Directives
- Treat `PROJECT_PLAN_LIVE.md` as the authoritative live session log, There are 5 sections in this file:
  - `## Objective`, this is high level directives serves as the project goal.
  - `## Project Roadmap`, this is the current project roadmap.
  - `## Validation Expectations`, this is the current validation instructions.
  - `## Project Specific Live Docs`, this is the list of live docs specific to the current project.
  - `## Immediate Next Actions`, this is a concise list of current tasks.  
- Use `README.md` and `docs/*` as references and update them as needed.
- Always follow `## Handoff Protocol` below when carrying out tasks.
- Ensure `## Immediate Next Actions` always lists the next concrete tasks before you stop.

---

## Daily Operation
- At session start, read `PROJECT_PLAN_LIVE.md`:
  - Review `LIVE_CONTEXT.md` and `SESSION_LOG.md` for detailed context and recent work.
  - Review `## Immediate Next Actions` for current tasks.
  - Explore relevant parts of the codebase to rebuild working context.
- Refresh `## Date` in `PROJECT_PLAN_LIVE.md` via `date +"%Y-%m-%d"`; use it for session timestamps.

---

## Environment Setup
- The **human operator** prepares and activates the project `venv` before execution.
- Assume all required dependencies are installed; do **not** install or upgrade packages manually.
- If you encounter a missing or incompatible dependency, **notify the operator** instead of attempting installation.

---

## Coding Style
- Comment concisely — explain intent, not syntax.  
- Keep implementations lean and document public entry points clearly.  
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
- Keep sensitive artifacts (weights, media, tokens) **out of version control**.
- Document required paths and environment variables explicitly.

---

## Handoff Protocol
- Execute a coherent batch of related tasks from `## Immediate Next Actions` — enough for measurable progress but with a **focused** scope.
- After completion, replace the existing `SESSION_LOG.md` content with a concise (~800 token) summary of prior sessions, then append today’s detailed entry covering code changes.
- Remove finished tasks from `## Immediate Next Actions`.
- Add follow-up tasks or new tasks derived from the roadmap/objective to `## Immediate Next Actions` in proper order; 
- Keep `## Immediate Next Actions` as a concise, timeless checklist.
- Audit and prune `## Project Roadmap`, cleaning up stale milestones.
- Review `## Validation Expectations` and update as needed.
- Update docs listed under `## Project Specific Live Docs` for changes if any.
- Record new insights or architectural notes in `LIVE_CONTEXT.md`; prune stale entries.
- Keep `PROJECT_PLAN_LIVE.md` clear, consistent, and easy for future agents to follow.