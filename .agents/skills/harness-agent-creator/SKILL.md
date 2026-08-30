---
name: harness-agent-creator
description: Create and maintain project-level subagents for Claude Code (.claude/agents/*.md), OpenCode (.opencode/agents/*.md), Codex (.codex/agents/*.toml) and GitHub (.github/agents/*.agent.md) from a single spec. Use when the user wants to create a new agent, add a subagent, scaffold an assistant for a harness, convert an agent to another harness, or asks about .claude/agents, .opencode/agents, .codex/agents, .github/agents, subagents, custom agents, or agent frontmatter/TOML. Also use when the user says create agent, new agent, agente, or subagente.
---

# Harness Agent Creator

One spec → four project agents. Fill the spec, verify against the official docs, then generate the native files under `.claude/agents/`, `.opencode/agents/`, `.codex/agents/` and `.github/agents/`.

## When to use

- New agent or subagent, or porting one across harnesses.
- Anything about `.claude/agents`, `.opencode/agents`, `.codex/agents`, `.github/agents` or agent frontmatter/TOML.

Not for a one-off prompt that doesn't need a file.

## Workflow

### 1. Discover

Check existing agents in the repo and `AGENTS.md` / `CLAUDE.md` if needed. Watch for name collisions.

### 2. Spec

Copy `references/spec-template.yaml` to `agent-spec.yaml` and fill it. Ask only what's missing — purpose, triggers / non-triggers, scope and constraints. If the user already gave the details, draft the spec and confirm it.

This spec is the source of truth. All four files are generated from it.

### 3. Verify against official docs

Fetch the current docs for each harness you target. Don't rely on memory.

- Claude Code: `https://code.claude.com/docs/en/sub-agents`
- OpenCode: `https://opencode.ai/docs/agents`
- Codex: `https://learn.chatgpt.com/docs/agent-configuration/subagents`
- GitHub: `https://docs.github.com/en/copilot/reference/custom-agents-configuration`

Note what you checked — the docs change. If a field is marked as ignored for a scope, don't use it.

### 4. Propose

Show the filled spec as a short proposal and wait for approval: name, one-line purpose, triggers, scope, one example, files you will create, docs you checked.

Don't write files before approval.

### 5. Generate

Render the native files from the spec.

- Claude Code: `references/claude-code.md` — `tools` allowlist
- OpenCode: `references/opencode.md` — `permission` map, `mode: subagent`
- Codex: `references/codex.md` — TOML with `name`, `description`, `developer_instructions`
- GitHub: `references/github.md` — `tools` allowlist, `argument-hint`, `user-invocable`

Keep `name` matching the filename. You can use `scripts/generate_agent.py --spec agent-spec.yaml --harnesses claude,opencode,codex,github` or write by hand.

If you created a new `agents/` directory, a restart may be needed for the watcher.

### 6. Validate

Run `scripts/validate_agent.py <path>` or the equivalent checks (`claude plugin validate .claude/agents`, `tomllib` for Codex). Fix and recheck before reporting. Then summarize what you created and how to verify it loads.

## References

| Need | File |
| --- | --- |
| Claude Code fields | `references/claude-code.md` |
| OpenCode fields | `references/opencode.md` |
| Codex fields | `references/codex.md` |
| GitHub fields | `references/github.md` |
| How the four map | `references/adapter-matrix.md` |
| Spec template | `references/spec-template.yaml` |

Only open what you need.

## Body — dense, descriptive, and explainable

The `description` triggers delegation, but the `body`/`developer_instructions` is what makes the agent precise and not generic. Do not write a minimal body.

Bad — too vague, the agent will be generic and miss the work:
```
body: |
  You are a reviewer. Focus on correctness and missing tests.
```

Good — dense, explains the job, the checks, the boundaries and how to report:
```
body: |
  You are the reviewer for NovaNN. You own one job: review recent changes
  for logic, error handling and test gaps. Check branching, preconditions,
  null/size-1/boundary, dispatching and status propagation. Cite
  file:line for every finding, include a minimal hardening idea without
  editing files, and say explicitly if there's nothing to flag.
  Work read-only and split gathering across parallel subagents when the
  diff is large.
```

Rules:
- Body must be longer and more specific than `description`. If `description` does the work and `body` is 2 lines, the agent will be vague.
- Explain the responsibility, the exact checks, the scope, what not to do, and the reporting format. A developer reading only the body should know how to do the job.
- Keep it dense and concrete (file patterns, checks, examples), not filler. No trivial over-explanations like "do not edit files unless asked" without context — explain why.
- Each agent's body is unique. Do not copy the same structure or skill table across agents.

## Notes

- One agent, one job. Split it if the scope creeps.
- Least privilege — start read-only, add `edit`/`bash` only if the spec needs it.
- `description` should say when to delegate, not what the product is.
- Use only fields that appear in the official docs.
