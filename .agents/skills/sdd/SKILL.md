---
name: sdd
description: Break an implementation idea into a detailed NovaNN specification under specs/. Use this skill whenever the user mentions spec, specify, specification, requirements, user stories, acceptance criteria, breaking down a feature, detailing an idea, or working in the specs/ folder, even if they do not say SDD or Spec Driven Development.
---

# SDD Spec Author

Turn a raw implementation idea into a modular, reviewable specification in `specs/`. Spec-only: this skill never writes code, plans tasks, or implements.

## Local-only rule (read first)

`specs/` is local scratch. It is never committed, never pushed, and never versioned: every clone creates its own specs for its own implementations, so a spec written here does not exist anywhere else.

Because of that, implementation code must never reference a spec at any time. No `specs/` paths, no folder numbers, no `REQ`/`AC`/`ERR` IDs, no spec titles, and no copied spec text in source files, headers, tests, build files, or code comments. Restate any needed rationale in the code's own terms without citing the spec. A reference to something the next clone will never have is a dangling pointer in the repository: it confuses every future reader and breaks the moment the local folder is deleted or renumbered.

## When to use

- New feature, kernel, dtype, backend, API, or refactor needs a written spec.
- User says: "spec this", "break this down", "detail requirements", "write the spec for X".
- Any work targeting `specs/` folder.

## When not to use

- User wants a build plan, task list, or code. Stop after the spec and say so.

## Workflow

Follow these steps in order.

1. Intake: capture the idea in one paragraph (what and why, not how).
2. Clarify: if scope, interfaces, dtypes, devices, or acceptance are ambiguous, ask up to 5 targeted questions. Encode answers back into the spec files. Do not invent requirements.
3. Scaffold: pick the next free prefix by listing `specs/` (format `NNN-kebab-name`, e.g. `specs/001-cpu-matmul/`). Create the folder.
4. Shape the contract: choose the file set for this implementation from `references/spec-template.md` (default is the four-file set; add, split, or omit files per its selection rules). Write only the chosen files.
5. Self-check against `references/checklist.md`. Fix failures at the source file before finishing.
6. Report: list created paths and open questions. Remind the user that `specs/` is local-only and unversioned. Never claim implementation coverage.

## Folder contract

The folder name is fixed; the interior file set varies by implementation:

```
specs/NNN-kebab-name/
├── spec.md         # always required: scope, context, user stories, non-goals
└── <chosen modules, e.g. requirements.md, constraints.md, acceptance.md, api.md>
```

Rules:

- One feature per folder. Split unrelated behaviors into separate folders.
- Kebab-case name, zero-padded number (`001`, `002`).
- `spec.md` is always required. Every other file is optional and selected per implementation (see selection rules in `references/spec-template.md`).
- Default for a standard feature: `spec.md`, `requirements.md`, `constraints.md`, `acceptance.md`. Shrink for trivial work (e.g. `spec.md` + `acceptance.md`), expand for complex work (split out `api.md`, `data-model.md`, `errors.md`).
- Keep each file focused; cross-link with relative paths instead of duplicating text.
- Small clarification-only change: edit the owning file in place. New behavior: new folder.

## NovaNN constraints primer

Every spec must respect `AGENTS.md`. Record only applicable constraints in `constraints.md`:

- Language split: C23 for portable core (tensor, storage, device, dtype, dispatch, repr, CPU backends); C++23 for autograd and CUDA/HIP kernels; Rust (`ncore_memory`) owns all buffer lifecycle via `reserve`/`retain`/`release`/`resize`; Cython validates args and maps `novaStatus_t` to Python exceptions; Python is the user API.
- Backends: CPU active, CUDA active, HIP Linux-only. CUDA and HIP are mutually exclusive at configure time. State target backend explicitly.
- Status handling: every native entry point reports through `novaStatus_t`.
- Generated code: files bannered `DO NOT EDIT — GENERATED CODE` (e.g. `ncore/native/cpu/dtype/DTypeCasting.c`) come from `tools/codegen/` via `uv run tools/codegen/generate.py gen --all --keep-going --run-formatters`. Specs must not require hand-editing them.
- Build verification: presets in `CMakePresets.json` (`<backend>-<config>[-<sanitizer>][-test][-os]`) via `scripts/build-presets.sh`, `scripts/compile-presets.sh`, `scripts/run-tests.sh`. Specs must name the presets that validate the feature.
- Verify, never assume: compiler and linkage facts come from the build tree (`CMakeCache.txt`, `build.ninja`, `ldd`), not memory.

## Reading order

1. `references/spec-template.md`: copy the section skeletons verbatim, then fill them.
2. `references/checklist.md`: run the quality gates before reporting done.
3. `references/example.md`: one worked minimal spec for tone and size reference.
