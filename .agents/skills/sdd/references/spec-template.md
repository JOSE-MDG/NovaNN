# Spec Templates

`spec.md` is always required. Every other file is chosen per implementation using the selection rules below. Start from the default set and adjust; copy each chosen skeleton verbatim, then fill it.

## Selection rules

Default set for a standard feature: `spec.md`, `requirements.md`, `constraints.md`, `acceptance.md`.

- Omit `requirements.md` only when there are no testable behaviors to number (pure documentation or question-only spike). Fold the one or two behaviors into `acceptance.md` instead.
- Omit `constraints.md` only when no backend, dtype, memory, threading, or build constraint applies. State why in `spec.md` Non-goals.
- Omit `acceptance.md` never: every spec needs at least one verifiable criterion, even if it is a single build or review check.
- Add `api.md` when the public surface exceeds one function or needs signatures, naming, and ownership spelled out.
- Add `data-model.md` when shapes, strides, layouts, or storage lifetime need their own diagrams or tables.
- Add `errors.md` when status codes, edge cases, or failure modes are large enough to crowd `requirements.md`.
- Split any file that exceeds roughly 200 lines or mixes two audiences into two focused files. Cross-link instead of duplicating.

## spec.md

```markdown
# Feature: <kebab-name>

## Context
<One paragraph: problem, who needs it, why now. Link v5.0.0 target vs current state per AGENTS.md.>

## Goals
- <Bulleted user-visible outcomes>

## Non-goals
- <Explicitly out of scope>

## User stories
- US-1: As a <role>, I want <capability> so that <benefit>.
- US-2: ...

## Interfaces (what, not how)
- <Public API shape, dtypes, devices, error surface via novaStatus_t>
- <Out of scope internals stay out>

## Open questions
- <Q1, owner if known>
```

## requirements.md

```markdown
# Requirements: <kebab-name>

Each requirement uses SHALL, has a stable ID, and maps to acceptance criteria in acceptance.md.

## Functional
- REQ-001: The implementation SHALL <behavior>. Maps to AC-001.
- REQ-002: ...

## Behavior and edge cases
- REQ-010: On <invalid input / OOM / unsupported dtype-device combo>, the API SHALL return <novaStatus_t code path> instead of aborting.
- REQ-011: ...

## Compatibility
- REQ-020: The feature SHALL build under <presets, e.g. cpu-release-linux, cuda-test-debug-linux>.
- REQ-021: ...
```

Rules: one behavior per requirement; no implementation detail (no file names, no kernel launch values); every REQ has at least one AC.

## constraints.md

```markdown
# Constraints: <kebab-name>

## Language and ownership
- Core logic in C23 under <ncore/src/... path if known>; GPU kernels in C++23 (CUDA/HIP); all buffers via Rust ncore_memory reserve/retain/release/resize.

## Backend and device
- Target backend: <cpu | cuda | hip | cpu+cuda...>. Note CUDA/HIP mutual exclusivity and HIP Linux-only limit.

## Dtypes and dispatch
- Supported dtypes: <list>. Unsupported combos return <status> via dispatch.

## Threading and memory
- Worker pool impact: <compute/autograd/dtloader or none>. Allocation path: <CPU std::alloc via Rust | GPU via memorycsrc bridge>.

## Generated code and editing
- Touched generated files (if any): <none expected | path + template + JSON rules under tools/codegen/>.
- Header suffixes: C `.h`, header-only C++ `.hh`, test utilities `.hpp`.

## Build and test
- Validate with: `scripts/build-presets.sh <preset>`, `scripts/compile-presets.sh <preset>`, `scripts/run-tests.sh <preset>`.
```

## acceptance.md

```markdown
# Acceptance: <kebab-name>

- AC-001: Given <setup>, when <action>, then <observable result>. Verifies REQ-001.
- AC-002: ...
- AC-EDGE-001: Given <edge input>, when <action>, then <status or message>. Verifies REQ-010.
```

Rules: each AC names its REQ IDs when `requirements.md` exists, otherwise each AC stands alone as the requirement. Each AC is checkable without reading implementation (build log, test name, repr output, status code).

## Optional modules (add only per selection rules)

### api.md

```markdown
# API: <kebab-name>

- Function: `<signature>`; inputs <types/devices>; output <type/device>; errors via <novaStatus_t path>.
- Ownership: caller owns <what>; callee owns <what>; no hidden allocation.
```

### data-model.md

```markdown
# Data model: <kebab-name>

- Shapes and strides: <rank, layout, contiguity assumptions>.
- Storage lifetime: <who reserves/retains/releases via Rust ncore_memory>.
```

### errors.md

```markdown
# Errors: <kebab-name>

- ERR-001: On <condition>, return <status> with <observable side effect>. Verifies REQ-XXX or stands alone.
```
