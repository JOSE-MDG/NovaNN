# Example: specs/001-cpu-matmul-float32/

Minimal tone and size reference. Fictional; check current code before reusing. This shows the default four-file shape; other implementations choose a different file set per the selection rules.

## spec.md

```markdown
# Feature: cpu-matmul-float32

## Context
NovaNN v5.0.0 targets a native matmul; no arithmetic op exists yet in any backend. A CPU float32 baseline unblocks API shape and testing before GPU ports.

## Goals
- Define observable behavior for CPU float32 matmul on 2D tensors.

## Non-goals
- GPU kernels, autograd rules, mixed dtypes.

## User stories
- US-1: As a model author, I want `nova.matmul(a, b)` on CPU float32 so that I can run small forward passes in tests.

## Interfaces (what, not how)
- Inputs: two 2D float32 CPU tensors with inner dims equal; output: 2D float32 CPU tensor.
- Shape mismatch and unsupported dtype-device combos report via novaStatus_t.

## Open questions
- Row-major only for v1 or accept strided inputs?
```

## requirements.md

```markdown
# Requirements: cpu-matmul-float32

## Functional
- REQ-001: The API SHALL compute C = A @ B for 2D float32 CPU tensors. Maps to AC-001.
- REQ-002: The API SHALL reject inner-dim mismatch with an error status. Maps to AC-002.

## Behavior and edge cases
- REQ-010: On zero-size dims, the API SHALL return an empty output with success status. Maps to AC-EDGE-001.

## Compatibility
- REQ-020: The feature SHALL build under cpu-release-linux and cpu-test-debug-linux. Maps to AC-003.
```

## constraints.md

```markdown
# Constraints: cpu-matmul-float32

## Language and ownership
- Core logic in C23 under ncore/native/cpu; buffers via Rust ncore_memory.

## Backend and device
- Target backend: cpu only.

## Dtypes and dispatch
- Supported dtypes: float32 in, float32 out. Other dtypes return unsupported status via dispatch.

## Threading and memory
- Worker pool impact: compute pool; no autograd registration in v1.

## Generated code and editing
- Touched generated files: none expected.

## Build and test
- Validate with: `scripts/build-presets.sh cpu-test-debug-linux`, `scripts/compile-presets.sh cpu-test-debug-linux`, `scripts/run-tests.sh cpu-test-debug-linux`.
```

## acceptance.md

```markdown
# Acceptance: cpu-matmul-float32

- AC-001: Given 2x3 and 3x2 float32 inputs, when matmul runs, then the 2x2 output matches the reference values within 1e-5. Verifies REQ-001.
- AC-002: Given inner-dim mismatch, when matmul runs, then an error status is returned and no output buffer is leaked. Verifies REQ-002.
- AC-EDGE-001: Given a zero-size dim, when matmul runs, then an empty output with success status is returned. Verifies REQ-010.
- AC-003: Given cpu-test-debug-linux preset, when configure/build/test run, then all steps succeed. Verifies REQ-020.
```
