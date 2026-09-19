# Spec Quality Checklist

Run before reporting done. Fix failures in the owning file.

## Structure

- [ ] Folder is `specs/NNN-kebab-name/` with the next free NNN.
- [ ] `spec.md` always exists; every other file was chosen per the selection rules in `references/spec-template.md` (omissions justified, splits cross-linked).
- [ ] No code, no task list, no plan. Spec-only.

## Clarity

- [ ] `spec.md` states what and why with no launch sizes, block sizes, or file-level design.
- [ ] Each user story names a role, capability, and benefit.
- [ ] Non-goals list at least one plausible near-miss that is out of scope.

## Requirements (if requirements.md present)

- [ ] Every requirement uses SHALL and has a stable ID (`REQ-NNN`).
- [ ] One behavior per requirement; no conjunction hiding two behaviors.
- [ ] Error paths specify `novaStatus_t` behavior, not abort or silent truncation.

## Constraints (if constraints.md present)

- [ ] Target backend stated; CUDA/HIP exclusivity and HIP Linux-only noted when relevant.
- [ ] Language split stated (C23 core, C++23 kernels/autograd, Rust memory, Cython status mapping).
- [ ] Generated-code impact stated; no requirement to hand-edit a generated file.
- [ ] Validating presets named explicitly.

## Acceptance

- [ ] Every REQ in `requirements.md` (if present) maps to at least one AC; every AC maps back to a REQ or stands alone with justification when `requirements.md` is omitted.
- [ ] Each AC is observable (test name, log, status code, repr output).
- [ ] Edge cases (unsupported dtype, OOM, shape mismatch) have AC coverage or an explicit why-not.

## Common failures

- Inventing requirements the user never confirmed: move to Open questions.
- Tech detail leaking into `spec.md`: move to `constraints.md` or delete.
- Unverifiable adjectives (fast, clean, robust): replace with measurable AC or delete.
- Telling the implementer to cite the spec in code (paths, REQ/AC IDs, spec quotes): forbidden, `specs/` is local-only and unversioned, so every such reference dangles on the next clone. Restate rationale without citation.
