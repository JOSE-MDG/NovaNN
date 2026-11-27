# Contribution Guide

Thank you for your interest in contributing to **NovaNN**.

NovaNN is moving to v5. v4 was a Python and NumPy framework built to learn how
deep learning frameworks work internally. That educational goal remains, but it
is no longer the only one. v5 keeps the Python API and replaces NumPy with a
native core under `ncore/`, written in C23, C++23 and Rust, and extends the
scope from training models to the whole life of a model: pretraining,
fine-tuning, quantization, adapters, inference and deployment on CPU, CUDA and
HIP. The Python package under `nova/` is still the stable v4.0.4 release and
remains published. Changes there are legacy maintenance, not the direction of
the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Environment Setup](#development-environment-setup)
- [Working with Agents](#working-with-agents)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Commits](#commits)
- [Testing](#testing)
- [CI](#ci)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)
- [Possible Questions](#possible-questions)

## Code of Conduct

This project aims to be a welcoming, collaborative and professional space.
It is expected:

- **Mutual respect**: treat everyone with courtesy and professionalism
- **Constructive criticism**: focus on the code, not on the people
- **Collaboration**: help others learn and grow
- **Patience**: give clear and actionable feedback

## How Can I Contribute?

### Priority Areas

Current work centers on the native core and the tooling around it:

- Native core
- CPU and GPU backends
- Memory management
- Code generation and build tooling
- Tests and documentation

Areas the ecosystem is moving toward, not all implemented yet:

- Transformer architectures and dataset handling
- Quantization workflows such as NF4 and FP8
- Efficient fine-tuning with adapters such as LoRA and QLoRA
- SFT and DPO trainers
- Inference, deployment and a model hub

The order and status of these areas are tracked in `v5-notes.md`.

### General Process

1. Fork the repository on GitHub.
2. Clone your fork with submodules.
3. Create a branch for your change: `git checkout -b feat/new-feature`.
   Use `feat/`, `fix/`, `infra/` or `ci/` so the purpose is clear.
4. Make the change following the standards below.
5. Commit with a Conventional Commit message.
6. Push to your fork: `git push origin feat/new-feature`.
7. Open a pull request against `dev`.
8. Once CI is green and the change is approved, open a pull request from `dev`
   to `main`, then the corresponding pull request to the upstream repository.

Opening a pull request directly against `main` is rejected by the
`branch-origin` check. Every change goes through `dev`.

## Development Environment Setup

### Prerequisites

- CMake >= 3.27 and Ninja
- Linux: GCC >= 15. The pinned `ubuntu-26.04` CI image already provides it.
- Windows: clang-cl and lld-link >= 20.1. MSVC `cl.exe` is not a project
  compiler, except as nvcc's host compiler on Windows.
- Rust toolchain for `ncore/memory`
- Python >= 3.12 with `uv`. `.python-version` pins 3.14.4.
- Git with submodule support. `thirdParty/vcpkg` is pinned by commit.
- Optional: CUDA or HIP, which are mutually exclusive at configure time
  (`-DUSE_CUDA=ON` or `-DUSE_HIP=ON`). HIP is Linux only.

### Install

```bash
git clone --recurse-submodules git@github.com:JOSE-MDG/NovaNN.git
cd NovaNN
```

A plain clone also works: `scripts/lib/vcpkg-env.sh` (or its PowerShell
counterpart) initializes and bootstraps the pinned vcpkg submodule on demand.

### Build and Test

```bash
scripts/build-presets.sh cpu-test-release-linux
scripts/compile-presets.sh -j $(nproc) cpu-test-release-linux
scripts/run-tests.sh cpu-test-release-linux
```

On Windows use the PowerShell scripts with a `*-windows` preset:

```powershell
.\scripts\build-presets.ps1 cpu-test-release-windows
.\scripts\compile-presets.ps1 -j $env:NUMBER_OF_PROCESSORS cpu-test-release-windows
.\scripts\run-tests.ps1 --force-stop-on-failure cpu-test-release-windows
```

`cmake --list-presets` lists every available preset. `cmake --workflow --preset <preset>` chains configure, build and test in one command. Preset names follow
`<backend>-<config>[-<sanitizer>][-test][-os]`, for example
`cpu-asan-test-debug-linux` or `hip-ubsan-test-release-linux`.

## Working with Agents

v5 is developed with help from AI agents. Humans and agents follow the same
rules, documented in `AGENTS.md`. The repo provides reusable skills under
`.agents/skills/`, with agent definitions mirrored in `.github/agents/`,
`.opencode/agents/` and `.claude/agents/`. Project agents include
`commit-creator`, `implementation-review`, `test-creator`, `test-integrity` and
`build-integrity`.

Work is spec-driven: a feature spec under `specs/` is written before
implementation. Generated files are regenerated from templates instead of
edited by hand. Agents report findings with `file:line` references, keep their
responses concise and write them in English.

Agent changes go through the same gate as human changes: draft, review, test,
then a pull request into `dev`. An agent never pushes, rebases, resolves
conflicts or applies a commit without explicit approval.

## Project Structure

```
NovaNN/
├── ncore/
│   ├── include/           # Public C and header-only C++ headers
│   ├── src/               # C and C++ implementation
│   ├── native/            # CPU, CUDA, HIP and kernel backends
│   ├── memory/            # Rust allocator (ncore_memory)
│   └── tests/             # GTest suites registered with CTest
├── tools/codegen/         # Templates, rules and generator entry point
├── scripts/               # Build, test and CI helper scripts
├── .github/workflows/     # GitHub Actions pipelines
├── nova/                  # Legacy v4.0.4 Python package
├── tests/                 # Legacy pytest suite
├── examples/              # Example scripts
└── benchmarks/            # Performance benchmarks
```

### Where Does Each Thing Go?

- **New core type or operation**: `ncore/include/` and `ncore/src/`
- **New backend kernel**: `ncore/native/` with its GTest case in `ncore/tests/`
- **New memory primitive**: `ncore/memory/`
- **New generated table**: a template and rule under `tools/codegen/`
- **New GTest case**: `ncore/tests/`, mirroring the module it covers
- **Legacy Python change**: `nova/` and `tests/`

## Code Standards

### C and C++

- Write C23 and C++23. Doxygen comments use one-line briefs; add details only
  when the reason is not obvious from the signature.
- Use designated initializers for structs and tables, one member per line.
- Headers shared with C++ cast through `STATIC_CAST` from
  `ncore/include/ncore/headeronly/macros.h`.
- Header suffixes: `.h` for C, `.hh` for header-only C++, `.hpp` for test-tree
  utilities.
- Comments state rationale, never narration. Do not use em-dash separators in
  comments or documentation.
- Never edit generated files directly. Edit the template and the JSON rules,
  then regenerate:

  ```bash
  uv run tools/codegen/generate.py gen --all --keep-going --run-formatters
  ```

### Python

- Use Google-style docstrings and keep the code formatted with the configured
  `ruff` setup.
- Run Python tooling through `uv`, never through bare `pip` or `poetry`.

Commit messages follow the rules in [Commits](#commits), which are shared by
humans and agents.

## Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Use a type that describes the nature of the change. This guide does not fix a
list of types; follow the convention currently used in the repository. The
scope is optional, a single lowercase word, and never a phrase. The subject has
a maximum of 72 characters and uses imperative mood ("add", not "added"). Body
lines have a maximum of 80 characters and never repeat the subject.

Choose the style by the size of the change:

- **Summarized**: one line, for trivial changes.
- **Normal**: subject plus a 2 to 4 line body, for moderate changes.
- **Dense**: subject, a 3 to 6 line body and bullets, for multi-file changes,
  features with several parts or fixes with several root causes. Bullets start
  with an action verb: `Add`, `Remove`, `Update`, `Replace`, `Implement`,
  `Extract`, `Rename`, `Improve`, `Standardize`, `Refactor`, `Optimize`,
  `Guard`, `Wrap`, `Split` or `Align`. One change per bullet.

Group one commit per file by default. Group files only when their changes are
strictly identical: mirrored CUDA and HIP pairs, a declaration with its
definition, or a template with its regenerated output. New files that belong to
one feature go together. When unsure, split.

Draft every pending commit in `commits.md` at the repository root. Each entry
lists the file paths, a blank line, the word `command:`, a blank line, and a
fenced text block with the `git add` and `git commit` commands. Drafting never
runs git writes.

Apply commits only on explicit approval. Execute the commands in file order,
one commit per command, then report the SHA and subject of each commit and
confirm with `git log` and a clean `git status` for tracked files. Stop on the
first failure; never skip, reorder or fold commits to work around it.

Merge commits are created only on explicit request. Verify the real range with
`git log` between the branch tips and describe exactly that range. A merge only
affects the branch you are on; the source branch pointer does not move.

Never push, rebase, resolve conflicts, use `git add -A` or edit generated files
directly. Stage exactly the listed files and leave untracked content alone.

## Testing

### Native Tests

Native tests are GTest suites run by CTest through a `*-test-*` preset. A failing
test or no tests at all fails the run; presets set `noTestsAction` to `error`.
Sanitizer presets get their environment from `suppr.txt` through the test
scripts. Do not widen or remove a `GTEST_SKIP` without a reason tied to the
specification.

```bash
scripts/run-tests.sh cpu-test-release-linux -- -V
```

### Legacy Python Tests

```bash
uv run --frozen pytest -q tests
```

Scope collection to `tests`. Running pytest from the repository root also
collects the vcpkg submodule tests, which are not part of NovaNN.

### Codegen Check

CI regenerates the outputs and runs `git diff --exit-code`. A pull request that
changes a template or rule must include the regenerated files.

## CI

| Workflow | Trigger | Required check |
| --- | --- | --- |
| `.github/workflows/ci.yml` | PR into `dev` or `main`, push to both branches, manual | `ci` |
| `.github/workflows/branch-policy.yml` | PR into `main` | `branch-origin` |
| `.github/workflows/gpu-sim.yml` | PR and push to `dev`/`main` for relevant paths, weekly, manual | no |
| `.github/workflows/nightly.yml` | weekly, manual | no |

The `ci` check is an aggregate job over CPU Linux, ASan, UBSan, Windows, Python
and codegen. The same gate applies to pull requests into `dev` and into `main`.
Every job that configures a preset installs uv first, because the CMake runtime
module lists codegen outputs through `uv run` during configure. The GPU
simulator workflow is informational: hosted runners have no GPU and a simulator
cannot cover timing or hardware-specific behavior.

The CPU jobs pin `ubuntu-26.04` because it provides GCC 15, CMake 4.5,
clang-format 20.1.8, Rust and Python 3.14.4 without extra installs. The GPU
simulator stays on `ubuntu-24.04` because that is what its setup action
supports. Windows pins `windows-2025`.

## Pull Request Process

### Before Opening the Pull Request

- [ ] The native presets affected by the change pass locally
- [ ] `uv run --frozen pytest -q tests` passes when Python code changed
- [ ] Generated files are in sync with their templates
- [ ] Documentation matches the behavior
- [ ] Commits follow the rules in [Commits](#commits)
- [ ] The pull request targets `dev`

### PR Template

```markdown
## Description

[Clear description of the change]

## Type of Change

[Describe the nature of the change]

## Checklist

- [ ] Tests added or updated
- [ ] Documentation updated
- [ ] Generated files regenerated when needed
- [ ] CI passes locally

## Testing

[Presets and commands used]

## Additional Notes

[Any relevant context]
```

### Review Process

1. A maintainer reviews the pull request into `dev`.
2. Changes or clarifications may be requested.
3. Once `ci` is green and the change is approved, it is merged into `dev`.
4. A pull request from `dev` to `main` runs the same gate plus `branch-origin`.
5. After approval and green checks, it is merged into `main`.

## Reporting Bugs

### Before Reporting

- Search [existing issues](https://github.com/JOSE-MDG/NovaNN/issues) to avoid duplicates.
- Make sure it is a NovaNN bug, not an environment problem.

### Issue Template

````markdown
**Bug Description**
[Clear and concise description]

**Steps to Reproduce**

1. Commit or version used
2. Command or preset executed
3. Error observed

**Expected Behavior**
[What you expected to happen]

**Current Behavior**
[What actually happened]

**Environment**

- NovaNN commit: [SHA]
- OS and version: [for example Ubuntu 26.04]
- Compiler: [for example GCC 15.2]
- CMake and Ninja versions
- Backend and device: [CPU, CUDA or HIP]

**Logs**
[Paste the relevant output]
````

For a legacy Python issue, add the Python version and the NumPy version.

## Proposing Features

```markdown
**Does the feature solve a problem?**
[Describe the problem you face]

**Describe Proposed Solution**
[How you would like it to work]

**Alternatives Considered**
[Other solutions you considered]

**Suggested Implementation**
[Files or modules you expect to change]

**Additional Context**
[Papers, references or examples from other frameworks]
```

## Possible Questions

### Can I contribute as a beginner?

Yes. Issues labeled `good first issue` are a good starting point.

### In which language should I write code and documentation?

Code identifiers and comments are in English. Documentation is maintained in
Spanish and English. Issues and pull requests can be in either language.

### Can I open a pull request directly against `main`?

No. The `branch-origin` check rejects it. Open the pull request against `dev`
first.

### Do I need tests for a documentation-only change?

No. New code needs tests.

### How are commits created when an agent does the work?

The agent drafts them in `commits.md` and waits. Nothing is staged or applied
until you explicitly approve. See [Commits](#commits).

### How long does the review take?

Usually one to three days. If it takes longer, feel free to ping the pull request.

## Contact

- **GitHub Issues**: for bugs and features
- **Email**: josepemlengineer@gmail.com
- **Discussions**: for general questions and discussions

---

Thank you for contributing to NovaNN.
