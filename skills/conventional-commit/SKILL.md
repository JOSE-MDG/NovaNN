---
name: conventional-commit
description: 'Prompt and workflow for generating conventional commit messages. Supports three styles: summarized (one-liner), normal (short body), and dense (body + bullets).'
---

### Instructions

This file provides instructions for writing conventional commit messages in three styles. Choose the style based on the complexity of the change.

### Workflow

**Follow these steps:**

1. Run `git status` to review changed files.
2. Run `git diff` or `git diff --cached` to inspect changes.
3. Run `git log --oneline` (or `git log | head -n 500` for more details) to inspect the full commit history style.
4. Stage your changes with `git add <file>` (must be confirmed).
5. Choose a commit style based on change complexity.
6. Write the message and commit.

### Allowed Types

| Type | Use for |
|------|---------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, whitespace, no logic change |
| `refactor` | Code restructuring, no feature/fix |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `build` | Build system or dependencies |
| `ci` | CI/CD configuration |
| `chore` | Maintenance tasks |
| `revert` | Reverting a previous commit |

### Scope Rules

- Scope is **optional** but recommended.
- Must be a **single lowercase word** (e.g., `cast`, `device`, `cmake`, `repr`, `simd`).
- No multi-word scopes. Use abbreviations if needed.

---

## Style 1: Summarized

Use for small, straightforward changes where the subject line says it all.

**Format:**
```
<type>(<scope>): <brief description>
```

**When to use:**
- Renaming, formatting, typo fixes
- Single-line changes
- Obvious changes that need no explanation

**Examples:**
```
docs(repr): document derived state engine
style: add newline between functions
fix(cast): update include paths to match restructured header layout
chore: remove unnecessary scripts section
refactor(api): rename include 'dense_layout.h' to unified 'layouts.h'
```

---

## Style 2: Normal

Use for moderate changes that benefit from a short explanation.

**Format:**
```
<type>(<scope>): <brief description>

<concise description in 2-4 lines>
```

**When to use:**
- Feature additions with non-obvious design decisions
- Refactors that change structure significantly
- Fixes that need context to understand

**Examples:**
```
feat(metadata): add GPU backend awareness and stabilize buffer sizing

This change adds explicit support for labeling GPU-resident tensors in
normal display mode and hardens the internal numeric buffers to prevent
potential overflows on extremely large dimension sizes.
```

```
refactor(options): rename ReprMode enums and upgrade documentation

The display verbosity enums have been renamed to follow the naming
standards, improving code consistency across the FFI boundaries.
The documentation has been significantly expanded to serve as a technical guide.
```

---

## Style 3: Dense

Use for complex changes that need both context and a detailed breakdown.

**Format:**
```
<type>(<scope>): <brief description>

<concise description in 3-6 lines>

- <action> ...
- <action> ...
- <action> ...
```

**When to use:**
- Multi-file changes
- Features with several sub-components
- Refactors with many distinct changes
- Fixes involving multiple root causes

**Action verbs for bullets:**
`Add`, `Remove`, `Update`, `Replace`, `Implement`, `Extract`, `Rename`, `Improve`, `Standardize`, `Refactor`, `Optimize`, `Guard`, `Wrap`, `Split`, `Align`

**Examples:**
```
feat(cmake): add sanitizer detection module

AddressSanitizer and UndefinedBehaviorSanitizer configuration was
not modularized into its own detection module

- Add USE_ASAN option (default OFF) for AddressSanitizer
- Add USE_UBSAN option (default OFF) for UndefinedBehaviorSanitizer
- Add detection logic for compiler support of both sanitizers
- Add status output to build summary
```

```
fix(detect): correct InitOnceExecuteOnce callback signature for Windows

The init_device_flags_lock() callback passed to InitOnceExecuteOnce()
had an incorrect signature (void(void)) instead of the required
PINIT_ONCE_FN (BOOL CALLBACK(PINIT_ONCE, PVOID, PVOID*)). This
caused undefined behaviour on Windows builds with mingw64/ucrt64.

- Split init_device_flags_lock() into platform-specific implementations
  behind #ifdef __linux__ / #elif defined(_WIN64) guards
- Linux: keep existing void(void) signature with mtx_init()
- Windows: use BOOL CALLBACK signature with InitializeCriticalSection()
  and return TRUE
```

```
refactor(cmake): extract compiler flags into NovaNNBuildFlags.cmake

Warning flags, debug/release flags, and compile options were
duplicated across multiple CMakeLists.txt files. This made it
difficult to maintain consistent flags across all targets.

- Create cmake/Modules/NovaNNBuildFlags.cmake with centralized
  NOVA_WARNING_FLAGS, NOVA_CXX_FLAGS, NOVA_DEBUG_FLAGS, and
  NOVA_RELEASE_FLAGS
- Add nova_configure_build_flags(TARGET) function that applies
  all flags via target_compile_options with generator expressions
- Add nova_configure_linker(TARGET) function for linker hardening
  flags (-Wl,-z,relro,-z,now -Wl,--as-needed -Wl,--no-undefined)
- Add USE_LTO option (default ON) with LTO detection
- Add USE_ASAN and USE_UBSAN sanitizer options
- Remove inline COMPILE_FLAGS, DEBUG_FLAGS, RELEASE_FLAGS from
  root CMakeLists.txt
```

---

## Validation

- **Type:** Must be one of the allowed types listed above.
- **Scope:** Optional, single lowercase word.
- **Description:** Required. Use imperative mood (e.g., "add", not "added").
- **Body:** Use for Style 2 and Style 3. Never repeat the subject line.
- **Bullets:** Use only in Style 3. Start with an action verb. One change per bullet.
- **Line length:** Subject line max 72 characters. Body lines max 80 characters.
