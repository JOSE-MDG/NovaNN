---
name: doxygen-c-cxx-documentation
description: 'Generate and manage modern Doxygen-style documentation for C and C++ source code in the NovaNN voice: one-line briefs, rationale-only details, no over-explanation. Includes the full special-commands reference.'
---

# Doxygen C/C++ Documentation: the NovaNN Voice

> **Scope**: This document assumes you are already familiar with Doxygen's toolchain and comment parsing. Section 1 is the project's voice: the rules that decide what a doc block says and what it deletes. Sections 2 and 3 cover mechanics and patterns with examples from this codebase. Section 5 is the full command reference, kept for lookup. All special commands use the `@` prefix.

---

## Table of Contents

1. [The Voice](#1-the-voice)
2. [Comment Styles and Attachment](#2-comment-styles-and-attachment)
3. [Patterns by Entity](#3-patterns-by-entity)
4. [Advanced Patterns](#4-advanced-patterns)
5. [Special Commands Reference](#5-special-commands-reference)

---

## 1. The Voice

A doc block earns its place by saying what the signature cannot. The
`@brief` states the job in one line. Everything else states a **why** a
maintainer cannot rediscover from the code: an ordering hazard, an
invariant, an ownership rule, a hardware constraint, a rejected
alternative.

The test for any block: *would a human reviewer miss it if it were
deleted?* If not, delete it. Prefer deleting a whole redundant block
over trimming it word by word.

### 1.1 Write rules

1. **`@brief`: one sentence, the job.** Not the implementation walk.
   Bad: `@brief Initializes the stream through the handle API.` on a
   constructor whose body is one init call. Good: `@brief Initializes
   the stream.`
2. **`@details`: rationale only.** Non-obvious whys: why the obvious
   alternative is wrong, what breaks if the order changes, what the
   caller must keep alive. If the block restates the signature, the
   params, or the class doc, it goes.
3. **Short sentences.** No semicolon-chained clauses. No em-dash or
   double-hyphen separators: use a colon, parentheses, or a new
   sentence.
4. **Banned in comments**: the word "production" (say "the
   implementation" or "the kernels"); narration of versions, timings,
   or migration history ("Registered in W0 of the rewrite"); debug
   commentary of any kind.
5. **`@par` scaffolding** (Ownership / Thread safety / Lifecycle
   stacks): only when the constraint is real and easy to miss. One
   prose sentence usually beats a titled section.
6. **`@li` lists**: for genuinely enumerable contracts (error codes,
   valid ranges, step sequences the caller must follow). Never for
   restating parameters or the class doc.
7. **Redundancy across commands**: if a `@return` would restate the
   `@details` error list verbatim, keep one of the two.
8. **Sentence openings**: vary them or drop them. A file where every
   block starts "This function..." reads machine-made.
9. **Markdown is forbidden inside blocks** (see 2.5): use `@c`, `@p`,
   `@ref`, `@li`.
10. **ReflowComments: Never** in `.clang-format`: only rewrite the
    lines your content edit touches; never rewrap a block you did not
    otherwise change.

### 1.2 Keep list (never trim these)

Documentation a maintainer cannot rediscover is the payload. When
sweeping or writing, these survive every cleanup:

- Null-path and kernel-drop rationale, stream ordering hazards.
- Parity contracts (host must equal device bitwise, and where that is
  pinned by tests).
- ABI notes, struct layout constraints, alignment requirements.
- Ownership and lifetime ("caller owns", what must outlive what).
- Units and scaling factors.
- Hardware constraints (ISA limits, driver behavior, platform gaps).
- Why-comments for non-obvious choices (why not the obvious design).

### 1.3 Real before/after from this codebase

Enum trailing comments (`ncore/include/ncore/core/device.h`):

```
before:  Meta,    ///< No-op backend; used for shape inference only.
after:   Meta,    ///< No-op backend; shape only, never allocates or launches.
```

The "after" adds what the name cannot say (it never launches) instead
of paraphrasing "shape inference".

Class doc with a load-bearing detail (`ncore/tests/core/utils/Rng.hpp`):

```cpp
/**
 * @class Generator
 * @brief One reproducible fixture stream.
 *
 * A failed init is reported by the next call, not by the constructor.
 * Copies fork the stream and each copy advances on its own. If a
 * device drain fails the fill already landed: regard the handle as
 * spent.
 */
```

Three sentences, three contracts, zero scaffolding.

Private helper with a hazard rationale (same file):

```cpp
  /**
   * @brief Waits for the queued device fills.
   *
   * The null path drops a freshly compiled kernel when the next
   * null-stream op waits host side behind it, so a one byte readback
   * copy shares that queue and orders it out before the caller
   * continues.
   */
```

This block survives any cleanup: the workaround looks gratuitous until
this paragraph explains the drop.

### 1.4 Common removals (sweep findings)

| Pattern | Action |
|---|---|
| `@par Ownership` / `@par Thread safety` stacks restating the obvious | Delete; keep one prose line if the constraint is real |
| `@li` bullets restating each parameter | Delete |
| `@return The underlying status, unmodified.` on every method of a status API | Delete; the status is the contract |
| "Note that...", "It is worth noting..." | Drop the opener, keep the fact if it matters |
| Version/history narration in source | Delete |
| Stale claims contradicting the code | Delete (they rot and mislead) |
| The same caveat repeated at every call site | Document once at the choke point |

---

## 2. Comment Styles and Attachment

Choosing the right comment style is not cosmetic: it determines what
Doxygen parses, where the block attaches, and how readable the source
stays.

### 2.1 Block Comment Styles

| Syntax | Name | Use When |
|---|---|---|
| `/** ... */` | Javadoc block | Standard choice for all entity docs preceding the declaration |
| `/*! ... */` | Qt/Doxygen block | Identical to `/**`; preferred in some Qt codebases |
| `/** @brief ... */` on one line | Compact block | Short single-concept docs |

Both `/**` and `/*!` are semantically equivalent. Pick one and be
consistent across the entire project. NovaNN uses `/**`.

```cpp
/**
 * One-line brief, then details only when a why exists.
 */
```

### 2.2 Single-Line Comment Styles

| Syntax | Name | Use When |
|---|---|---|
| `///` | C++ triple-slash | Preceding a declaration on its own line |
| `//!` | Doxygen exclamation | Preceding a declaration; alternative to `///` |

```cpp
/// Brief description of the following entity.
/// Consecutive lines merge into one block.
int myFunction();
```

### 2.3 Trailing (Inline / Member) Comment Styles

The `<` modifier attaches the comment to the *preceding* entity.

| Syntax | Name | Use When |
|---|---|---|
| `///<` | Trailing C++ line | Struct/class members, enum values, inline |
| `//!<` | Trailing exclamation line | Alternative to `///<` |
| `/**< ... */` | Trailing Javadoc block | Multi-sentence trailing comment (sparingly) |

```cpp
enum class Status {
    OK      = 0, ///< Operation completed successfully.
    TIMEOUT = 1  ///< Deadline passed before completion.
};
```

A trailing comment adds what the member name cannot say. `int x;
///< X coordinate.` is noise; `int x; ///< X in world units, not
pixels.` is documentation.

### 2.4 The Exact Rules for Comment Attachment

- A `/** */`, `/*!`, or `///` placed **directly before** a declaration
  attaches to that entity.
- A `///<`, `//!<`, `/**< */`, or `/*!< */` placed **on the same line
  as** or **immediately after** a declaration attaches to that entity.
- There must be **no blank lines** between the comment and the entity
  it documents for the attachment to hold.
- For **files**, the doc block must contain an explicit `@file`
  command OR must be the first doc block in the file (depending on
  config).

### 2.5 Markdown Is Forbidden in Doxygen Comments

Doxygen comments use Doxygen's native markup. Markdown syntax
(`**bold**`, backticks, `-`/`1.` lists, `[text](url)`, `#` headers,
`---`) must never appear inside a doc block: it renders inconsistently
and defeats the native commands.

| Markdown (forbidden) | Doxygen command (use instead) |
|---|---|
| `` `code` `` (backticks) | `@c code` (inline code font) |
| `` `param` `` in prose | `@p param` (parameter reference, code font) |
| `` `symbol` `` (types, functions, macros, files) | `@ref symbol` (clickable cross-reference) |
| `**bold**` | `@b word` |
| `*italic*` | `@e word` (or `@a word` for parameter names) |
| `- item` / `* item` (bullets) | `@li item` |
| `1. item` (numbered lists) | `@li 1. item` (keep the number manually) |
| `# Header` / `## Header` (file docs, pages) | `@section <id> Header` |
| `**Header**:` (titled paragraph in an entity doc) | `@par Header` |
| `[text](url)` (external links) | Bare URL (Doxygen auto-links it) |
| `[text](@ref sym)` (internal links) | `@ref sym` directly |
| `> quote` | `@par` / `@note` / `@attention` / `@warning` |
| `---` (horizontal rule) | A blank line starting a new paragraph |
| ``` ``` ``` (code fences) | `@code{.lang}` / `@endcode` |

The swept public headers under `ncore/include/ncore/` are the project's
reference implementation of these rules.

### 2.6 Style Decision Matrix

```
Documenting a function, class, struct, typedef, namespace → /** ... */
Documenting a struct/class member inline                   → ///<
Documenting an enum value inline                           → ///<
Short one-liner before a forward declaration               → ///
Multi-sentence trailing comment on a member                → /**< ... */ (sparingly)
File-level documentation block                             → /** @file ... */
```

---

## 3. Patterns by Entity

All examples below are drawn from or modeled on this codebase. They
follow the voice rules of section 1.

### 3.1 File Header

Every `.h` / `.hpp` / `.c` / `.cpp` file that exposes documented
entities has a file doc block. Without it, globals, free functions, and
enums are silently omitted unless `EXTRACT_ALL=YES`.

NovaNN file headers are minimal: `@file` plus a one-line `@brief`. No
`@author`, `@date`, `@version`, or `@copyright` (version control owns
that metadata).

```cpp
/**
 * @file generator.h
 * @brief Reproducible random fills over seed and subsequence.
 */
```

When the file needs one non-obvious fact, it goes in the same block:

```cpp
/**
 * @file philox.h
 * @brief The Philox4x32 counter-based generator, host and device.
 *
 * Host and device share this exact code so fills match bitwise.
 */
```

### 3.2 Namespace

```cpp
/**
 * @namespace tests::rng
 * @brief Fixture fill helpers over the generator handle API.
 */
namespace tests::rng {
```

One line unless the namespace carries a real usage rule.

### 3.3 Class / Struct

`@brief` is the elevator pitch. The detailed section holds only the
contracts a user would otherwise get wrong. `@tparam` documents
template parameters.

```cpp
/**
 * @class Generator
 * @brief One reproducible fixture stream.
 *
 * A failed init is reported by the next call, not by the constructor.
 * Copies fork the stream and each copy advances on its own. If a
 * device drain fails the fill already landed: regard the handle as
 * spent.
 */
class Generator {
```

For plain C structs, trailing `///<` per member is the cleanest form:

```cpp
/**
 * @brief Live state for one fill stream.
 */
typedef struct Generator_ {
    RngEngine engine;     ///< Backend driving the fills.
    Device_ device;       ///< Placement the handle fills.
    uint64_t seed;        ///< Root of the stream.
    uint64_t subsequence; ///< Stream index within the seed.
    union {
        PhiloxState philox;   ///< Live state when @c engine is Philox4x32.
        XoshiroState xoshiro; ///< Live state when @c engine is Xoshiro256.
    } state;
    uint32_t lane_cache[RAND_LANE_CACHE_CAP];
    uint32_t lane_count;  ///< Parked lanes in @c lane_cache (0 to cap).
} Generator_;
```

Member comments that restate the member name (`uint32_t count; ///<
The count.`) are deleted on sight.

### 3.4 Enum

Document the enum type when it carries a rule; document each value
when the value's behavior is not in its name.

```cpp
/**
 * @enum RngEngine
 * @brief Generator backend behind a handle.
 */
typedef enum RngEngine_ {
    RNG_PHILOX4X32 = 0, ///< Counter-based, subsequence lands in the key.
    RNG_XOSHIRO256 = 1  ///< Host only, init costs one jump per unit.
} RngEngine;
```

### 3.5 Function (C status API)

NovaNN's C API returns `novaStatus_t` everywhere. Document the
contract, the params with direction, and the return in one place. Do
not split the same fact across `@details`, `@retval`, and `@return`.

```cpp
/**
 * @brief Initialize a handle from seed and subsequence.
 *
 * Xoshiro reaches a subsequence by walking one jump from a fresh
 * state per unit, so initialization costs O(subsequence) time; worker
 * indices should stay small, or prefer Philox4x32 whose subsequence
 * lands directly in the key. Impossible engine/device pairs are
 * rejected before the handle is touched. Initialization performs no
 * device probe.
 *
 * @param[out] gen         Destination handle. Must not be nullptr.
 * @param[in]  engine      Engine backing the handle.
 * @param[in]  device      Placement the handle fills.
 * @param[in]  seed        Seed value. Zero is valid.
 * @param[in]  subsequence Stream index. Worker @c w uses @c w.
 *
 * @return Status with @c err set to @c novaSuccess on success, or
 *         @c novaInvalidValue for an impossible engine/device pair.
 */
novaStatus_t generator_init(Generator_ *gen, RngEngine engine,
                            Device_ device, uint64_t seed,
                            uint64_t subsequence);
```

Note what this block does NOT have: no `@retval` table duplicating the
`@return` sentence, no `@note` wrapper around facts that are just
facts, no `@pre` restating "must not be nullptr" that `@param` already
carries.

### 3.6 Class Method (C++ with a hazard)

When a method's body contains a workaround, the doc explains the hazard
it defends against, once:

```cpp
  /**
   * @brief Fills @p n 32-bit lanes.
   *
   * A device placement drains the null path before returning.
   */
  [[nodiscard]] novaStatus_t fillU32(uint32_t *dst, size_t n);
```

The drain's own helper (section 1.3) carries the full why. The public
method stays one line plus the contract.

### 3.7 Overloaded Functions

```cpp
/**
 * @brief Draws @p count instances of the bound geometry.
 *
 * @param[in] count        Number of instances. Must be >= 1.
 * @param[in] firstVertex  Index of the first vertex.
 */
void draw(uint32_t count, uint32_t firstVertex = 0);

/**
 * @brief Draws indexed geometry bound via @ref bindIndexBuffer().
 *
 * @param[in] indexCount   Number of indices to draw.
 * @param[in] firstIndex   Offset into the bound index buffer.
 * @param[in] vertexOffset Added to each index before fetching.
 */
void draw(uint32_t indexCount, uint32_t firstIndex, int32_t vertexOffset);
```

Both overloads get real briefs; `@overload` boilerplate adds nothing
here. Use `@overload` only when the generated text is genuinely what
you want.

### 3.8 Constructor & Destructor

```cpp
  /**
   * @brief Initializes the stream.
   *
   * Xoshiro runs host side only.
   */
  explicit Generator(Device_ device, uint64_t seed = kDefaultSeed,
                     uint64_t subsequence = 0, RngEngine engine = Philox4x32);
```

If the destructor has no contract beyond "frees", it takes no comment
at all.

### 3.9 `#define` Macro

```cpp
/**
 * @def CLAMP(val, lo, hi)
 * @brief Clamps @p val to the closed interval [@p lo, @p hi].
 *
 * Each argument is evaluated once. Behavior is undefined if
 * @p lo > @p hi.
 *
 * @warning Arguments with side effects (e.g. @c CLAMP(i++, 0, 10))
 *          misbehave: this is a macro, not a function.
 */
#define CLAMP(val, lo, hi) ((val) < (lo) ? (lo) : ((val) > (hi) ? (hi) : (val)))
```

The `@warning` earns its place: macro argument evaluation is a real
hazard.

### 3.10 `typedef` and Type Aliases

```cpp
/**
 * @brief 32-bit lane index into vertex buffers.
 *
 * Alias kept so changing the index width is a one-line edit.
 */
using VertexIndex = uint32_t;
```

If the alias is self-evident (`using Buffer = VkBuffer;`), one line or
nothing.

### 3.11 Global Variables

```cpp
/**
 * @brief Maximum frames the CPU may run ahead of the GPU.
 *
 * Higher values hide CPU stalls and add latency. Changing this after
 * device init is undefined behavior.
 */
extern uint32_t g_maxFramesInFlight;
```

---

## 4. Advanced Patterns

### 4.1 Module / Group Documentation

Group related functionality that spans multiple files. Define the
group once; add members from any file with `@ingroup` or
`@addtogroup` + `@{` / `@}`.

```cpp
/**
 * @defgroup ncore_rand Random fills
 * @brief Counter-based generators, handles, and fill kernels.
 */
```

```cpp
/**
 * @addtogroup ncore_rand
 * @{
 */

/// Advances one handle by @p n lanes without writing memory.
novaStatus_t generator_skip(Generator_ *gen, size_t n);

/// Copies a handle and its parked lanes.
novaStatus_t generator_clone(const Generator_ *src, Generator_ *dst);

/** @} */
```

### 4.2 Internal vs Public

```cpp
/**
 * @brief Initializes the memory allocator.
 *
 * @internal
 * Two-level buddy system, 64-byte minimum block. Pool sizes come from
 * the VRAM reported at startup, not from a config constant.
 * @endinternal
 *
 * @param device  Vulkan logical device.
 */
void initAllocator(VkDevice device);
```

`@internal` hides implementation notes from the public reference while
keeping them in the source.

### 4.3 `@copydoc` for Overridden Virtuals

When an override has identical semantics to the base, do not duplicate:

```cpp
  /** @copydoc Device::submit() */
  void submit(CommandBuffer& cmd) override;
```

Add a line only for implementation-specific facts:

```cpp
  /**
   * @copydoc Device::submit()
   *
   * Submits to the graphics queue.
   */
  void submit(CommandBuffer& cmd) override;
```

### 4.4 Requirement Traceability Is Out of Scope for Source

Do not add `@requirement`, `@satisfies`, or `@verifies` to NovaNN
source: requirement specs stay local and code never cites them. Keep
traceability in the specs, not in committed comments.

### 4.5 C Struct as OOP (Manual Inheritance)

Rare in this codebase; use only where an existing pattern already
simulates inheritance:

```cpp
/**
 * @brief Base "class" for backend handles.
 */
struct Animal {
    const char *name; ///< Display name.
    int age;          ///< Whole years.
};

/**
 * @brief Makes a sound appropriate for the animal.
 * @memberof Animal
 * @param[in] self The instance. Must not be nullptr.
 */
void animal_speak(struct Animal *self);
```

---

## 5. Special Commands Reference

Lookup material. The voice rules of section 1 decide *what to say*;
this section catalogues *what exists*. Commands accept arguments with
the following scoping rules:

- `<word>`: single word (stops at first whitespace)
- `(line)`: entire remainder of the line
- `{paragraph}`: text up to the next blank line or sectioning command

### 5.1 Structural / Entity Declaration Commands

These commands explicitly tell Doxygen what kind of entity a detached
comment block documents. They are only needed when the doc block is
**not** placed directly before the entity.

| Command | Summary |
|---|---|
| `@file [name]` | Marks a comment block as the documentation for a source or header file. Required for globals to appear in the output unless `EXTRACT_ALL=YES`. |
| `@class <name> [header] [headername]` | Declares that the block documents a class. Optionally specifies the header to `#include`. |
| `@struct <name> [header] [headername]` | Same as `@class` but for `struct`. |
| `@union <name> [header] [headername]` | Same as `@class` but for `union`. |
| `@enum <name>` | Marks the block as documenting an enumeration. |
| `@fn (declaration)` | Marks the block as documenting a function. Full declaration required on one line. Avoid unless the block is detached from the function. |
| `@var (declaration)` | Marks the block as documenting a variable, global, or member. |
| `@typedef (declaration)` | Marks the block as documenting a `typedef`. |
| `@property (name)` | Marks the block as documenting a property. |
| `@def <name>` | Marks the block as documenting a `#define` macro. |
| `@namespace <name>` | Marks the block as documenting a namespace. |
| `@concept <name>` | Marks the block as documenting a C++20 concept. |
| `@module <name>` | Marks the block as documenting a C++20 module. |
| `@interface <name> [header] [headername]` | Documents an interface (Objective-C / IDL). |
| `@protocol <name> [header] [headername]` | Documents an Objective-C protocol. |
| `@category <name> [header] [headername]` | Documents an Objective-C category. |
| `@idlexcept <name>` | Documents an IDL exception. |
| `@headerfile <header> [headername]` | Specifies the header file users must include to use the class/struct/union documented in the current block. |
| `@dir [path]` | Marks the block as documenting a directory. |
| `@page <name> (title)` | Creates a standalone documentation page not tied to any code entity. |
| `@mainpage [(title)]` | Customizes the root index page of the generated documentation. |
| `@example[{lineno}] <filename>` | Associates the block with a source code example file. |

### 5.2 Description Commands

| Command | Summary |
|---|---|
| `@brief {text}` | One-sentence summary. Used in index listings and member declarations. Synonymous with `@short`. |
| `@short {text}` | Alias for `@brief`. |
| `@details {text}` | Begins the detailed description. A blank line after `@brief` implicitly starts `@details`. |
| `@par [(title)] {text}` | Custom-titled paragraph. Without a title, starts a new paragraph. |
| `@note {text}` | Indented "Note:" callout paragraph. |
| `@warning {text}` | Indented "Warning:" callout. Use for behavior that can crash, corrupt data, or cause UB. |
| `@attention {text}` | Indented "Attention:" callout. Slightly less severe than `@warning`. |
| `@important {text}` | Indented "Important:" callout. |
| `@remark {text}` / `@remarks {text}` | Indented "Remarks:" paragraph. |
| `@author {text}` / `@authors {text}` | Lists the author(s). Not used in NovaNN (version control owns attribution). |
| `@version {text}` | Documents the version string. Not used in NovaNN. |
| `@date {text}` | Documents creation or modification date. Not used in NovaNN. |
| `@copyright {text}` | Documents the copyright notice. Not used in NovaNN. |
| `@since {text}` | Since which version or date an entity is available. |
| `@deprecated {text}` | Marks the entity as deprecated. Cross-referenced in a global "Deprecated" list. |
| `@bug {text}` | Documents a known bug. Cross-referenced in a global "Bug List". |
| `@todo {text}` | Documents a pending task. Cross-referenced in a global "Todo List". |
| `@test {text}` | Describes a test case. Cross-referenced in a global "Test List". |
| `@xrefitem <key> "heading" "list title" {text}` | Generalized version of `@todo`/`@bug`; creates custom cross-referenced lists. |

### 5.3 Parameter & Return Commands

| Command | Summary |
|---|---|
| `@param[dir] <name> {desc}` | Documents a function parameter. `[dir]` is optional: `[in]`, `[out]`, `[in,out]`. |
| `@tparam <name> {desc}` | Documents a template parameter. |
| `@return {desc}` / `@returns {desc}` / `@result {desc}` | Documents the return value. All three are synonymous. |
| `@retval <value> {desc}` | Documents a specific named return value. Use when the value list adds information beyond `@return`. |
| `@throw <exception> {desc}` / `@throws <exception> {desc}` / `@exception <exception> {desc}` | Documents a thrown exception. All three are synonymous. |
| `@pre {desc}` | Precondition that must hold before the call. |
| `@post {desc}` | Postcondition guaranteed after the call. |
| `@invariant {desc}` | Class invariant: always true between public method calls. |
| `@parblock` / `@endparblock` | Wraps a multi-paragraph description inside a `@param` or similar single-paragraph command. |

### 5.4 Linking & Cross-Reference Commands

| Command | Summary |
|---|---|
| `@ref <name> ["text"]` | Cross-reference link to a symbol, section, page, or anchor. |
| `@link <target>` / `@endlink` | Hyperlink with custom label text. |
| `@anchor <word>` | Invisible named anchor, reachable via `@ref`. |
| `@see {refs}` / `@sa {refs}` | "See also" section. Both synonymous. |
| `@cite[{opts}] <label>` | Bibliographic citation from a BibTeX `.bib` file. |
| `@refitem <name>` | Like `@ref` but inserts the reference into a secreflist. |
| `@secreflist` / `@endsecreflist` | Wraps a list of `@refitem` entries into a formatted reference list. |
| `@subpage <name> ["text"]` | Links to a subpage, creating a parent-child page relationship. |

### 5.5 Grouping Commands

| Command | Summary |
|---|---|
| `@defgroup <name> (title)` | Defines a documentation group (topic). |
| `@addtogroup <name> [(title)]` | Adds entities to an existing group without warning on redefinition. Used with `@{` / `@}`. |
| `@weakgroup <name> [(title)]` | Same as `@addtogroup` but with lower priority on title conflicts. |
| `@ingroup <groupname>` | Marks the current entity as belonging to one or more groups. |
| `@name [(header)]` | Creates a named member group within a class, preceded by `@{` / `@}`. |
| `@nosubgrouping` | Placed in a class doc block to prevent member groups from appearing as sub-groups of access sections. |
| `@{ / @}` | Opens/closes a group membership block. |

### 5.6 Graph Control Commands

These toggle automatic graph generation per entity, overriding global
config.

| Command | Summary |
|---|---|
| `@callgraph` / `@hidecallgraph` | Force-shows or force-hides the call graph for a function. |
| `@callergraph` / `@hidecallergraph` | Force-shows or force-hides the caller graph. |
| `@collaborationgraph` / `@hidecollaborationgraph` | Force-shows or force-hides the collaboration graph for a class. |
| `@inheritancegraph[{opt}]` / `@hideinheritancegraph` | Force-shows or force-hides the inheritance graph. |
| `@includegraph` / `@hideincludegraph` | Force-shows or force-hides the include dependency graph for a file. |
| `@includedbygraph` / `@hideincludedbygraph` | Force-shows or force-hides the included-by graph. |
| `@directorygraph` / `@hidedirectorygraph` | Force-shows or force-hides the directory dependency graph. |
| `@groupgraph` / `@hidegroupgraph` | Force-shows or force-hides the group dependency graph. |
| `@showenumvalues` / `@hideenumvalues` | Force-shows or force-hides numeric values in an enum. |
| `@showinlinesource` / `@hideinlinesource` | Force-shows or force-hides inline source for a member. |
| `@showinitializer` / `@hideinitializer` | Force-shows or force-hides the initializer value. |
| `@showrefby` / `@hiderefby` | Force-shows or force-hides the "referenced by" overview. |
| `@showrefs` / `@hiderefs` | Force-shows or force-hides the "references" overview. |

### 5.7 Conditional Documentation Commands

| Command | Summary |
|---|---|
| `@cond [(label)]` / `@endcond` | Conditionally excludes an entire section (spanning multiple comment blocks). Controlled by `ENABLED_SECTIONS` in Doxyfile. |
| `@if (label)` / `@endif` | Conditionally includes a section **within** a single comment block. |
| `@ifnot (label)` / `@endif` | Same as `@if` but inverted. |
| `@else` | Alternative branch of an `@if` / `@ifnot`. |
| `@elseif (label)` | Else-if branch. |

### 5.8 Code & Verbatim Inclusion Commands

| Command | Summary |
|---|---|
| `@code[{lang}]` / `@endcode` | Syntax-highlighted code block. `{.cpp}`, `{.c}`, `{.py}` control the language. |
| `@verbatim` / `@endverbatim` | Block of text with no formatting or interpretation. |
| `@include[{opts}] <file>` | Includes the full content of a file. Options: `{lineno}`, `{doc}`, `{local}`. |
| `@includelineno <file>` | Includes a file with line numbers. |
| `@dontinclude <file>` | Marks a file for selective inclusion; `@line`, `@skip`, `@skipline`, `@until` navigate it. |
| `@line <pattern>` | Includes the next line matching `pattern` from the `@dontinclude` file. |
| `@skip <pattern>` | Skips forward to the line matching `pattern`. |
| `@skipline <pattern>` | Skips to and includes the line matching `pattern`. |
| `@until <pattern>` | Includes lines up to and including the match. |
| `@snippet[{opts}] <file> <id>` | Includes a labeled snippet from an external file. |
| `@snippetlineno <file> <id>` | Same as `@snippet` with line numbers. |
| `@snippetdoc <file> <id>` | Same as `@snippet` but also parses Doxygen commands in the snippet. |
| `@verbinclude <file>` | Includes a file verbatim (no highlighting). |
| `@htmlinclude <file>` | Includes HTML content directly in HTML output. |
| `@latexinclude <file>` | Includes LaTeX content in LaTeX output. |
| `@docbookinclude <file>` | Includes DocBook content in DocBook output. |
| `@maninclude <file>` | Includes content in man-page output. |
| `@rtfinclude <file>` | Includes content in RTF output. |
| `@xmlinclude <file>` | Includes content in XML output. |
| `@includedoc <file>` | Includes and processes a file as documentation. |

### 5.9 Output-Format-Specific Commands

| Command | Summary |
|---|---|
| `@htmlonly` / `@endhtmlonly` | Block only included in HTML output. |
| `@latexonly` / `@endlatexonly` | Block only included in LaTeX output. |
| `@docbookonly` / `@enddocbookonly` | Block only included in DocBook output. |
| `@manonly` / `@endmanonly` | Block only included in man-page output. |
| `@rtfonly` / `@endrtfonly` | Block only included in RTF output. |
| `@xmlonly` / `@endxmlonly` | Block only included in XML output. |

### 5.10 Diagram Commands

| Command | Summary |
|---|---|
| `@dot` / `@enddot` | Embeds a Graphviz DOT diagram inline. |
| `@dotfile <file> ["caption"] [{size}]` | Embeds a DOT diagram from an external file. |
| `@msc` / `@endmsc` | Embeds a Message Sequence Chart inline. |
| `@mscfile <file> ["caption"] [{size}]` | Embeds an MSC diagram from an external file. |
| `@startuml` / `@enduml` | Embeds a PlantUML diagram. |
| `@mermaid` / `@endmermaid` | Embeds a Mermaid diagram. |
| `@mermaidfile <file> ["caption"]` | Embeds a Mermaid diagram from an external file. |
| `@diafile <file> ["caption"] [{size}]` | Embeds a Dia diagram from an external file. |
| `@image {format} <file> ["caption"] [{size}]` | Inserts an image. Format: `html`, `latex`, `docbook`, etc. |
| `@vhdlflow [(title)]` | VHDL-specific: flow chart from a process block. |

### 5.11 Math Commands (LaTeX)

| Command | Summary |
|---|---|
| `@f$` ... `@f$` | Inline LaTeX math formula. |
| `@f[` ... `@f]` | Display (block) LaTeX math formula. |
| `@f(` ... `@f)` | Formula in a custom environment (defined by `FORMULA_MACROFILE`). |
| `@f{env}` ... `@f}` | Formula in a named LaTeX environment (e.g. `@f{align*}`). |

### 5.12 Inline Formatting Commands

| Command | Summary |
|---|---|
| `@a <word>` | Italic; conventionally a parameter name inline. |
| `@b <word>` | **Bold**. |
| `@c <word>` | `Monospace` (code font). |
| `@e <word>` / `@em <word>` | *Italic* (emphasis). Synonymous. |
| `@p <word>` | `Code font`; conventionally a parameter reference. |
| `@n` | Forced line break. |
| `@li {text}` / `@arg {text}` | Single bulleted list entry. |
| `@emoji <name>` | Emoji by name (e.g. `@emoji smile`). |

### 5.13 Section Structure Commands (for Pages)

| Command | Summary |
|---|---|
| `@section <id> (title)` | Top-level section within a `@page` or `@mainpage`. |
| `@subsection <id> (title)` | Subsection within a `@section`. |
| `@subsubsection <id> (title)` | Sub-subsection. |
| `@paragraph <id> (title)` | Titled paragraph below subsection level. |
| `@subparagraph <id> (title)` | Sub-paragraph. |
| `@subsubparagraph <id> (title)` | Sub-sub-paragraph. |
| `@tableofcontents` | Table of contents for the current page. |
| `@addindex (text)` | Adds an entry to the LaTeX/DocBook/RTF index. |

### 5.14 Traceability & Requirements Commands

`@requirement`, `@satisfies`, `@verifies`: defined by Doxygen, but see
section 4.4. NovaNN keeps requirement traceability out of source
comments.

### 5.15 Visibility & Access Commands

Primarily for C (no native access control) or to override detection.

| Command | Summary |
|---|---|
| `@public` | Marks the member as public. |
| `@protected` | Marks the member as protected. |
| `@private` | Marks the member as private. |
| `@publicsection` | Starts a public members section. |
| `@protectedsection` | Starts a protected members section. |
| `@privatesection` | Starts a private members section. |
| `@static` | Marks the member as static (languages without native static). |
| `@pure` | Marks the member as pure virtual. |
| `@qualifier <label>` | Custom qualifier badge (e.g. `@qualifier threadsafe`). |

### 5.16 Inheritance Commands (for C)

| Command | Summary |
|---|---|
| `@memberof <name>` | Makes a free function a member of a class in the docs (C OOP simulation). |
| `@extends <name>` | Declares an inheritance relation manually. |
| `@implements <name>` | Declares interface implementation manually. |
| `@relates <name>` / `@related <name>` | Puts a free function in the "Related functions" section of a class. |
| `@relatesalso <name>` / `@relatedalso <name>` | Same as `@relates` but keeps the function in its original location. |
| `@overload [(declaration)]` | Generates the standard overloaded function boilerplate text. |

### 5.17 Documentation Copying Commands

| Command | Summary |
|---|---|
| `@copydoc <symbol>` | Copies the full documentation (brief + detailed) from another symbol. |
| `@copybrief <symbol>` | Copies only the `@brief`. |
| `@copydetails <symbol>` | Copies only the detailed section. |

### 5.18 Utility & Miscellaneous Commands

| Command | Summary |
|---|---|
| `@internal` / `@endinternal` | Marks a block as internal, hidden unless `INTERNAL_DOCS=YES`. |
| `@noop (text)` | Ignores the rest of the line. Used in `ALIASES` to neutralize commands. |
| `@raisewarning (text)` | Emits a Doxygen warning during generation. |
| `@showdate "<fmt>" [datetime]` | Renders a formatted date/time string. |
| `@fileinfo[{opt}]` | Inserts the current file name (options: `name`, `extension`, `filename`, `directory`, `full`). |
| `@lineinfo` | Inserts the current line number. |
| `@doxyconfig <option>` | Inserts the value of a Doxygen configuration option. |

### 5.19 Character Escape Commands

| Command | Renders |
|---|---|
| `@$` | `$` |
| `@@` | `@` |
| `@\` | `\` |
| `@&` | `&` |
| `@~[lang]` | Toggles language-specific output |
| `@<` | `<` |
| `@>` | `>` |
| `@#` | `#` |
| `@%` | `%` |
| `@"` | `"` |
| `@.` | `.` (ends a sentence before Doxygen would otherwise stop `@brief`) |
| `@?` | `?` |
| `@!` | `!` |
| `@::` | `::` |
| `@\|` | `\|` |
| `@--` | en-dash |
| `@---` | em-dash |

NovaNN comments avoid dash separators entirely; these escapes exist for
the rare case a range or minus sign must render literally.

---

## Quick Reference: Comment Style Cheat Sheet

```
// ─── Before a declaration ───────────────────────────────────────────────────

/** One-line brief. */
void simpleFunc();

/**
 * @brief Brief description.
 *
 * Details only for a why the signature cannot carry: ordering,
 * ownership, invariants, hardware constraints.
 *
 * @param[in]  x  First input.
 * @param[out] y  Receives the result.
 * @return Status code; see the details block for the error cases.
 */
bool complexFunc(int x, int *y);

/// Short single-line description.
constexpr int MAX_JOINTS = 256;

// ─── Inline / trailing (struct members, enum values) ────────────────────────

struct Foo {
    int bar; ///< Adds what the name cannot say (units, range, lifetime).
    int baz; ///< Same bar here.
};

// ─── A status API function, complete ────────────────────────────────────────

/**
 * @brief Normalizes @p x into [0, 1].
 *
 * Subnormal inputs flush to zero: the SIMD path has no denormal
 * handling and the scalar path must match it bitwise.
 *
 * @param[in] x  Input value, any finite float.
 * @return Status; @c novaInvalidValue when @p x is not finite.
 */
float normalize(float x);
```

The kitchen-sink block (every command stacked on one function) is the
anti-pattern this guide exists to remove: a doc that lists `@note`,
`@attention`, `@pre`, `@post`, `@todo`, `@bug`, `@since`, and
`@deprecated` on one function is noise with formatting.
