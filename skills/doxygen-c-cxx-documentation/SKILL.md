---
name: doxygen-c-cxx-documentation
description: 'Generate and manage modern Doxygen-style documentation for C and C++ source code.'
---

# Doxygen C/C++ Documentation — Complete Reference

> **Scope**: This document assumes you are already familiar with Doxygen's toolchain and comment parsing. It is a dense, prescriptive reference for writing high-quality, structured documentation in C/C++ projects. All special commands use the `@` prefix (modern C/C++ convention).

---

## Table of Contents

1. [Comment Style Guide](#1-comment-style-guide)
2. [Special Commands Reference](#2-special-commands-reference)
3. [Documentation Patterns by Entity](#3-documentation-patterns-by-entity)
4. [Advanced Patterns](#4-advanced-patterns)

---

## 1. Comment Style Guide

Choosing the right comment style is not cosmetic — it determines **what Doxygen parses**, **where it attaches the doc block**, and **how readable the source remains**.

### 1.1 Block Comment Styles

| Syntax | Name | Use When |
|---|---|---|
| `/** ... */` | Javadoc block | Standard choice for all entity docs preceding the declaration |
| `/*! ... */` | Qt/Doxygen block | Identical to `/**`; preferred in some Qt codebases |
| `/** @brief ... */` on one line | Compact block | Short single-concept docs |

Both `/**` and `/*!` are semantically equivalent. Pick one and be consistent across the entire project.

```cpp
/**
 * This is a Javadoc-style block.
 * It continues over multiple lines.
 * Every line typically starts with ' * ' for alignment (optional but conventional).
 */

/*!
 * This is a Qt/Doxygen-style block.
 * Functionally identical to the one above.
 */
```

### 1.2 Single-Line Comment Styles

| Syntax | Name | Use When |
|---|---|---|
| `///` | C++ triple-slash | Preceding a declaration on its own line |
| `//!` | Doxygen exclamation | Preceding a declaration; alternative to `///` |

```cpp
/// Brief description of the following entity.
/// Can span multiple consecutive lines — they are merged.
int myFunction();

//! Alternative single-line style.
//! Also mergeable across consecutive lines.
int anotherFunction();
```

### 1.3 Trailing (Inline / Member) Comment Styles

These are attached **after** the entity they document, on the same line or immediately following. The `<` modifier tells Doxygen to associate the comment with the *preceding* entity.

| Syntax | Name | Use When |
|---|---|---|
| `///<` | Trailing C++ line | Documenting struct/class members, enum values inline |
| `//!<` | Trailing exclamation line | Alternative to `///<` |
| `/**< ... */` | Trailing Javadoc block | When you need a multi-sentence trailing comment |
| `/*!< ... */` | Trailing Qt block | Alternative to `/**<` |

```cpp
struct Vertex {
    float x; ///< X coordinate in world space.
    float y; ///< Y coordinate in world space.
    float z; ///< Z coordinate in world space.
    float w; ///< Homogeneous coordinate (1.0 for points, 0.0 for directions).
};

enum class Status {
    OK      = 0, ///< Operation completed successfully.
    TIMEOUT = 1, ///< Operation exceeded the time limit.
    ERROR   = 2  ///< Unrecoverable internal error.
};
```

### 1.4 The Exact Rules for Comment Attachment

- A `/** */` or `/*!` or `///` placed **directly before** a declaration is attached to that entity.
- A `///<`, `//!<`, `/**< */`, or `/*!< */` placed **on the same line as** or **immediately after** a declaration is attached to that entity.
- There must be **no blank lines** between the comment and the entity it documents for the attachment to hold.
- For **files**, the doc block must contain an explicit `@file` command OR must be the first doc block in the file (depending on config).

### 1.5 Style Decision Matrix

```
Documenting a function, class, struct, typedef, namespace → /** ... */
Documenting a struct/class member inline                   → ///<
Documenting an enum value inline                           → ///<
Short one-liner before a forward declaration               → ///
Multi-sentence trailing comment on a member                → /**< ... */
File-level documentation block                             → /** @file ... */
```

---

## 2. Special Commands Reference

All commands are prefixed with `@`. Commands accept arguments with the following scoping rules:

- `<word>` — Single word (stops at first whitespace)
- `(line)` — Entire remainder of the line
- `{paragraph}` — Text up to the next blank line or sectioning command

### 2.1 Structural / Entity Declaration Commands

These commands explicitly tell Doxygen what kind of entity a detached comment block documents. They are only needed when the doc block is **not** placed directly before the entity.

| Command | Summary |
|---|---|
| `@file [name]` | Marks a comment block as the documentation for a source or header file. Required for globals to appear in the output unless `EXTRACT_ALL=YES`. |
| `@class <name> [header] [headername]` | Declares that the block documents a class. Optionally specifies the header to `#include`. |
| `@struct <name> [header] [headername]` | Same as `@class` but for `struct`. |
| `@union <name> [header] [headername]` | Same as `@class` but for `union`. |
| `@enum <name>` | Marks the block as documenting an enumeration. |
| `@fn (declaration)` | Marks the block as documenting a function. Full declaration required on one line. Avoid unless the block is detached from the function. |
| `@var (declaration)` | Marks the block as documenting a variable, global, or member. Equivalent to `@fn` for variables. |
| `@typedef (declaration)` | Marks the block as documenting a `typedef`. Equivalent to `@fn`. |
| `@property (name)` | Marks the block as documenting a property. Equivalent to `@fn`. |
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

### 2.2 Description Commands

These commands structure the body of a documentation block.

| Command | Summary |
|---|---|
| `@brief {text}` | One-sentence (or one-paragraph) summary. Used in index listings and member declarations. Synonymous with `@short`. |
| `@short {text}` | Alias for `@brief`. |
| `@details {text}` | Begins the detailed description. A blank line after `@brief` implicitly starts `@details`. |
| `@par [(title)] {text}` | Creates a custom-titled paragraph. Without a title, just starts a new paragraph. Useful for grouping related notes inside a block. |
| `@note {text}` | Indented "Note:" callout paragraph. |
| `@warning {text}` | Indented "Warning:" callout. Use for behavior that can crash, corrupt data, or cause UB. |
| `@attention {text}` | Indented "Attention:" callout. Slightly less severe than `@warning`. |
| `@important {text}` | Indented "Important:" callout. |
| `@remark {text}` / `@remarks {text}` | Indented "Remarks:" paragraph for observations that don't fit other categories. |
| `@author {text}` / `@authors {text}` | Lists the author(s) of the entity. Multiple `@author` lines are merged. |
| `@version {text}` | Documents the version string. |
| `@date {text}` | Documents creation or modification date. |
| `@copyright {text}` | Documents the copyright notice for the entity. |
| `@since {text}` | Indicates since which version or date an entity is available. |
| `@deprecated {text}` | Marks the entity as deprecated. Cross-referenced in a global "Deprecated" list. |
| `@bug {text}` | Documents a known bug. Cross-referenced in a global "Bug List". |
| `@todo {text}` | Documents a pending implementation task. Cross-referenced in a global "Todo List". |
| `@test {text}` | Describes a test case. Cross-referenced in a global "Test List". |
| `@xrefitem <key> "heading" "list title" {text}` | Generalized version of `@todo`/`@bug`; creates custom cross-referenced lists. |

### 2.3 Parameter & Return Commands

| Command | Summary |
|---|---|
| `@param[dir] <name> {desc}` | Documents a function parameter. `[dir]` is optional: `[in]`, `[out]`, `[in,out]`. |
| `@tparam <name> {desc}` | Documents a template parameter for a function or class template. |
| `@return {desc}` / `@returns {desc}` / `@result {desc}` | Documents the return value. All three are synonymous. |
| `@retval <value> {desc}` | Documents a specific named return value (useful for enums or error codes returned). |
| `@throw <exception> {desc}` / `@throws <exception> {desc}` / `@exception <exception> {desc}` | Documents a thrown exception. All three are synonymous. |
| `@pre {desc}` | Describes a precondition that must be true before calling the function. |
| `@post {desc}` | Describes a postcondition guaranteed after the function returns. |
| `@invariant {desc}` | Describes a class invariant — a condition always true between public method calls. |
| `@parblock` / `@endparblock` | Wraps a multi-paragraph description inside a `@param` or similar single-paragraph command. |

### 2.4 Linking & Cross-Reference Commands

| Command | Summary |
|---|---|
| `@ref <name> ["text"]` | Creates a cross-reference link to a symbol, section, page, or anchor by name. |
| `@link <target>` / `@endlink` | Creates a hyperlink with custom text. Text between `@link` and `@endlink` is the label. |
| `@anchor <word>` | Places an invisible named anchor at the current location, reachable via `@ref`. |
| `@see {refs}` / `@sa {refs}` | "See also" section. Lists related classes, functions, or URLs. Both are synonymous. |
| `@cite[{opts}] <label>` | Inserts a bibliographic citation from a BibTeX `.bib` file. |
| `@refitem <name>` | Like `@ref` but inserts the reference into a secreflist. |
| `@secreflist` / `@endsecreflist` | Wraps a list of `@refitem` entries into a formatted reference list. |
| `@subpage <name> ["text"]` | Links to a subpage, creating a parent-child page relationship. |

### 2.5 Grouping Commands

| Command | Summary |
|---|---|
| `@defgroup <name> (title)` | Defines a documentation group (topic). The primary way to create module-level groupings. |
| `@addtogroup <name> [(title)]` | Adds entities to an existing group without warning on redefinition. Used to spread group members across files using `@{` / `@}`. |
| `@weakgroup <name> [(title)]` | Same as `@addtogroup` but with lower priority in case of group title conflicts. |
| `@ingroup <groupname>` | Marks the current entity as belonging to one or more groups. |
| `@name [(header)]` | Creates a named member group within a class, preceded by `@{` / `@}`. |
| `@nosubgrouping` | Placed in a class doc block to prevent member groups from appearing as sub-groups of access sections. |
| `@{ / @}` | Opens/closes a group membership block used with `@addtogroup`, `@defgroup`, or `@name`. |

### 2.6 Graph Control Commands

These toggle the automatic graph generation Doxygen can produce, overriding global config settings per entity.

| Command | Summary |
|---|---|
| `@callgraph` / `@hidecallgraph` | Force-shows or force-hides the call graph for a function. |
| `@callergraph` / `@hidecallergraph` | Force-shows or force-hides the caller graph for a function. |
| `@collaborationgraph` / `@hidecollaborationgraph` | Force-shows or force-hides the collaboration graph for a class. |
| `@inheritancegraph[{opt}]` / `@hideinheritancegraph` | Force-shows or force-hides the inheritance graph for a class. |
| `@includegraph` / `@hideincludegraph` | Force-shows or force-hides the include dependency graph for a file. |
| `@includedbygraph` / `@hideincludedbygraph` | Force-shows or force-hides the included-by graph for a header. |
| `@directorygraph` / `@hidedirectorygraph` | Force-shows or force-hides the directory dependency graph. |
| `@groupgraph` / `@hidegroupgraph` | Force-shows or force-hides the group dependency graph. |
| `@showenumvalues` / `@hideenumvalues` | Force-shows or force-hides numeric values in an enum documentation. |
| `@showinlinesource` / `@hideinlinesource` | Force-shows or force-hides inline source code for a member. |
| `@showinitializer` / `@hideinitializer` | Force-shows or force-hides the initializer value of a define or variable. |
| `@showrefby` / `@hiderefby` | Force-shows or force-hides the "referenced by" overview for a member. |
| `@showrefs` / `@hiderefs` | Force-shows or force-hides the "references" overview for a member. |

### 2.7 Conditional Documentation Commands

| Command | Summary |
|---|---|
| `@cond [(label)]` / `@endcond` | Conditionally excludes an entire section (spanning multiple comment blocks). Controlled by `ENABLED_SECTIONS` in Doxyfile. |
| `@if (label)` / `@endif` | Conditionally includes a section **within** a single comment block. |
| `@ifnot (label)` / `@endif` | Same as `@if` but inverted — section shown when label is **not** in `ENABLED_SECTIONS`. |
| `@else` | Starts the alternative branch of an `@if` / `@ifnot`. |
| `@elseif (label)` | Starts an else-if branch. |

### 2.8 Code & Verbatim Inclusion Commands

| Command | Summary |
|---|---|
| `@code[{lang}]` / `@endcode` | Inserts a syntax-highlighted code block. `{.cpp}`, `{.c}`, `{.py}` etc. control the language. |
| `@verbatim` / `@endverbatim` | Inserts a block of text with no formatting or interpretation. |
| `@include[{opts}] <file>` | Includes the full content of a file in the documentation. Options: `{lineno}`, `{doc}`, `{local}`. |
| `@includelineno <file>` | Includes a file with line numbers. |
| `@dontinclude <file>` | Marks a file for selective inclusion; subsequent `@line`, `@skip`, `@skipline`, `@until` commands navigate it. |
| `@line <pattern>` | Includes the next line matching `pattern` from the `@dontinclude` file. |
| `@skip <pattern>` | Skips forward to the line matching `pattern`. |
| `@skipline <pattern>` | Skips to and includes the line matching `pattern`. |
| `@until <pattern>` | Includes lines up to and including the line matching `pattern`. |
| `@snippet[{opts}] <file> <id>` | Includes a labeled snippet from an external file. |
| `@snippetlineno <file> <id>` | Same as `@snippet` but with line numbers. |
| `@snippetdoc <file> <id>` | Same as `@snippet` but also parses Doxygen commands in the snippet. |
| `@verbinclude <file>` | Includes a file verbatim (no syntax highlighting). |
| `@htmlinclude <file>` | Includes HTML content directly in the output. |
| `@latexinclude <file>` | Includes LaTeX content in LaTeX output. |
| `@docbookinclude <file>` | Includes DocBook content in DocBook output. |
| `@maninclude <file>` | Includes content in man-page output. |
| `@rtfinclude <file>` | Includes content in RTF output. |
| `@xmlinclude <file>` | Includes content in XML output. |
| `@includedoc <file>` | Includes and processes a file as documentation. |

### 2.9 Output-Format-Specific Commands

| Command | Summary |
|---|---|
| `@htmlonly` / `@endhtmlonly` | Block only included in HTML output. |
| `@latexonly` / `@endlatexonly` | Block only included in LaTeX output. |
| `@docbookonly` / `@enddocbookonly` | Block only included in DocBook output. |
| `@manonly` / `@endmanonly` | Block only included in man-page output. |
| `@rtfonly` / `@endrtfonly` | Block only included in RTF output. |
| `@xmlonly` / `@endxmlonly` | Block only included in XML output. |

### 2.10 Diagram Commands

| Command | Summary |
|---|---|
| `@dot` / `@enddot` | Embeds a Graphviz DOT diagram directly in the documentation. |
| `@dotfile <file> ["caption"] [{size}]` | Embeds a DOT diagram from an external `.dot` file. |
| `@msc` / `@endmsc` | Embeds a Message Sequence Chart (MSC) diagram. |
| `@mscfile <file> ["caption"] [{size}]` | Embeds an MSC diagram from an external file. |
| `@startuml` / `@enduml` | Embeds a PlantUML diagram. |
| `@mermaid` / `@endmermaid` | Embeds a Mermaid diagram. |
| `@mermaidfile <file> ["caption"]` | Embeds a Mermaid diagram from an external file. |
| `@diafile <file> ["caption"] [{size}]` | Embeds a Dia diagram from an external file. |
| `@dotfile`, `@plantumlfile`, `@mscfile` | Load diagrams from files rather than inline. |
| `@image {format} <file> ["caption"] [{size}]` | Inserts an image into the documentation. Format: `html`, `latex`, `docbook`, etc. |
| `@vhdlflow [(title)]` | VHDL-specific: generates a flow chart from a process block. |

### 2.11 Math Commands (LaTeX)

| Command | Summary |
|---|---|
| `@f$` ... `@f$` | Inline LaTeX math formula. |
| `@f[` ... `@f]` | Display (block) LaTeX math formula. |
| `@f(` ... `@f)` | Formula in a custom environment (defined by `FORMULA_MACROFILE`). |
| `@f{env}` ... `@f}` | Formula using a named LaTeX environment (e.g., `@f{align*}`). |

### 2.12 Inline Formatting Commands

These apply inline text markup within a paragraph.

| Command | Summary |
|---|---|
| `@a <word>` | Renders `word` in italic — conventionally used to refer to a parameter name inline. |
| `@b <word>` | Renders `word` in **bold**. |
| `@c <word>` | Renders `word` in `monospace` (code font). |
| `@e <word>` / `@em <word>` | Renders `word` in *italic* (emphasis). Synonymous. |
| `@p <word>` | Renders `word` in `code font`. Synonymous with `@c`, conventionally used for parameter references. |
| `@n` | Inserts a forced line break. |
| `@li {text}` / `@arg {text}` | Creates a single-item bulleted list entry. |
| `@emoji <name>` | Inserts an emoji by name (e.g., `@emoji smile`). |

### 2.13 Section Structure Commands (for Pages)

| Command | Summary |
|---|---|
| `@section <id> (title)` | Creates a top-level section within a `@page` or `@mainpage`. |
| `@subsection <id> (title)` | Creates a subsection within a `@section`. |
| `@subsubsection <id> (title)` | Creates a sub-subsection. |
| `@paragraph <id> (title)` | Creates a paragraph with a title below subsection level. |
| `@subparagraph <id> (title)` | Creates a sub-paragraph. |
| `@subsubparagraph <id> (title)` | Creates a sub-sub-paragraph. |
| `@tableofcontents` | Inserts a table of contents for the current page. |
| `@addindex (text)` | Adds an entry to the LaTeX/DocBook/RTF index. |

### 2.14 Traceability & Requirements Commands

| Command | Summary |
|---|---|
| `@requirement <id> [(title)]` | Defines a documented requirement, collected on a dedicated page. |
| `@satisfies <req-id> [(desc)]` | Links a function/class to a requirement it implements. |
| `@verifies <req-id> [(desc)]` | Links a function/class to a requirement it tests. |

### 2.15 Visibility & Access Commands

These are primarily for C (which has no native access control) or to override detection.

| Command | Summary |
|---|---|
| `@public` | Marks the documented member as public. |
| `@protected` | Marks the documented member as protected. |
| `@private` | Marks the documented member as private. |
| `@publicsection` | Starts a public members section (like `public:` in C++). |
| `@protectedsection` | Starts a protected members section. |
| `@privatesection` | Starts a private members section. |
| `@static` | Marks the documented member as static (for languages without native static). |
| `@pure` | Marks the documented member as pure virtual. |
| `@qualifier <label>` | Adds a custom qualifier badge (e.g., `@qualifier threadsafe`). |

### 2.16 Inheritance Commands (for C)

| Command | Summary |
|---|---|
| `@memberof <name>` | Makes a free function a true member of a class in the docs (C OOP simulation). |
| `@extends <name>` | Manually declares an inheritance relation (C OOP simulation). |
| `@implements <name>` | Manually declares interface implementation (C OOP simulation). |
| `@relates <name>` / `@related <name>` | Puts a free function in the "Related functions" section of a class. |
| `@relatesalso <name>` / `@relatedalso <name>` | Same as `@relates` but also keeps the function in its original location. |
| `@overload [(declaration)]` | Generates the standard overloaded function boilerplate text. |

### 2.17 Documentation Copying Commands

| Command | Summary |
|---|---|
| `@copydoc <symbol>` | Copies the full documentation (brief + detailed) from another symbol. |
| `@copybrief <symbol>` | Copies only the `@brief` from another symbol. |
| `@copydetails <symbol>` | Copies only the detailed section from another symbol. |

### 2.18 Utility & Miscellaneous Commands

| Command | Summary |
|---|---|
| `@internal` / `@endinternal` | Marks a block as internal documentation, hidden unless `INTERNAL_DOCS=YES`. |
| `@noop (text)` | Ignores everything on the rest of the line. Used in `ALIASES` to neutralize commands. |
| `@raisewarning (text)` | Emits a Doxygen warning with the given text during generation. |
| `@showdate "<fmt>" [datetime]` | Renders a formatted date/time string in the output. |
| `@fileinfo[{opt}]` | Inserts the current file name (or part of it) into the documentation. Options: `name`, `extension`, `filename`, `directory`, `full`. |
| `@lineinfo` | Inserts the current line number into the documentation. |
| `@doxyconfig <option>` | Inserts the value of a Doxygen configuration option in the output. |

### 2.19 Character Escape Commands

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
| `@.` | `.` (can be used to end a sentence before Doxygen would otherwise stop `@brief`) |
| `@?` | `?` |
| `@!` | `!` |
| `@::` | `::` |
| `@\|` | `\|` |
| `@--` | en-dash |
| `@---` | em-dash |

---

## 3. Documentation Patterns by Entity

### 3.1 File Header

Every `.h` / `.hpp` / `.cpp` file that exposes documented entities **must** have a file doc block. Without it, globals, free functions, and enums in that file are silently omitted unless `EXTRACT_ALL=YES`.

```cpp
/**
 * @file render_pipeline.h
 * @brief Declares the RenderPipeline class and supporting types.
 *
 * This header exposes the primary interface for submitting draw commands
 * to the GPU. It is the only header users need to include for basic rendering.
 *
 * @author Jane Doe <jane@example.com>
 * @date 2024-01-15
 * @version 2.3.0
 * @copyright Copyright (c) 2024 Example Corp. All rights reserved.
 *
 * @note This file requires a C++20-capable compiler and a Vulkan 1.3 runtime.
 */
```

### 3.2 Namespace

```cpp
/**
 * @namespace gfx
 * @brief Graphics subsystem — all rendering-related types live here.
 *
 * The `gfx` namespace wraps the entire rendering backend. Code outside
 * the rendering subsystem should interact exclusively through the interfaces
 * declared here, never through internal implementation headers.
 */
namespace gfx {
```

### 3.3 Class / Struct

Use `@brief` for the elevator-pitch summary. Put everything a user needs to understand the class contract — ownership model, thread safety, lifecycle — in the detailed section. `@tparam` documents template parameters.

```cpp
/**
 * @class RenderPipeline
 * @brief Manages the state machine for issuing GPU draw calls.
 *
 * RenderPipeline encapsulates a compiled shader pipeline, a descriptor set
 * layout, and the associated render pass. It is the central object users
 * interact with to submit geometry for rendering.
 *
 * **Ownership**: RenderPipeline owns its Vulkan handles. Copying is deleted;
 * move is supported and leaves the source in a valid but empty state.
 *
 * **Thread safety**: A single RenderPipeline instance must not be used
 * concurrently from multiple threads. Create one per thread or synchronize
 * externally.
 *
 * **Lifecycle**:
 * 1. Construct with a `Device&` reference and a `PipelineDesc`.
 * 2. Call `bind()` inside a render pass.
 * 3. Issue `draw*()` calls.
 * 4. Destroy (or let it go out of scope) — handles are freed automatically.
 *
 * @tparam VertexT Vertex layout type. Must satisfy the `VertexLayout` concept.
 *
 * @invariant `device_` is always a valid, live reference as long as `this` exists.
 *
 * @see gfx::Device
 * @see gfx::PipelineDesc
 */
template <VertexLayout VertexT>
class RenderPipeline {
```

### 3.4 Struct with Inline Member Documentation

For plain data structures, inline `///<` comments are the cleanest approach. They keep the struct definition and its documentation tightly coupled and visually aligned.

```cpp
/**
 * @struct PipelineDesc
 * @brief Describes the configuration for creating a RenderPipeline.
 *
 * All fields must be filled before passing the struct to the RenderPipeline
 * constructor. Fields tagged [optional] have documented defaults.
 */
struct PipelineDesc {
    const char*   vertShaderPath;  ///< Absolute path to the compiled SPIR-V vertex shader.
    const char*   fragShaderPath;  ///< Absolute path to the compiled SPIR-V fragment shader.
    VkRenderPass  renderPass;      ///< The render pass this pipeline will execute in.
    VkExtent2D    viewport;        ///< Viewport dimensions in pixels.
    uint32_t      subpass = 0;     ///< [optional] Subpass index within `renderPass`. Default: 0.
    bool          depthTest = true;///< [optional] Enable depth testing. Default: true.
    bool          wireframe = false;///< [optional] Render in wireframe mode. Default: false.
};
```

### 3.5 Enum

Document the enum type itself and each value. Trailing `///<` is preferred for values.

```cpp
/**
 * @enum BlendMode
 * @brief Controls how fragment colours are blended with the framebuffer.
 *
 * @note The `ADDITIVE` and `MULTIPLY` modes require the render pass
 * attachment to have `VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT` set.
 */
enum class BlendMode : uint8_t {
    OPAQUE   = 0, ///< No blending. Fragment colour replaces the destination.
    ALPHA    = 1, ///< Standard alpha blending (src_alpha / one_minus_src_alpha).
    ADDITIVE = 2, ///< Destination colour is brightened by the source colour.
    MULTIPLY = 3  ///< Destination colour is multiplied by the source colour.
};
```

### 3.6 Function — Complete Documentation

A fully documented function specifies brief, details, every parameter with direction, return value, every exception, pre/postconditions, and side effects.

```cpp
/**
 * @brief Allocates and binds a vertex buffer for the given mesh data.
 *
 * Creates a device-local `VkBuffer`, stages the data through a temporary host-
 * visible buffer, and issues the copy command. The staging buffer is destroyed
 * immediately after the copy fence signals.
 *
 * The caller retains ownership of `vertices`; this function copies the data
 * and does not hold a reference after returning.
 *
 * @tparam VertexT  Vertex layout type matching the pipeline's template argument.
 *
 * @param[in]  vertices   Pointer to the vertex array. Must not be null.
 * @param[in]  count      Number of vertices in `vertices`. Must be > 0.
 * @param[out] bufferOut  Receives the created buffer handle on success.
 *                        Unchanged on failure.
 * @param[in]  usage      Additional `VkBufferUsageFlags` to OR with the
 *                        mandatory `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`.
 *                        Pass 0 for the default.
 *
 * @return `VK_SUCCESS` on success, or a Vulkan result code on failure.
 *
 * @retval VK_SUCCESS               Buffer created and data uploaded.
 * @retval VK_ERROR_OUT_OF_DEVICE_MEMORY  Not enough VRAM to allocate the buffer.
 * @retval VK_ERROR_INITIALIZATION_FAILED Device lost during transfer.
 *
 * @throws std::invalid_argument  If `vertices` is null or `count` is zero.
 *
 * @pre  The `Device` associated with this pipeline must be in a valid state.
 * @pre  `count * sizeof(VertexT)` must not exceed `VkPhysicalDeviceLimits::maxStorageBufferRange`.
 * @post On `VK_SUCCESS`, `*bufferOut` holds a valid, device-local buffer containing `count` vertices.
 * @post On failure, the pipeline's internal state is unchanged.
 *
 * @note The function submits a one-time command buffer on the transfer queue
 *       and blocks until the copy is complete. For large meshes, prefer
 *       the async variant `uploadVertexBufferAsync()`.
 *
 * @warning Calling this function from a render pass (between `vkCmdBeginRenderPass`
 *          and `vkCmdEndRenderPass`) results in undefined behaviour.
 *
 * @see uploadVertexBufferAsync()
 * @see destroyBuffer()
 *
 * @callgraph
 */
template <VertexLayout VertexT>
VkResult RenderPipeline<VertexT>::uploadVertexBuffer(
    const VertexT*    vertices,
    uint32_t          count,
    VkBuffer*         bufferOut,
    VkBufferUsageFlags usage = 0);
```

### 3.7 Overloaded Functions

```cpp
/**
 * @brief Draws `count` instances of the bound geometry.
 *
 * Issues a `vkCmdDraw` call. Must be called inside an active render pass
 * with this pipeline bound.
 *
 * @param[in] count     Number of instances to draw. Must be >= 1.
 * @param[in] firstVertex  Index of the first vertex in the bound vertex buffer.
 */
void draw(uint32_t count, uint32_t firstVertex = 0);

/**
 * @overload
 *
 * Draws using an index buffer previously bound via `bindIndexBuffer()`.
 * Issues a `vkCmdDrawIndexed` call.
 *
 * @param[in] indexCount   Number of indices to draw.
 * @param[in] firstIndex   Offset into the bound index buffer.
 * @param[in] vertexOffset Added to each index value before fetching a vertex.
 */
void draw(uint32_t indexCount, uint32_t firstIndex, int32_t vertexOffset);
```

### 3.8 Constructor & Destructor

```cpp
/**
 * @brief Constructs and compiles a RenderPipeline from the given description.
 *
 * Compiles shader modules, creates the `VkPipeline`, and allocates descriptor
 * sets. This is a potentially expensive operation — do it at load time, not
 * per frame.
 *
 * @param[in] device  Live device reference. Must outlive this pipeline.
 * @param[in] desc    Complete pipeline description. All fields must be valid.
 *
 * @throws gfx::ShaderCompileError  If a shader path is invalid or the SPIR-V is malformed.
 * @throws gfx::DeviceLostError     If the Vulkan device is in a lost state.
 *
 * @note The `device` reference is stored internally. Ensure the device is not
 *       destroyed before this pipeline.
 */
explicit RenderPipeline(Device& device, const PipelineDesc& desc);

/**
 * @brief Destroys the pipeline and releases all Vulkan resources.
 *
 * Waits for any in-flight work referencing this pipeline to complete before
 * destroying handles. Safe to call even if construction threw an exception.
 */
~RenderPipeline() noexcept;
```

### 3.9 `#define` Macro

```cpp
/**
 * @file math_utils.h
 * @brief Numeric utility macros.
 */

/**
 * @def CLAMP(val, lo, hi)
 * @brief Clamps `val` to the closed interval [`lo`, `hi`].
 *
 * Each argument is evaluated once. Arguments must be of a type that supports
 * `<` comparison. Behaviour is undefined if `lo > hi`.
 *
 * @param val  The value to clamp.
 * @param lo   Inclusive lower bound.
 * @param hi   Inclusive upper bound.
 *
 * @warning This is a macro, not a function. Avoid passing expressions with
 *          side effects (e.g., `CLAMP(i++, 0, 10)`).
 */
#define CLAMP(val, lo, hi) ((val) < (lo) ? (lo) : ((val) > (hi) ? (hi) : (val)))
```

### 3.10 `typedef` and Type Aliases

```cpp
/**
 * @typedef VertexIndex
 * @brief 32-bit unsigned integer used as an index into vertex buffers.
 *
 * Use this alias throughout the rendering subsystem rather than `uint32_t`
 * directly so that changing the index size only requires updating this typedef.
 */
using VertexIndex = uint32_t;

/**
 * @typedef DrawCallback
 * @brief Function signature for per-frame draw submission callbacks.
 *
 * @param cmd   Command buffer to record into. Guaranteed valid for the frame duration.
 * @param dt    Delta time since the previous frame, in seconds.
 */
using DrawCallback = std::function<void(VkCommandBuffer cmd, float dt)>;
```

### 3.11 Global Variables

```cpp
/**
 * @var g_maxFramesInFlight
 * @brief Maximum number of frames the CPU may be ahead of the GPU.
 *
 * Increasing this value reduces CPU stalls but increases latency and memory usage.
 * Valid range: 1–3. Modifying this after device initialization is undefined behaviour.
 */
extern uint32_t g_maxFramesInFlight;
```

### 3.12 Template Class with `@tparam`

```cpp
/**
 * @class RingBuffer
 * @brief Lock-free single-producer single-consumer ring buffer.
 *
 * @tparam T       Element type. Must be trivially copyable.
 * @tparam Capacity Fixed capacity in number of elements. Must be a power of two.
 *
 * @note Only safe for exactly one producer thread and one consumer thread.
 *       For multi-producer or multi-consumer use, add external synchronisation.
 *
 * @invariant `head_` and `tail_` are always valid indices modulo `Capacity`.
 */
template <typename T, std::size_t Capacity>
    requires std::is_trivially_copyable_v<T> && (Capacity > 0) && ((Capacity & (Capacity - 1)) == 0)
class RingBuffer {
```

---

## 4. Advanced Patterns

### 4.1 Module / Group Documentation

Group related functionality that spans multiple files into a named module. Define the group once in a dedicated header or in a `groups.h` file; add members from any file using `@ingroup` or `@addtogroup`.

```cpp
// groups.h — define all top-level groups here

/**
 * @defgroup gfx_core Core Rendering
 * @brief Pipeline creation, command submission, and resource management.
 */

/**
 * @defgroup gfx_ui UI Rendering
 * @brief Immediate-mode and retained-mode UI primitives.
 * @ingroup gfx_core
 */
```

```cpp
// render_pipeline.h — add this class to the group

/**
 * @class RenderPipeline
 * @ingroup gfx_core
 * @brief ...
 */
```

```cpp
// utils.cpp — add a batch of free functions to a group using @{ / @}

/**
 * @addtogroup gfx_core
 * @{
 */

/// Creates a default pipeline descriptor with sensible defaults.
PipelineDesc makeDefaultPipelineDesc();

/// Destroys a previously allocated buffer and zeroes the handle.
void destroyBuffer(VkBuffer& buffer);

/** @} */ // end of gfx_core
```

### 4.2 Conditional Documentation (Internal vs Public)

```cpp
/**
 * @brief Initializes the memory allocator.
 *
 * @internal
 * The allocator uses a two-level buddy system with a minimum block size of 64 bytes.
 * Pool sizes are determined at startup based on available VRAM reported by
 * `vkGetPhysicalDeviceMemoryProperties`. This detail is not part of the public API.
 * @endinternal
 *
 * @param device  Vulkan logical device.
 * @param physDev Vulkan physical device handle for querying memory properties.
 */
void initAllocator(VkDevice device, VkPhysicalDevice physDev);
```

### 4.3 Custom Cross-Reference Lists (`@xrefitem`)

Define custom lists in your Doxyfile, then reference them in code:

```ini
# Doxyfile
ALIASES += "performance=@xrefitem perfnotes \"Performance Note\" \"Performance Notes\""
ALIASES += "threadsafe=@xrefitem tsafe \"Thread Safety\" \"Thread Safety Notes\""
```

```cpp
/**
 * @brief Sorts the draw call list by depth for correct alpha blending.
 *
 * @performance This function runs on the main thread every frame.
 *              For 10,000+ draw calls, consider caching the sorted list
 *              and only re-sorting when the scene changes.
 *
 * @threadsafe Must be called from the main thread only.
 */
void sortDrawCalls();
```

### 4.4 Embedded Diagrams

```cpp
/**
 * @brief Advances the swap chain to the next frame.
 *
 * The frame lifecycle is:
 *
 * @startuml
 * [*] --> AcquireImage
 * AcquireImage --> RecordCommands
 * RecordCommands --> SubmitQueue
 * SubmitQueue --> Present
 * Present --> AcquireImage : next frame
 * Present --> [*] : window closed
 * @enduml
 *
 * @return `true` if the frame was presented, `false` if the swap chain
 *         needs to be rebuilt (window resize or minimise).
 */
bool presentFrame();
```

### 4.5 `@copydoc` for Overridden Virtuals

When an overriding implementation has identical semantics to the base, avoid duplicating documentation:

```cpp
/**
 * @class VulkanDevice
 * @brief Vulkan implementation of the Device interface.
 */
class VulkanDevice : public Device {
public:
    /**
     * @copydoc Device::submit()
     * @note Vulkan implementation: submits to the graphics queue.
     */
    void submit(CommandBuffer& cmd) override;
};
```

### 4.6 Mainpage and Pages

```cpp
/**
 * @mainpage My Graphics Engine
 *
 * @tableofcontents
 *
 * @section intro Introduction
 *
 * Welcome to the API reference for **MyEngine**. This documentation covers
 * the complete public API as of version 3.0.
 *
 * @section quickstart Quick Start
 *
 * @code{.cpp}
 * gfx::Device device;
 * gfx::PipelineDesc desc = gfx::makeDefaultPipelineDesc();
 * gfx::RenderPipeline<MyVertex> pipeline(device, desc);
 * pipeline.bind(cmdBuffer);
 * pipeline.draw(mesh.indexCount);
 * @endcode
 *
 * @section modules Modules
 *
 * - @ref gfx_core — Pipeline and resource management
 * - @ref gfx_ui   — UI rendering primitives
 *
 * @section links See Also
 *
 * - [GitHub Repository](https://github.com/example/myengine)
 * - [Changelog](@ref changelog)
 */
```

### 4.7 Requirements Traceability

```cpp
/**
 * @requirement REQ-RENDER-001 Frame Rate Target
 * The renderer shall maintain a minimum of 60 FPS on the reference hardware
 * under the standard benchmark scene.
 */

/**
 * @brief Submits the frame for presentation.
 *
 * @satisfies REQ-RENDER-001 Direct implementation of the frame submission path.
 */
void presentFrame();

/**
 * @brief Benchmark test for frame submission throughput.
 *
 * @verifies REQ-RENDER-001 Measures sustained frame rate over 10,000 frames.
 * @test Run with the standard benchmark scene on the reference hardware.
 */
void test_frameRateBenchmark();
```

### 4.8 C Struct as OOP (Manual Inheritance)

```cpp
/**
 * @struct Animal
 * @brief Base "class" for all animals (C OOP pattern).
 */
struct Animal {
    const char* name; ///< Display name of the animal.
    int         age;  ///< Age in whole years.
};

/**
 * @brief Makes a sound appropriate for the animal.
 * @public @memberof Animal
 * @param[in] self  The animal instance.
 */
void Animal_speak(Animal* self);

/**
 * @struct Dog
 * @brief Canine specialisation of Animal.
 * @extends Animal
 */
struct Dog {
    Animal base; ///< @protected Base Animal data.
    int    boneCount; ///< Number of bones the dog has buried.
};

/**
 * @brief Fetches a thrown item.
 * @public @memberof Dog
 * @param[in]  self     The Dog instance.
 * @param[in]  itemName Name of the item to fetch.
 * @return `true` if the dog retrieved the item.
 */
bool Dog_fetch(Dog* self, const char* itemName);
```

---

## Quick Reference: Comment Style Cheat Sheet

```
// ─── Before a declaration ────────────────────────────────────────────────────

/** Brief one-liner. */
void simpleFunc();

/**
 * @brief Brief description.
 *
 * Detailed description spanning multiple paragraphs.
 *
 * @param[in]  x  First input.
 * @param[out] y  Receives the result.
 * @return  True on success.
 */
bool complexFunc(int x, int* y);

/// Short single-line description (no body needed).
constexpr int MAX_JOINTS = 256;

// ─── Inline / trailing (struct members, enum values) ─────────────────────────

struct Foo {
    int bar;  ///< Brief description of bar.
    int baz;  ///< Brief description of baz.
              /**< Multi-sentence trailing block when one line isn't enough.
               *   Use this style sparingly; prefer ///<. */
};

// ─── Sections inside a block ─────────────────────────────────────────────────

/**
 * @brief Does important work.
 * @details Extended explanation here.
 * @note Something the user should know.
 * @warning Do not call on a null pointer.
 * @attention Resource must be initialised first.
 * @pre x must be positive.
 * @post result is normalised.
 * @param[in] x  Input value.
 * @return Normalised result.
 * @throws std::range_error if x is zero.
 * @since 2.1
 * @deprecated Use newFunc() instead.
 * @todo Add SIMD fast path.
 * @bug Returns NaN for subnormal inputs.
 * @see relatedFunc()
 */
float importantFunc(float x);
```
