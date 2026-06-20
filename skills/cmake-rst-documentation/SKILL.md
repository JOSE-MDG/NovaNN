---
name: cmake-rst-documentation
description: 'Prompt and workflow for writing reStructuredText documentation in CMakeLists.txt and .cmake module files. Follows official CMake developer documentation standards for documenting modules, functions, and macros.'
---

### Instructions

This file provides instructions for writing reStructuredText (rST) documentation directly inside `CMakeLists.txt` and `.cmake` files using CMake bracket comments. This is the official CMake documentation style used in the CMake source tree itself. Documentation written this way can be processed by Sphinx via the `moderncmakedomain` package or by CMake's own `cmake --help-module` command.

### Workflow

**Follow these steps to document any CMake build file:**

1. **Read the file** to understand what it does. Identify all `function()`, `macro()`, `option()`, `set()` calls, and project-level logic.
2. **Map the dependency tree.** Check `include()`, `find_package()`, `add_subdirectory()`, and `target_link_libraries()` calls to understand how this file relates to the rest of the build system.
3. **Write the file-level documentation block** at the top (after any license header), describing the module's purpose, the variables it defines, and the targets it creates.
4. **Write per-function/macro documentation blocks** immediately before each `function()` or `macro()` definition.
5. **Verify formatting** by running `cmake --help-module <module-name>` if applicable, or by building Sphinx docs with `SPHINX_HTML=ON`.

---

## Bracket Comment Format

All documentation must be placed inside **CMake bracket comments** that start with `.rst:`. 71 of `=` may be used in the opening and closing brackets as long as they match.

```cmake
#[=======================================================================[.rst:
Module Name
-----------

This is the module documentation written in reStructuredText.
#]=======================================================================]
```

**Rules:**
- All rST comment blocks **must start with `#` in column 1** (no leading whitespace).
- Content on the line containing the closing bracket `#]` is excluded if and only if that line starts with `#`.
- Additional `.rst:` bracket comment blocks may appear **anywhere** in the file (typically just before each function/macro definition).

---

## File-Level Documentation Structure

Place the main documentation block **at the top of the file**, after any license header (if there is). The structure follows this order:

```cmake
#[=======================================================================[.rst:
MyModule
--------

This module provides functionality for X. It detects Y and
configures Z targets accordingly.

This module defines the following variables:

``MY_MODULE_FOUND``
  True if the module was found.

``MY_MODULE_VERSION``
  The version string of the detected module.

``MY_MODULE_INCLUDE_DIRS``
  Include directories for the module headers.

``MY_MODULE_LIBRARIES``
  Libraries to link against.

This module defines the following imported targets:

``MyModule::MyLibrary``
  The main library target.

This module defines the following functions and macros:

.. command:: my_module_configure_target

  Configures a target with module-specific settings:

  .. code-block:: cmake

    my_module_configure_target(<target> [OPTIONS <options>])

#]=======================================================================]
```

### Section Order

1. **Module name and underline** (using `-` character).
2. **Purpose description** — what the module does, in 2–4 sentences.
3. **Variables section** — list each variable using `` ``VAR_NAME``` ` with a description indented on the next line.
4. **Imported targets section** — list each target using `` ``Namespace::Target``` `.
5. **Functions/macros overview** — list each function using `.. command::` directive with a brief signature.

---

## Documenting Functions and Macros

Each `function()` or `macro()` should have its own `.rst:` block placed **immediately before** the definition. Use the `.. command::` directive from the CMake Domain.

```cmake
#[=======================================================================[.rst:
.. command:: my_module_add_library

  Creates and configures a library target for the MyModule project:

  .. code-block:: cmake

    my_module_add_library(
      <name>
      SOURCES <source1> [<source2> ...]
      [DEPENDENCIES <dep1> [<dep2> ...]]
      [OUTPUT_DIR <dir>]
    )

  This function creates a static or shared library (controlled by
  ``BUILD_SHARED_LIBS``) and applies standard compile options.

  The ``<name>`` argument specifies the target name.

  ``SOURCES``
    List of source files to compile.

  ``DEPENDENCIES``
    Optional list of target dependencies.

  ``OUTPUT_DIR``
    Optional output directory for the library artifact.

  The function sets the following target properties:

  ``MY_MODULE_VERSION``
    Version string passed to the target.

#]=======================================================================]
function(my_module_add_library NAME)
  cmake_parse_arguments(MY "" "OUTPUT_DIR" "SOURCES;DEPENDENCIES" ${ARGN})
  add_library(${NAME} ${MY_SOURCES})
  target_link_libraries(${NAME} PRIVATE ${MY_DEPENDENCIES})
endfunction()
```

### Documentation Conventions for Functions/Macros

| Element | Convention |
|---------|-----------|
| Directive | `.. command:: <function_name>` |
| Signature | Show in a `.. code-block:: cmake` inside the directive |
| Required args | `<angle brackets>` |
| Optional args | `[square brackets]` |
| Repeatable args | Trailing `...` |
| Parameter docs | Each on its own line, name in backticks, description indented below |
| Variable references | `` ``VARIABLE_NAME``` ` (inline literal) |
| Return values | Document at the end, list what the function sets/returns |

---

## Documenting CMakeLists.txt Files (Project Root)

For `CMakeLists.txt` files that define project structure rather than reusable functions, use the same bracket comment format but focus on:

1. **Project overview** — what the project builds.
2. **Options** — all `option()` calls with their defaults and descriptions.
3. **Subdirectories** — what each `add_subdirectory()` brings.
4. **Targets** — what targets are created and how they connect.

```cmake
cmake_minimum_required(VERSION 3.27)

#[=======================================================================[.rst:
NovaNN Project
--------------

This is the root CMakeLists.txt for the NovaNN project. It configures
the core build options and includes all subdirectories.

Project Options
^^^^^^^^^^^^^^^

``BUILD_SHARED_LIBS``
  Build shared libraries instead of static. Default: ``OFF``.

``USE_CUDA``
  Enable CUDA support. Default: ``ON``.

``USE_SANITIZERS``
  Enable AddressSanitizer and UndefinedBehaviorSanitizer. Default: ``OFF``.

Subdirectories
^^^^^^^^^^^^^^

The following subdirectories are included:

- ``src/`` — Core library sources.
- ``tests/`` — Unit and integration tests.
- ``cmake/Modules/`` — Custom CMake modules.
#]=======================================================================]

project(NovaNN
  VERSION 1.0.0
  LANGUAGES C CXX
  DESCRIPTION "Neural network inference library"
)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(USE_CUDA "Enable CUDA support" ON)
option(USE_SANITIZERS "Enable ASAN/UBSAN" OFF)

add_subdirectory(core)
add_subdirectory(tests)
```

---

## Documenting Find Modules (FindXxx.cmake)

For find modules, follow the CMake convention with a standardized structure:

```cmake
#[=======================================================================[.rst:
FindFoo
-------

Find the Foo library.

Imported Targets
^^^^^^^^^^^^^^^^

``Foo::Foo``
  The Foo library target.

Result Variables
^^^^^^^^^^^^^^^^

``Foo_FOUND``
  True if the library was found.

``Foo_INCLUDE_DIRS``
  Include directories for Foo headers.

``Foo_LIBRARIES``
  Libraries to link against.

``Foo_VERSION``
  The version of Foo found.

Hints
^^^^^

Set ``Foo_ROOT`` to the installation prefix to search first.
#]=======================================================================]

# ... find logic ...
```

---

## Cross-References and Inline Markup

Use reStructuredText inline markup consistently:

| What to reference | Syntax | Example |
|-------------------|--------|---------|
| CMake command | `` :command:`command_name` `` | ``:command:`add_library` `` |
| CMake variable | `` :variable:`VAR_NAME` `` | ``:variable:`CMAKE_CXX_STANDARD` `` |
| Target property | `` :prop_tgt:`PROP_NAME` `` | ``:prop_tgt:`AUTOMOC` `` |
| CMake policy | `` :policy:`CMP0000` `` | ``:policy:`CMP0077` `` |
| Generator expression | `` :genex:`$<CONDITION>` `` | ``:genex:`$<BUILD_INTERFACE:...>` `` |
| Code literal | ``` ``code`` ``` | ``` ``my_function()`` ``` |
| File/path literal | ``` ``path/to/file`` ``` | ``` ``CMakeLists.txt`` ``` |

---

## Style Rules

Based on the official CMake developer documentation standards:

### Section Headers

Use **only a line below** the title (no line above). Capitalize the first letter of each non-minor word.

```
Title Text
----------
```

**Underline character hierarchy:**

| Character | Use for |
|-----------|---------|
| `#` | Manual group (part) in the master document |
| `*` | Manual (chapter) title |
| `=` | Section within a manual |
| `-` | Subsection or CMake Domain object document title |
| `^` | Subsubsection or CMake Domain object document section |
| `"` | Paragraph or CMake Domain object document subsection |
| `~` | CMake Domain object document subsubsection |

### Whitespace

- **Two spaces** for indentation (not tabs).
- **Two spaces** between sentences in prose.
- Line width: **75–80 columns** (soft limit).

### Prose

- Use **American English** spellings.
- Use **imperative mood** in descriptions (e.g., "Create a target" not "Creates a target").

### Boolean Constants

- Use `ON`/`OFF` for user-modifiable values (options, cache variables).
- Use `TRUE`/`FALSE` for inherent values that cannot be changed after being set.

### Literal Blocks

- End a paragraph with `::` to start a literal block. The `::` is printed and the indented block follows.
- When using `.. code-block:: cmake`, end the preceding paragraph with `:` instead of `::`.

---

## Complete Documentation Workflow

When documenting any CMake file, follow this checklist:

1. **Analyze the file:**
   - Read the entire file.
   - Identify every `function()`, `macro()`, `option()`, `set()`, and `add_library()`/`add_executable()` call.
   - Trace `include()`, `find_package()`, and `add_subdirectory()` to understand the dependency tree.

2. **Write file-level block:**
   - Module/project name with underline.
   - Purpose (2–4 sentences).
   - Variables defined.
   - Imported targets created.
   - Functions/macros provided (list with `.. command::` directives).

3. **Write function/macro blocks:**
   - One `.rst:` block immediately before each `function()`/`macro()`.
   - `.. command::` directive with the function name.
   - `.. code-block:: cmake` with the signature.
   - Description of each parameter.
   - Description of return values / side effects.

4. **Verify:**
   - Run `cmake --help-module <module-name>` to check formatting.
   - Or build with `cmake -S Utilities/Sphinx -B build -DSPHINX_HTML=ON` and inspect output.

---

## Validation Checklist

- [ ] Every `.rst:` bracket comment starts with `#` in column 1.
- [ ] Opening and closing bracket `=` counts match.
- [ ] Module name has a proper underline (`---`).
- [ ] Section hierarchy uses correct underline characters.
- [ ] All variables are documented with `` ``VAR_NAME``` ` format.
- [ ] All functions/macros have `.. command::` directives before their definition.
- [ ] Function signatures use `.. code-block:: cmake` with proper placeholder syntax.
- [ ] Parameters use `<angle>` for required, `[square]` for optional, `...` for repeatable.
- [ ] Two-space indentation throughout.
- [ ] Line length is 75–80 columns or less.
- [ ] `ON`/`OFF` used for user-modifiable booleans, `TRUE`/`FALSE` for inherent values.
- [ ] Cross-references use `:command:`, `:variable:`, `:prop_tgt:` roles where applicable.

---

## Example: Full Module Documentation

```cmake
#[=======================================================================[.rst:
NovaNNBuildFlags
----------------

Centralized build flag configuration for the NovaNN project.

This module provides functions to apply compiler warnings, debug/release
flags, sanitizer options, and linker hardening flags to targets.

This module defines the following options:

``USE_ASAN``
  Enable AddressSanitizer. Default: ``OFF``.

``USE_UBSAN``
  Enable UndefinedBehaviorSanitizer. Default: ``OFF``.

``USE_LTO``
  Enable Link-Time Optimization. Default: ``ON``.

This module defines the following functions:

.. command:: nova_configure_build_flags

  Applies all build flags to the given target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

.. command:: nova_configure_linker

  Applies linker hardening flags to the given target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

#]=======================================================================]

option(USE_ASAN "Enable AddressSanitizer" OFF)
option(USE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(USE_LTO "Enable Link-Time Optimization" ON)

#[=======================================================================[.rst:
.. command:: nova_configure_build_flags

  Applies all build flags to the given target:

  .. code-block:: cmake

    nova_configure_build_flags(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function applies:

  - Warning flags from ``NOVA_WARNING_FLAGS``.
  - Debug or release flags from ``NOVA_DEBUG_FLAGS`` / ``NOVA_RELEASE_FLAGS``.
  - Sanitizer flags if ``USE_ASAN`` or ``USE_UBSAN`` are enabled.

#]=======================================================================]
function(nova_configure_build_flags TARGET)
  target_compile_options(${TARGET} PRIVATE
    $<$<CONFIG:Debug>:${NOVA_DEBUG_FLAGS}>
    $<$<CONFIG:Release>:${NOVA_RELEASE_FLAGS}>
    ${NOVA_WARNING_FLAGS}
  )
  if(USE_ASAN)
    target_compile_options(${TARGET} PRIVATE -fsanitize=address)
  endif()
  if(USE_UBSAN)
    target_compile_options(${TARGET} PRIVATE -fsanitize=undefined)
  endif()
endfunction()

#[=======================================================================[.rst:
.. command:: nova_configure_linker

  Applies linker hardening flags to the given target:

  .. code-block:: cmake

    nova_configure_linker(<target>)

  The ``<target>`` argument specifies the CMake target to configure.

  This function applies the following linker flags:

  - ``-Wl,-z,relro,-z,now`` for RELRO hardening.
  - ``-Wl,--as-needed`` to avoid unnecessary linking.
  - ``-Wl,--no-undefined`` to enforce symbol resolution.

#]=======================================================================]
function(nova_configure_linker TARGET)
  target_link_options(${TARGET} PRIVATE
    -Wl,-z,relro,-z,now
    -Wl,--as-needed
    -Wl,--no-undefined
  )
endfunction()
```
