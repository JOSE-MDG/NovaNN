---
name: gtest-cpp23
description: Comprehensive reference for GoogleTest/GTest in C++23, compatible with GCC/Clang on Linux and clang-cl on Windows. Use this skill whenever the user mentions C++ testing, unit tests, GTest, GoogleTest, test fixtures, mocking with GMock, parameterized tests, death tests, or needs to write, debug, or understand C++ test code. Also trigger when users ask about CMake setup for C++ projects with testing, mention assertions like EXPECT_EQ or ASSERT_TRUE, or discuss test frameworks in C++. This skill covers everything from basic TEST() macros to advanced parameterized tests, type-parameterized tests, and GMock integration.
---

# GoogleTest in C++23 — Complete Reference

Comprehensive guide to GoogleTest/GTest for C++23 on GCC/Clang (Linux) and clang-cl (Windows). Covers basic assertions through GMock and parameterized tests. This file is the entry point — follow pointers to reference files for deep dives.

## Quick Navigation

| Need | Read |
|------|------|
| Basic `TEST()`, assertions, first test | `references/basics.md` |
| Fixtures, `SetUp`/`TearDown`, test organization | `references/intermediate.md` |
| `TEST_P`, `TYPED_TEST`, death tests, GMock | `references/advanced.md` |

Read the reference file that matches the user's need. If they're new to GTest, start with basics and work forward.

---

## Assertion Quick Reference

### Core Assertions

| Assertion | Meaning |
|-----------|---------|
| `EXPECT_EQ(a, b)` | a == b (non-fatal) |
| `ASSERT_EQ(a, b)` | a == b (fatal, stops test on failure) |
| `EXPECT_NE(a, b)` | a != b |
| `EXPECT_LT(a, b)` | a < b |
| `EXPECT_LE(a, b)` | a <= b |
| `EXPECT_GT(a, b)` | a > b |
| `EXPECT_GE(a, b)` | a >= b |

### Boolean & String

| Assertion | Meaning |
|-----------|---------|
| `EXPECT_TRUE(cond)` | cond is true |
| `EXPECT_FALSE(cond)` | cond is false |
| `EXPECT_STREQ(a, b)` | C strings equal |
| `EXPECT_STRNE(a, b)` | C strings not equal |
| `EXPECT_STRCASEEQ(a, b)` | C strings equal ignoring case |

### Floating Point

| Assertion | Meaning |
|-----------|---------|
| `EXPECT_FLOAT_EQ(a, b)` | floats approximately equal |
| `EXPECT_DOUBLE_EQ(a, b)` | doubles approximately equal |
| `EXPECT_NEAR(a, b, abs_err)` | within absolute error |

### Exceptions

| Assertion | Meaning |
|-----------|---------|
| `EXPECT_THROW(stmt, ex_type)` | statement throws ex_type |
| `EXPECT_NO_THROW(stmt)` | statement throws nothing |
| `EXPECT_ANY_THROW(stmt)` | statement throws anything |

**Rule**: Use `EXPECT_*` when you want the test to continue after failure. Use `ASSERT_*` when continuing makes no sense (e.g., pointer is null so dereferencing would crash).

---

## CMake Build Setup

### `nova_configure_gtest_target()`

Test targets in the project use the helper defined by [DetectGTest.cmake](../../../cmake/Detect/testing/DetectGTest.cmake), which links `GTest::gtest` + `GTest::gtest_main`.

```cmake
# ncore/tests/<area>/CMakeLists.txt — one executable per directory
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(my_tests ${TEST_SOURCES})
add_executable(my::tests ALIAS my_tests)

target_link_libraries(my_tests
    PRIVATE ncore::obj::core
    PRIVATE ncore::memory
)

nova_configure_gtest_target(my_tests)
nova_configure_build_flags(my_tests)

if(BUILD_TESTING)
  gtest_discover_tests(my_tests
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
```

### Presets Workflow

All NovaNN workflows go through `CMakePresets.json` (version 6). Every preset maps to `build/<preset>`.

#### **`Linux`** (All 36 presets):

```bash
# Use the --help option to view the options for the convenience wrappers
./scripts/build-presets.sh --help
./scripts/compile-presets.sh --help

# configure / compile every matching preset
./scripts/build-presets.sh cuda # Configure cuda-test-release-linux, cuda-test-debug-linux, etc ...
./scripts/compile-presets.sh cuda
```

#### **`Windows`** (Only 14 available presets):

```powershell
# Use the --help option to view the options for the convenience wrappers
.\scripts\build-presets.ps1 --help
.\scripts\compile-presets.ps1 --help

# configure / compile every matching preset
.\scripts\build-presets.ps1 cuda # Configure cuda-test-release-windows, cuda-test-debug-windows, etc ...
.\scripts\compile-presets.ps1 cuda
```
#### Output example **(Linux)**:
```text
╭─    ~/Projects/NovaNN  on   feat/tensor-core ⇡375 !3 ?23 
╰─ scripts/build-presets.sh cuda && scripts/compile-presets.sh -j $(nproc) cuda

NovaNN — configure presets
12 preset(s) → build/<preset>  ·  full logs: build/logs

  [ 1/12] ▸ cuda-release-linux
  ✔ configured  → build/cuda-release-linux  (15s)

  [ 2/12] ▸ cuda-debug-linux
  ✔ configured  → build/cuda-debug-linux  (14s)

  [ 3/12] ▸ cuda-asan-release-linux
  ✔ configured  → build/cuda-asan-release-linux  (14s)

  [ 4/12] ▸ cuda-asan-debug-linux
  ✔ configured  → build/cuda-asan-debug-linux  (14s)

  [ 5/12] ▸ cuda-ubsan-release-linux
  ✔ configured  → build/cuda-ubsan-release-linux  (15s)

  [ 6/12] ▸ cuda-ubsan-debug-linux
  ✔ configured  → build/cuda-ubsan-debug-linux  (14s)

  [ 7/12] ▸ cuda-test-release-linux
  ✔ configured  → build/cuda-test-release-linux  (14s)

  [ 8/12] ▸ cuda-test-debug-linux
  ✔ configured  → build/cuda-test-debug-linux  (14s)

  [ 9/12] ▸ cuda-asan-test-release-linux
  ✔ configured  → build/cuda-asan-test-release-linux  (14s)

  [10/12] ▸ cuda-asan-test-debug-linux
  ✔ configured  → build/cuda-asan-test-debug-linux  (15s)

  [11/12] ▸ cuda-ubsan-test-release-linux
  ✔ configured  → build/cuda-ubsan-test-release-linux  (14s)

  [12/12] ▸ cuda-ubsan-test-debug-linux
  ✔ configured  → build/cuda-ubsan-test-debug-linux  (14s)

✔ All 12 preset(s) configured successfully.

NovaNN — compile presets
12 preset(s) → cmake --build build/<preset>  ·  full logs: build/logs
parallel jobs: 24

  [ 1/12] ▸ cuda-release-linux  (Release)
  ✔ built  → build/cuda-release-linux  (7s)

  [ 2/12] ▸ cuda-debug-linux  (Debug)
  ✔ built  → build/cuda-debug-linux  (6s)

  [ 3/12] ▸ cuda-asan-release-linux  (Release)
  ✔ built  → build/cuda-asan-release-linux  (7s)

  [ 4/12] ▸ cuda-asan-debug-linux  (Debug)
  ✔ built  → build/cuda-asan-debug-linux  (5s)

  [ 5/12] ▸ cuda-ubsan-release-linux  (Release)
  ✔ built  → build/cuda-ubsan-release-linux  (6s)

  [ 6/12] ▸ cuda-ubsan-debug-linux  (Debug)
  ✔ built  → build/cuda-ubsan-debug-linux  (6s)

  [ 7/12] ▸ cuda-test-release-linux  (Release)
  ✔ built  → build/cuda-test-release-linux  (8s)

  [ 8/12] ▸ cuda-test-debug-linux  (Debug)
  ✔ built  → build/cuda-test-debug-linux  (7s)

  [ 9/12] ▸ cuda-asan-test-release-linux  (Release)
  ✔ built  → build/cuda-asan-test-release-linux  (7s)

  [10/12] ▸ cuda-asan-test-debug-linux  (Debug)
  ✔ built  → build/cuda-asan-test-debug-linux  (7s)

  [11/12] ▸ cuda-ubsan-test-release-linux  (Release)
  ✔ built  → build/cuda-ubsan-test-release-linux  (9s)

  [12/12] ▸ cuda-ubsan-test-debug-linux  (Debug)
  ✔ built  → build/cuda-ubsan-test-debug-linux  (7s)

✔ All 12 preset(s) built successfully.
╭─    ~/Projects/NovaNN  on   feat/tensor-core ⇡375 !3 ?23 
╰─
```
---

## Platform-Specific Notes

### Linux — GCC / Clang

- Sanitizers are enabled through the `USE_ASAN` / `USE_UBSAN` options, exposed as ready-made presets: `cpu-asan-test-debug`, `cpu-ubsan-test-release`, and the same combinations for `cuda-` and `hip-`.
- Both compilers are fully supported.

### Windows — clang-cl

- Use the MSVC-compatible Clang frontend: `clang-cl.exe`.
- Link against the MSVC runtime.
- Death tests (`EXPECT_DEATH`) behave differently on Windows due to process creation model. See `references/advanced.md` for details.

---

## C++23 Features That Work Well With GTest

GTest itself doesn't require C++23, but these C++23 features enhance test code:

- **`std::expected`** — Use `EXPECT_TRUE(result.has_value())` or `EXPECT_EQ(result.error(), some_error)`.
- **`std::format` / `std::print`** — Use in `SCOPED_TRACE` or custom failure messages for richer diagnostics.
- **`std::generator`** — Generate test input sequences lazily.
- **Deducing `this`** — Simplify CRTP-like test helper patterns.
- **`std::ranges` additions** — `std::ranges::to<>` for collecting into containers in test setup.
- **`static operator()`** — Captureless lambdas in tests can be `static`, avoiding unnecessary closure objects.
> _See [`skills/cpp23-features/SKILL.md`](../cpp23-features/SKILL.md) for more details_
---

## Reading Order

1. **`references/basics.md`** — First stop. Covers `TEST()` macro, all assertion types with examples, compiling and running tests, and understanding test output.
2. **`references/intermediate.md`** — Test fixtures (`TEST_F`), `SetUp`/`TearDown` lifecycle, `SetUpTestSuite`/`TearDownTestSuite` for suite-level setup, `SCOPED_TRACE`, test filtering with `--gtest_filter`, and test organization patterns.
3. **`references/advanced.md`** — Value-parameterized tests (`TEST_P`, `INSTANTIATE_TEST_SUITE_P`), type-parameterized tests (`TYPED_TEST`, `TYPED_TEST_SUITE`), death tests (`EXPECT_DEATH`, `EXPECT_EXIT`), and full GMock integration (`MOCK_METHOD`, `EXPECT_CALL`, matchers, actions, multi-expectation ordering).

Open the relevant reference file and follow the examples.
