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

### Option A: find_package

NovaNN gets GTest from vcpkg (declared in `vcpkg.json`) and detects it with `find_package(GTest CONFIG)` in `cmake/Detect/testing/DetectGTest.cmake`. The project enforces C23/C++23 with extensions disabled:

```cmake
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(my-tests CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTest REQUIRED CONFIG)

enable_testing()
add_executable(my_tests test_basics.cpp)
target_link_libraries(my_tests PRIVATE GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(my_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
```

### Option B: Inside NovaNN — `nova_configure_gtest_target` (Recommended)

Test targets in the project use the helper defined by [DetectGTest.cmake](../../../cmake/Detect/testing/DetectGTest.cmake), which links `GTest::gtest` + `GTest::gtest_main`. When GTest is absent (`NOVA_HAS_GTEST = 0`) the helper is a no-op, so the same CMakeLists works with or without tests:

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

### Option C: FetchContent (no vcpkg / system dependency)

For standalone projects that cannot rely on a package manager:

```cmake
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(my-tests CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(my_tests test_basics.cpp)
target_link_libraries(my_tests PRIVATE GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(my_tests)
```

### Preset Workflow (NovaNN)

All NovaNN workflows go through `CMakePresets.json` (version 6). Every preset maps to `build/<preset>`; test presets set `outputOnFailure` and fail when no tests are registered:

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 27 },
  "configurePresets": [
    {
      "name": "cpu-test-debug",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "BUILD_TESTING": "ON"
      }
    }
  ],
  "buildPresets": [
    { "name": "cpu-test-debug", "configurePreset": "cpu-test-debug" }
  ],
  "testPresets": [
    {
      "name": "cpu-test-debug",
      "configurePreset": "cpu-test-debug",
      "output": { "outputOnFailure": true },
      "execution": { "noTestsAction": "error" }
    }
  ]
}
```

```sh
# Configure, build, and test one preset
cmake --preset cpu-test-debug
cmake --build --preset cpu-test-debug
ctest --preset cpu-test-debug

# NovaNN convenience wrappers — configure / compile every matching preset
./scripts/build-presets.sh cpu
./scripts/compile-presets.sh cpu
```

---

## Platform-Specific Notes

### Linux — GCC / Clang

- Link with `-lpthread` (CMake handles this automatically with GTest targets).
- Sanitizers are enabled through the `USE_ASAN` / `USE_UBSAN` options, exposed as ready-made presets: `cpu-asan-test-debug`, `cpu-ubsan-test-release`, and the same combinations for `cuda-` and `hip-`. No need to pass `-fsanitize=...` by hand.
- GTest 1.15+ requires at minimum GCC 12 or Clang 16 for C++23 support. NovaNN itself requires GCC ≥ 15 or Clang ≥ 20.1 (enforced at configure time by `cmake/Utils/CheckCompilerVersion.cmake`).
- Both compilers are fully supported — no known issues.

### Windows — clang-cl

- Use the MSVC-compatible Clang frontend: `clang-cl.exe`.
- Link against the MSVC runtime — no `-lpthread` needed, Windows threading is automatic.
- When using FetchContent, ensure your generator is "Ninja".
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
