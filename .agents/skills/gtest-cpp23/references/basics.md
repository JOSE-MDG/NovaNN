# GTest Basics — TEST() Macros and Assertions

## What GoogleTest Is

GoogleTest is a C++ testing framework. Tests are functions that exercise your code and check results using assertion macros. GTest discovers and runs all tests, reporting pass/fail for each.

## Anatomy of a Test — `TEST()`

The `TEST()` macro defines a single test:

```cpp
#include <gtest/gtest.h>

int add(int a, int b) { return a + b; }

TEST(MathTest, AdditionWorks) {
  EXPECT_EQ(add(2, 3), 5);
  EXPECT_EQ(add(-1, 1), 0);
  EXPECT_EQ(add(0, 0), 0);
}
```

- First argument: **test suite name** — groups related tests together. Use a descriptive name like `MathTest`, `ParserTest`, `NetworkTest`.
- Second argument: **test name** — describes what this specific test checks. Use descriptive names like `AdditionWorks`, `HandlesNegativeNumbers`, `RejectsEmptyInput`.
- Body: code with assertions. If all assertions pass, the test passes.

### Naming Convention

Use `PascalCase` for suite names and descriptive `snake_case` or `PascalCase` for test names:

```cpp
TEST(VectorTest, PushBackIncreasesSize) { ... }
TEST(VectorTest, PopBackOnEmptyThrows) { ... }
TEST(ParserTest, EmptyStringReturnsNullopt) { ... }
```

---

## Assertion Categories

### EXPECT vs ASSERT

- `EXPECT_*` — Non-fatal. Records the failure and continues executing the test.
- `ASSERT_*` — Fatal. Stops the test immediately on failure.

Use `ASSERT_*` when continuing would cause undefined behavior:

```cpp
TEST(PointerTest, DereferenceAfterNullCheck) {
  auto* ptr = get_optional_pointer();
  ASSERT_NE(ptr, nullptr);  // If null, stop — dereferencing below would crash
  EXPECT_EQ(ptr->value, 42);
}
```

### Comparison Assertions

```cpp
EXPECT_EQ(actual, expected);   // actual == expected
EXPECT_NE(actual, expected);   // actual != expected
EXPECT_LT(val, threshold);     // val < threshold
EXPECT_LE(val, threshold);     // val <= threshold
EXPECT_GT(val, threshold);     // val > threshold
EXPECT_GE(val, threshold);     // val >= threshold
```

These work with any type that has the corresponding operator defined. GTest prints both values on failure:

```cpp
// On failure, prints:
// Expected: 42
// Actual:   7
EXPECT_EQ(compute_answer(), 42);
```

### Boolean Assertions

```cpp
EXPECT_TRUE(condition);   // condition is true
EXPECT_FALSE(condition);  // condition is false
```

Prefer `EXPECT_EQ` over `EXPECT_TRUE(a == b)` — the error message is better because it shows both values.

### String Assertions (C Strings)

For `const char*` or `char[]`:

```cpp
EXPECT_STREQ(str1, str2);       // equal
EXPECT_STRNE(str1, str2);       // not equal
EXPECT_STRCASEEQ(str1, str2);   // equal ignoring case
EXPECT_STRCASENE(str1, str2);   // not equal ignoring case
```

For `std::string`, plain `EXPECT_EQ` works fine.

### Substring Matching

```cpp
EXPECT_THAT(big_string, ::testing::HasSubstr("needle"));
EXPECT_THAT(big_string, ::testing::StartsWith("prefix"));
EXPECT_THAT(big_string, ::testing::EndsWith("suffix"));
EXPECT_THAT(big_string, ::testing::MatchesRegex(R"(\d{3}-\d{4})"));
```

These require including `<gmock/gmock.h>` (the matchers live in GMock but work in regular `TEST()` macros).

### Floating Point Assertions

Never use `EXPECT_EQ` for floats/doubles — rounding errors cause spurious failures. Use:

```cpp
EXPECT_FLOAT_EQ(a, b);               // floats, ~4 ULPs tolerance
EXPECT_DOUBLE_EQ(a, b);              // doubles, ~4 ULPs tolerance
EXPECT_NEAR(a, b, absolute_error);   // custom absolute tolerance
```

Example:

```cpp
double result = compute_precise_value();
EXPECT_DOUBLE_EQ(result, 3.141592653589793);
EXPECT_NEAR(result, 3.14, 0.01);  // within 0.01
```

### Exception Assertions

```cpp
EXPECT_THROW(statement, exception_type);   // expects specific exception
EXPECT_NO_THROW(statement);                // expects no exception
EXPECT_ANY_THROW(statement);               // expects any exception
```

Example:

```cpp
TEST(ParserTest, EmptyInputThrows) {
  EXPECT_THROW(parse(""), std::invalid_argument);
}

TEST(ParserTest, ValidInputDoesNotThrow) {
  EXPECT_NO_THROW(parse("valid input"));
}
```

To also check the exception message:

```cpp
EXPECT_THROW(
  {
    try {
      parse("");
    } catch (const std::invalid_argument& e) {
      EXPECT_STREQ("input must not be empty", e.what());
      throw;  // re-throw so EXPECT_THROW sees it
    }
  },
  std::invalid_argument
);
```

---

## Complete Example: Compile and Run

### Source File: `test_math.cpp`

```cpp
#include <gtest/gtest.h>
#include <cmath>

double safe_sqrt(double x) {
  if (x < 0.0) throw std::domain_error("negative input");
  return std::sqrt(x);
}

TEST(SafeSqrtTest, PositiveInput) {
  EXPECT_DOUBLE_EQ(safe_sqrt(4.0), 2.0);
  EXPECT_DOUBLE_EQ(safe_sqrt(2.25), 1.5);
}

TEST(SafeSqrtTest, ZeroInput) {
  EXPECT_DOUBLE_EQ(safe_sqrt(0.0), 0.0);
}

TEST(SafeSqrtTest, NegativeThrows) {
  EXPECT_THROW(safe_sqrt(-1.0), std::domain_error);
}
```

### CMakeLists.txt

```cmake
add_executable(test_math test_math.cpp)
add_executable(test::math ALIAS test_math)

nova_configure_gtest_target(test_math)
nova_configure_build_flags(test_math)

if(BUILD_TESTING)
  gtest_discover_tests(test_math
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
```

### Build and Run

```bash
# configure/build every cpu preset
./scripts/build-presets.sh cpu-test-debug-linux
./scripts/compile-presets.sh cpu-test-debug-linux

# run the test binary directly for verbose output
./build/cpu-test-debug-linux/test_math --gtest_print_time=1
```

---

## Understanding Test Output

Passing output:
```text
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from SafeSqrtTest
[ RUN      ] SafeSqrtTest.PositiveInput
[       OK ] SafeSqrtTest.PositiveInput (0 ms)
[ RUN      ] SafeSqrtTest.ZeroInput
[       OK ] SafeSqrtTest.ZeroInput (0 ms)
[ RUN      ] SafeSqrtTest.NegativeThrows
[       OK ] SafeSqrtTest.NegativeThrows (0 ms)
[----------] 3 tests from SafeSqrtTest (0 ms total)
[==========] 3 tests from 1 test suite ran. (0 ms total)
[  PASSED  ] 3 tests.
```

Failing output includes the file, line, and both values:
```text
test_math.cpp:12: Failure
Expected: safe_sqrt(4.0)
      Which is: 2.0
To be equal to: 3.0
[  FAILED  ] SafeSqrtTest.PositiveInput (1 ms)
```

---

## Useful Command-Line Flags

| Flag | Effect |
|------|--------|
| `--gtest_filter=SuiteName.TestName` | Run only matching tests. `*` is wildcard: `--gtest_filter=SafeSqrt*` |
| `--gtest_repeat=N` | Repeat tests N times (find flaky tests) |
| `--gtest_shuffle` | Randomize test order |
| `--gtest_print_time=1` | Show per-test timing |
| `--gtest_list_tests` | List all tests without running them |
| `--gtest_output=xml:results.xml` | Output results as XML |

---

## Next Steps

Once comfortable with `TEST()` and assertions, move to `references/intermediate.md` to learn about test fixtures (`TEST_F`), `SetUp`/`TearDown`, and organizing larger test suites.
