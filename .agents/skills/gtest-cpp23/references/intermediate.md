# GTest Intermediate — Fixtures, Lifecycle, and Organization

## Why Test Fixtures

`TEST()` works for isolated tests. When multiple tests share setup (creating objects, opening files, connecting to a test DB), duplicating that code in every test is verbose and error-prone. Test fixtures centralize shared setup and teardown.

## `TEST_F()` — Test Fixture Basics

A fixture is a class inheriting from `::testing::Test`. Tests use `TEST_F()` instead of `TEST()`:

```cpp
#include <gtest/gtest.h>
#include <vector>

class VectorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Called before each test
    vec = {10, 20, 30};
  }

  void TearDown() override {
    // Called after each test — clean up here
    vec.clear();
  }

  std::vector<int> vec;
};

TEST_F(VectorTest, InitialSizeIsThree) {
  EXPECT_EQ(vec.size(), 3);
}

TEST_F(VectorTest, FrontElementIsTen) {
  EXPECT_EQ(vec.front(), 10);
}

TEST_F(VectorTest, PushBackIncreasesSize) {
  vec.push_back(40);
  EXPECT_EQ(vec.size(), 4);
  EXPECT_EQ(vec.back(), 40);
}
```

Key points:
- The fixture class name becomes the test suite name.
- GTest creates a **fresh instance** of the fixture class for each test — tests are isolated.
- `SetUp()` runs before each test body. `TearDown()` runs after (even if the test fails).
- Fixture members are accessed directly in the test body via `this->` or just by name.

---

## Full Lifecycle Order

For a fixture with three tests, the execution order is:

```text
Fixture constructor
SetUp()
Test body 1
TearDown()
Fixture destructor

Fixture constructor   // fresh instance
SetUp()
Test body 2
TearDown()
Fixture destructor

Fixture constructor   // fresh instance
SetUp()
Test body 3
TearDown()
Fixture destructor
```

Each test gets its own fixture instance. No shared state between tests unless explicitly done via static members.

---

## Suite-Level Setup and Teardown

When setup is expensive (database connection pool, loading large files, starting an embedded server), use suite-level setup:

```cpp
class DatabaseTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Called ONCE before any test in this suite
    pool = create_connection_pool(10);
  }

  static void TearDownTestSuite() {
    // Called ONCE after all tests in this suite
    close_pool(pool);
  }

  void SetUp() override {
    // Called before each test — grab a connection from the pool
    conn = pool->acquire();
  }

  void TearDown() override {
    pool->release(conn);
  }

  static ConnectionPool* pool;
  Connection* conn;
};

ConnectionPool* DatabaseTest::pool = nullptr;
```

Lifecycle with `SetUpTestSuite`:
```text
SetUpTestSuite()           // once
  SetUp() / test / TearDown()  // per-test, repeated
  SetUp() / test / TearDown()
  SetUp() / test / TearDown()
TearDownTestSuite()        // once
```

---

## `SCOPED_TRACE` — Better Error Messages

When a helper function is called from multiple tests, a failure inside the helper shows the helper's line number — not helpful for finding which test case triggered it. `SCOPED_TRACE` adds context to the failure backtrace:

```cpp
void verify_element(const std::vector<int>& v, size_t idx, int expected) {
  SCOPED_TRACE(::testing::Message() << "idx=" << idx);
  ASSERT_LT(idx, v.size());  // If this fails, trace shows the idx
  EXPECT_EQ(v[idx], expected);
}

TEST_F(VectorTest, AllElementsCorrect) {
  verify_element(vec, 0, 10);
  verify_element(vec, 1, 20);
  verify_element(vec, 2, 30);
}
```

Failure output includes the trace:
```text
test.cpp:15: Failure
Expected: (idx) < (v.size()), actual: 3 vs 3
Google Test trace:
  test.cpp:28: idx=3
```

Multiple `SCOPED_TRACE` calls nest — each one adds a layer to the failure trace.

---

## Test Filtering with `--gtest_filter`

Select specific tests to run:

```sh
# Run all tests in VectorTest suite (direct binary invocation)
./build/cpu-test-debug/tests --gtest_filter=VectorTest.*

# Same selection through CTest (test names registered by gtest_discover_tests)
ctest --preset cpu-test-debug -R 'VectorTest.*'

# Run one specific test
./build/cpu-test-debug/tests --gtest_filter=VectorTest.FrontElementIsTen

# Run multiple patterns (colon-separated)
./build/cpu-test-debug/tests --gtest_filter=VectorTest.*:DatabaseTest.InsertWorks

# Exclude tests (leading minus)
./build/cpu-test-debug/tests --gtest_filter=VectorTest.*:-VectorTest.SlowTest
```

The pattern uses `*` for any characters and `?` for a single character.

---

## Test Organization Patterns

### Pattern 1: One Test File Per Source File

```text
src/
  math.cpp          -> tests/math_test.cpp
  parser.cpp        -> tests/parser_test.cpp
  network.cpp       -> tests/network_test.cpp
```

Each test file has its own fixture(s) and tests. This scales well for large projects.

### Pattern 2: Test Fixture Hierarchy

When fixtures share some setup, use inheritance:

```cpp
class BaseFixture : public ::testing::Test {
protected:
  void SetUp() override { /* common setup */ }
};

class NetworkTest : public BaseFixture {
protected:
  void SetUp() override {
    BaseFixture::SetUp();  // call base setup
    open_socket();
  }
  void TearDown() override {
    close_socket();
    BaseFixture::TearDown();
  }
};
```

### Pattern 3: Sharing Test Data Across Fixtures

For immutable test data used by multiple fixtures, use a shared header:

```cpp
// test_data.h
#pragma once
inline const std::vector<int> kSampleNumbers = {1, 2, 3, 5, 8, 13};
inline const std::string kLoremIpsum = "Lorem ipsum dolor sit amet...";
```

Then include it in whichever test files need it.

---

## `testing::AssertionResult` — Custom Assertions

For domain-specific checks, write custom assertions that produce readable failure messages:

```cpp
::testing::AssertionResult IsEven(int n) {
  if (n % 2 == 0)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << n << " is odd, expected even";
}

TEST(NumberTest, AllEven) {
  EXPECT_TRUE(IsEven(4));
  EXPECT_TRUE(IsEven(6));
}
```

Output on failure:
```text
test.cpp:12: Failure
Value of: IsEven(7)
  Actual: false (7 is odd, expected even)
```

---

## Implicit Conversions in Assertions

GTest assertions use `==` under the hood. Be aware of implicit conversions:

```cpp
EXPECT_EQ(42, 42.0);  // OK — int 42 == double 42.0
EXPECT_EQ('A', 65);   // OK — 'A' promoted to int 65
```

If types don't have `operator==`, the test won't compile. Use `EXPECT_TRUE(a == b)` as a workaround, or implement the missing operator.

---

## Disabling Tests

Prefix the test name with `DISABLED_`:

```cpp
TEST(FailingTest, DISABLED_NotReadyYet) {
  // This test is skipped
}
```

GTest reports disabled tests at the end of the run. Use `--gtest_also_run_disabled_tests` to run them anyway.

---

## Next Steps

Move to `references/advanced.md` for parameterized tests (`TEST_P`), type-parameterized tests (`TYPED_TEST`), death tests, and full GMock integration.
