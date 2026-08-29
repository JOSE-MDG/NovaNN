# GTest Advanced — Parameterized Tests, Death Tests, and GMock

## Value-Parameterized Tests (`TEST_P`)

When the same test logic applies to many different inputs, parameterized tests eliminate duplication:

```cpp
#include <gtest/gtest.h>

bool is_prime(int n) {
  if (n < 2) return false;
  for (int i = 2; i * i <= n; ++i)
    if (n % i == 0) return false;
  return true;
}

class PrimeTest : public ::testing::TestWithParam<int> {};

TEST_P(PrimeTest, IsPrime) {
  EXPECT_TRUE(is_prime(GetParam()));
}

TEST_P(PrimeTest, IsNotEven) {
  if (GetParam() > 2) {
    EXPECT_NE(GetParam() % 2, 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
  SmallPrimes,
  PrimeTest,
  ::testing::Values(2, 3, 5, 7, 11, 13, 17, 19)
);
```

### Key Elements

1. **Fixture inherits from `TestWithParam<T>`** — `T` is the parameter type.
2. **`TEST_P()`** — Same as `TEST_F()` but parameterized.
3. **`GetParam()`** — Returns the current parameter value (type `T`).
4. **`INSTANTIATE_TEST_SUITE_P()`** — Generates the concrete tests:
   - First arg: prefix for the instantiation name (used in filtering).
   - Second arg: fixture class name.
   - Third arg: parameter generator.

### Parameter Generators

```cpp
// Explicit list
::testing::Values(1, 2, 3, 4)

// Range [start, end) with step
::testing::Range(0, 100, 10)    // 0, 10, 20, ..., 90

// Combine multiple generators (cartesian product)
::testing::Combine(
  ::testing::Values("http", "https"),
  ::testing::Values("example.com", "test.org")
)
// Produces: ("http","example.com"), ("http","test.org"),
//           ("https","example.com"), ("https","test.org")

// Values from a container
::testing::ValuesIn(my_vector)
::testing::ValuesIn(my_array)

// Bool (false, true)
::testing::Bool()
```

### Naming Instantiations

By default, GTest names tests like `PrimeTest/IsPrime/0`, `PrimeTest/IsPrime/1`, etc. Add a fourth argument for custom names:

```cpp
INSTANTIATE_TEST_SUITE_P(
  Primes,
  PrimeTest,
  ::testing::Values(2, 3, 5),
  [](const auto& info) {
    return std::to_string(info.param);  // Names: Primes/PrimeTest.IsPrime/2
  }
);
```

---

## Type-Parameterized Tests (`TYPED_TEST`)

When the same test logic applies to multiple types (e.g., all numeric types, all container types):

```cpp
#include <gtest/gtest.h>
#include <vector>
#include <list>
#include <deque>

// Define the type-parameterized fixture
template <typename T>
class SequenceContainerTest : public ::testing::Test {
protected:
  T container;
};

// Register the types to test
using MyTypes = ::testing::Types<
  std::vector<int>,
  std::list<int>,
  std::deque<int>
>;
TYPED_TEST_SUITE(SequenceContainerTest, MyTypes);

// Write tests — TYPED_TEST, not TYPED_TEST_P
TYPED_TEST(SequenceContainerTest, StartsEmpty) {
  EXPECT_TRUE(this->container.empty());
  EXPECT_EQ(this->container.size(), 0);
}

TYPED_TEST(SequenceContainerTest, PushBackAddsElement) {
  this->container.push_back(42);
  EXPECT_FALSE(this->container.empty());
  EXPECT_EQ(this->container.size(), 1);
  EXPECT_EQ(this->container.back(), 42);
}
```

GTest generates separate tests for each type:
- `SequenceContainerTest/0.StartsEmpty` (vector)
- `SequenceContainerTest/1.StartsEmpty` (list)
- `SequenceContainerTest/2.StartsEmpty` (deque)

### Combining With Value Parameterization

For a matrix of types × values, nest parameterization:

```cpp
template <typename T>
class NumericTest : public ::testing::TestWithParam<T> {};

TYPED_TEST_SUITE_P(NumericTest);

TYPED_TEST_P(NumericTest, NonZeroDivisionWorks) {
  TypeParam val = GetParam();
  EXPECT_NE(val, 0);
  EXPECT_EQ(val / val, 1);
}

REGISTER_TYPED_TEST_SUITE_P(NumericTest, NonZeroDivisionWorks);

// Instantiate for each type
using NumericTypes = ::testing::Types<int, double, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, NumericTest, NumericTypes);
```

This is an older, two-step registration pattern — prefer `TYPED_TEST_SUITE` with a type list for simplicity unless you need the flexibility.

---

## Death Tests

Death tests verify that code terminates (crashes, aborts, or exits) under expected conditions. Use them to test assertions, fatal errors, and security boundaries.

```cpp
#include <gtest/gtest.h>

void configure(int port) {
  if (port < 0 || port > 65535) {
    std::cerr << "Invalid port: " << port << std::endl;
    std::abort();
  }
}

TEST(DeathTest, NegativePortAborts) {
  EXPECT_DEATH(configure(-1), "Invalid port");
}

TEST(DeathTest, PortTooLargeAborts) {
  EXPECT_DEATH(configure(99999), "Invalid port");
}
```

### Death Test Macros

| Macro | What it checks |
|-------|---------------|
| `EXPECT_DEATH(stmt, regex)` | stmt terminates with a message matching regex |
| `EXPECT_DEATH_IF_SUPPORTED(stmt, regex)` | Same, but skips on platforms without death test support |
| `EXPECT_EXIT(stmt, predicate, regex)` | stmt calls `exit()` or `_exit()` with code matching predicate, message matching regex |
| `ASSERT_DEATH(stmt, regex)` | Fatal variant — stops test on failure |

### `EXPECT_EXIT` — Checking Exit Codes

```cpp
void exit_with_code(int code) { _exit(code); }

TEST(ExitTest, ExitsWithCode) {
  EXPECT_EXIT(exit_with_code(42),
              ::testing::ExitedWithCode(42),
              "");
}
```

### Death Test Styles

GTest offers two implementation styles, set via `--gtest_death_test_style`:

- **`fast`** (default on most platforms) — Uses `fork()` on Linux. Fast but doesn't survive crashes in the child process in all cases.
- **`threadsafe`** — Spawns a separate process. Slower but more robust, especially with threads.

Set in code:
```cpp
::testing::FLAGS_gtest_death_test_style = "threadsafe";
```

### Windows / clang-cl Notes

- Death tests work on Windows but use a different mechanism (process spawning, not fork).
- The `fast` style is not available on Windows — it defaults to `threadsafe`.
- Regex matching uses std::regex (MSVC's implementation), which has slight differences from POSIX regex — prefer simple substring patterns like `"Invalid port"` over complex regex.
- `EXPECT_EXIT` with `ExitedWithCode` is reliable; `EXPECT_DEATH` with signal-based termination (segfaults) may behave differently than on Linux.

---

## GMock Integration

GMock is part of GoogleTest (merged since v1.10). It lets you create mock objects for dependency isolation.

### Include

```cpp
#include <gmock/gmock.h>
// or just <gtest/gtest.h> if gmock_main is linked
```

### `MOCK_METHOD` — Declaring Mocks

```cpp
class DatabaseInterface {
public:
  virtual ~DatabaseInterface() = default;
  virtual bool connect(std::string_view host, int port) = 0;
  virtual std::string query(std::string_view sql) = 0;
  virtual int row_count() const = 0;
};

class MockDatabase : public DatabaseInterface {
public:
  MOCK_METHOD(bool, connect, (std::string_view host, int port), (override));
  MOCK_METHOD(std::string, query, (std::string_view sql), (override));
  MOCK_METHOD(int, row_count, (), (const, override));
};
```

`MOCK_METHOD` syntax:
```cpp
MOCK_METHOD(ReturnType, MethodName, (ArgTypes), (Qualifiers));
```

- Return type first, then method name.
- Argument types in parentheses.
- Qualifiers (`const`, `override`, `noexcept`) in the last parentheses.

### Setting Expectations — `EXPECT_CALL`

```cpp
#include <gmock/gmock.h>

TEST(DatabaseTest, QueryIsCalledWithCorrectSQL) {
  MockDatabase mock;

  // Expect connect() called once with specific args, return true
  EXPECT_CALL(mock, connect("localhost", 5432))
    .Times(1)
    .WillOnce(::testing::Return(true));

  // Expect query() called with this exact SQL, return a result
  EXPECT_CALL(mock, query("SELECT * FROM users"))
    .WillOnce(::testing::Return("user1,user2,user3"));

  // Also allow row_count() to be called any number of times
  EXPECT_CALL(mock, row_count())
    .WillRepeatedly(::testing::Return(3));

  // Exercise the mock
  ASSERT_TRUE(mock.connect("localhost", 5432));
  EXPECT_EQ(mock.query("SELECT * FROM users"), "user1,user2,user3");
  EXPECT_EQ(mock.row_count(), 3);
}
```

### Cardinality — `.Times()`

| Cardinality | Meaning |
|-------------|---------|
| `.Times(n)` | Called exactly n times |
| `.Times(AtLeast(n))` | Called n or more times |
| `.Times(AtMost(n))` | Called n or fewer times |
| `.Times(Between(m, n))` | Called between m and n times |
| `.Times(0)` | Must not be called |
| (omitted) | `.Times(1)` by default |

If the expectation is `WillRepeatedly`, default is `.Times(AtLeast(0))`.

### Actions — `.WillOnce()` / `.WillRepeatedly()`

| Action | Effect |
|--------|--------|
| `Return(value)` | Return `value` |
| `ReturnRef(ref)` | Return a reference |
| `Throw(exception)` | Throw exception |
| `Invoke(f)` | Call function `f(args...)` |
| `InvokeWithoutArgs(f)` | Call `f()` ignoring args |
| `DoAll(action1, action2)` | Execute multiple actions in sequence |
| `SetArgReferee<N>(value)` | Set the Nth argument (0-indexed) by reference |

Chain multiple `WillOnce` calls for sequential behavior:

```cpp
EXPECT_CALL(mock, get_value())
  .WillOnce(::testing::Return(1))
  .WillOnce(::testing::Return(2))
  .WillOnce(::testing::Return(3));
// Returns 1, then 2, then 3 on successive calls
```

### Matchers

Matchers validate arguments flexibly:

```cpp
using ::testing::_;           // Wildcard — matches anything
using ::testing::Eq;          // Equality (default)
using ::testing::Ne;          // Not equal
using ::testing::Gt;          // Greater than
using ::testing::Lt;          // Less than
using ::testing::Ge;          // Greater or equal
using ::testing::Le;          // Less or equal
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::HasSubstr;
using ::testing::StartsWith;
using ::testing::EndsWith;
using ::testing::Contains;    // Container contains element
using ::testing::ElementsAre; // Exact sequence match
using ::testing::UnorderedElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::AllOf;       // All matchers must match
using ::testing::AnyOf;       // Any matcher matches
using ::testing::Not;         // Negate matcher
```

Examples:

```cpp
// Wildcard: any connection params OK
EXPECT_CALL(mock, connect(_, _))
  .WillOnce(::testing::Return(true));

// Specific host, any port > 1024
EXPECT_CALL(mock, connect("db.example.com", ::testing::Gt(1024)))
  .WillOnce(::testing::Return(true));

// Multiple constraints
EXPECT_CALL(mock, query(::testing::AllOf(
  ::testing::HasSubstr("SELECT"),
  ::testing::Not(::testing::HasSubstr("DROP"))
)));

// Container matchers
EXPECT_CALL(mock, process(::testing::ElementsAre(1, 2, 3)));
EXPECT_CALL(mock, process(::testing::UnorderedElementsAre(3, 1, 2)));
```

### Mocking Non-Virtual Functions (Templates/CRTP)

For mocking templates or non-virtual functions, use the **high-perf** (formerly "Naggy") mock pattern with a mock interface:

```cpp
template <typename Db>
class UserService {
public:
  explicit UserService(Db& db) : db_(db) {}
  std::string get_user(int id) {
    return db_.query("SELECT name FROM users WHERE id=" + std::to_string(id));
  }
private:
  Db& db_;
};

// Testing: inject a mock that satisfies the implicit interface
struct MockDb {
  MOCK_METHOD(std::string, query, (std::string_view sql), ());
};

TEST(UserServiceTest, GetUserQueriesCorrectly) {
  MockDb mock;
  EXPECT_CALL(mock, query("SELECT name FROM users WHERE id=42"))
    .WillOnce(::testing::Return("Alice"));

  UserService<MockDb> service(mock);
  EXPECT_EQ(service.get_user(42), "Alice");
}
```

No virtual needed — the template binds at compile time.

### Multi-Expectation Ordering

By default, expectations on different methods are unordered. Use `InSequence` to enforce call order:

```cpp
TEST(OrderTest, MethodsCalledInSequence) {
  MockDatabase mock;

  {
    ::testing::InSequence seq;
    EXPECT_CALL(mock, connect(_, _)).WillOnce(::testing::Return(true));
    EXPECT_CALL(mock, query(_)).WillOnce(::testing::Return("data"));
    EXPECT_CALL(mock, row_count()).WillOnce(::testing::Return(1));
  }
  // If query() is called before connect(), the test fails
}
```

### Mocking `const` Methods

Use the `(const)` qualifier in `MOCK_METHOD`:

```cpp
class MockReader {
public:
  MOCK_METHOD(std::string, read_line, (), (const, override));
  MOCK_METHOD(bool, eof, (), (const, override));
};

TEST(ReaderTest, ReadLineOnConstRef) {
  const MockReader reader;
  EXPECT_CALL(reader, read_line()).WillOnce(::testing::Return("hello"));
  // Works — read_line is const-qualified
}
```

---

## C++23 Features in GTest

### `std::expected`

```cpp
#include <expected>

std::expected<int, std::string> parse_number(std::string_view sv) {
  if (sv.empty()) return std::unexpected("empty input");
  return std::stoi(std::string(sv));
}

TEST(ParseTest, ValidInput) {
  auto result = parse_number("42");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
  // Or:
  EXPECT_EQ(result.value(), 42);
}

TEST(ParseTest, InvalidInput) {
  auto result = parse_number("");
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), "empty input");
}
```

### `std::generator` for Test Data

```cpp
#include <generator>

std::generator<int> fibonacci(int n) {
  int a = 0, b = 1;
  for (int i = 0; i < n; ++i) {
    co_yield a;
    int next = a + b;
    a = b;
    b = next;
  }
}

TEST(FibonacciTest, FirstFiveValues) {
  auto gen = fibonacci(5);
  auto it = gen.begin();
  EXPECT_EQ(*it++, 0);
  EXPECT_EQ(*it++, 1);
  EXPECT_EQ(*it++, 1);
  EXPECT_EQ(*it++, 2);
  EXPECT_EQ(*it++, 3);
}
```

### `static operator()` in Lambdas

```cpp
TEST(LambdaTest, StaticLambda) {
  auto square = [](int x) static -> int { return x * x; };
  EXPECT_EQ(square(5), 25);
}
```

`static` ensures no capture overhead — useful in test utilities.

### `std::ranges::to<>` for Container Construction

```cpp
#include <ranges>
#include <algorithm>

std::vector<int> get_evens(const std::vector<int>& nums) {
  return nums
    | std::views::filter([](int n) { return n % 2 == 0; })
    | std::ranges::to<std::vector<int>>();
}

TEST(RangesTest, FiltersEvens) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6};
  auto result = get_evens(input);
  EXPECT_THAT(result, ::testing::ElementsAre(2, 4, 6));
}
```
