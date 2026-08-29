---
name: cpp23-features
description: Reference for C++23 (ISO/IEC 14882:2024) language and library features — what each one replaces or simplifies, a short example, and current toolchain support across GCC, Clang, nvcc (CUDA) and hipcc/amdclang++ (ROCm/HIP). Use this skill whenever writing new C++ code, reviewing or auditing existing C++ code for modernization, deciding whether a construct should be upgraded to C++23 (e.g. duplicated const/non-const overloads -> deducing `this`, manual error-code-plus-value structs -> std::expected, chained std::optional null-checks -> monadic operations, raw C handle pointers -> std::out_ptr/std::inout_ptr, nested operator[] indexing -> multidimensional subscript operator), or producing a C++ modernization report. Also trigger when the user asks "is this in C++23", "what's the C++23 way to do X", or needs to validate a C++23 recommendation against the actual build toolchain (e.g. nvcc/hipcc device-code restrictions) before suggesting it.
license: MIT
---

# C++23 Features

A working reference for modernizing C++ code to C++23. Use it both to write new C++23 code and to evaluate whether existing pre-C++23 code should be flagged for modernization.

## How to use this skill

1. Check the **toolchain support** table below before recommending any C++23 feature, especially library features that depend on a fully C++23-conformant standard library (libstdc++/libc++), not just compiler language support.
2. Distinguish **device code** (CUDA `__device__`/`__global__`, HIP kernels) from **host code** when auditing: device code typically has a much smaller usable standard library even when the compiler accepts the language standard flag. A feature being language-legal in device code doesn't mean its library counterpart (e.g. `std::expected`, `std::stacktrace`) is available there.
3. When auditing code, distinguish mechanical rewrites (e.g. `contains()`, `std::to_underlying`) from real design changes (e.g. `std::expected` changes the function's return contract and how callers handle the result). Flag them separately in reports.

## Toolchain support (current as of mid-2026)

C++23 support is more mature than C23 across mainstream compilers, but library-level features (anything in `<expected>`, `<stacktrace>`, `<spanstream>`) generally lag behind language features.

| Toolchain | Status |
|---|---|
| **GCC** | Partial, experimental C++23 support since GCC 11 via `-std=c++2b`/`-std=c++23`. Coverage has matured significantly across GCC 12–15; verify specific library headers (`<expected>`, `<stacktrace>`) on older GCC versions. |
| **Clang** | Partial C++23 support added progressively from Clang 13 through Clang 18, selectable via `-std=c++23`. By Clang 18 most language features are in place; library support depends on the paired standard library (libstdc++ vs libc++). |
| **nvcc (CUDA)** | Historically partial/evolving C++23 support tied to the host compiler. As of CUDA 13.3, NVIDIA announced full C++23 integration in both `nvcc` and `nvrtc`. Older CUDA toolkits should be assumed to have host-compiler-dependent, incomplete C++23 support — check the specific CUDA version in use. Device-code standard library coverage is provided separately via `libcu++` (`<cuda/std/...>`), which backports recent C++ standard library features but does **not** mirror the full `std::` namespace in device code. |
| **hipcc / amdclang++ (ROCm/HIP)** | Both are Clang-based, so language-level C++23 support tracks upstream Clang. Device code has substantially reduced standard-library availability regardless of the `-std=` flag. An ongoing `libhipcxx` effort aims to backport more standard-library functionality to device code. |

**Practical rule for multi-backend projects:** prefer C++23 *language* features freely once the compiler floor is met, but treat C++23 *library* features (`std::expected`, `std::stacktrace`, `<spanstream>`, monadic `std::optional`) as host-code-only unless confirmed available in device code via `libcu++`/`libhipcxx` for the relevant backend.

## Language features

### `if consteval`
Branches based on whether the surrounding context is being constant-evaluated. **Replaces:** `if constexpr` combined with trait checks to detect constant evaluation context.
```c++
consteval int f(int i) { return i; }

constexpr int g(int i) {
  if consteval {
      return f(i);
  } else {
      return 42;
  }
}
```

### Deducing `this`
Explicit object member functions deduce the type and value category (lvalue/rvalue, const/non-const) of the object via its first parameter, prefixed with `this`. **Replaces:** duplicated const/non-const (and lvalue/rvalue) overloads of a member function.
```c++
// C++23: one function instead of two (or more) overloads
struct T {
  decltype(auto) operator[](this auto& self, std::size_t idx) {
    return self.mVector[idx];
  }
};

// Pre-C++23
struct T {
  value_t& operator[](std::size_t idx) { return mVector[idx]; }
  const value_t& operator[](std::size_t idx) const { return mVector[idx]; }
};
```
Relevant for any container-like type — collapses 2+ near-duplicate overloads into one templated function. Note: this is an API-surface change if the type is part of a public interface.

### Multidimensional subscript operator
`operator[]` can now take zero or more arguments instead of being limited to exactly one. **Replaces:** `operator()(z, y, x)` abused as a workaround for multi-index access, or chained `operator[]`.
```c++
template <typename T, std::size_t Z, std::size_t Y, std::size_t X>
struct Array3d {
  std::array<T, X * Y * Z> m{};
  T& operator[](std::size_t z, std::size_t y, std::size_t x) {
      return m[z * Y * X + y * X + x];
  }
};

Array3d<int, 4, 3, 2> v;
v[3, 2, 1] = 42;
```

### Increasing range-based `for` safety
Several previously-dangling-reference patterns are now well-defined — no source change required to benefit, but removes the need for "hoist the temporary into a named variable" defensive workarounds added pre-C++23.
```c++
for (auto e : getTmp().getRef())              // temporary's lifetime now extended correctly
for (auto e : getVector()[0])
for (auto valueElem : getMap()["key"])
for (auto e : get<0>(getTuple()))
for (auto e : getOptionalCollection().value())
for (char c : get<std::string>(getVariant()))
```

## Library features

### `<stacktrace>`
Portable stack trace. **Replaces:** platform-specific APIs (`backtrace()` on Linux, `CaptureStackBackTrace` on Windows) or third-party libraries.
```c++
#include <print>
#include <stacktrace>

int main() {
    std::println("{}", std::stacktrace::current());
}
// Output example:
//   0#  main at /app/example.cpp:5 [0x5ee42e3db747]
//   1#  <unknown> [0x76e76dc29d8f]
```

### `contains()` for strings and string views
**Replaces:** `find(...) != std::string::npos` idiom. Same semantics, clearer intent — safe to apply mechanically.
```c++
std::string{"foobarbaz"}.contains("bar"); // true
std::string{"foobarbaz"}.contains("bat"); // false
```

### `std::to_underlying`
**Replaces:** `static_cast<std::underlying_type_t<E>>(e)`.
```c++
enum class MyEnum : int { A = 1, B, C };
std::to_underlying(MyEnum::A); // 1
std::to_underlying(MyEnum::C); // 3
```

### `<spanstream>`
Non-owning, non-reallocating stream I/O over an existing buffer. **Replaces:** `strstream` (deprecated).
```c++
char input[] = "10 20 30";
std::ispanstream is{std::span<char>{input}};
int i;
is >> i; // 10
is >> i; // 20
is >> i; // 30
```
```c++
char output[30]{};
std::ospanstream os{std::span<char>{output}};
os << 10 << 20 << 30;
std::span<char> sp = os.span();
```

### `std::out_ptr` / `std::inout_ptr`
Safely bridges C APIs that write to a `T**` out-parameter with C++ smart pointers, updating the smart pointer when the temporary pointer-to-pointer goes out of scope (including under exceptions). **Replaces:** manual raw-pointer juggling at C API boundaries.
```c++
int c_api_create_handle(MyHandle** p_handle);
int c_api_recreate_handle(MyHandle** p_handle);

std::unique_ptr<MyHandle, resource_deleter> resource(nullptr);
int err = c_api_create_handle(std::out_ptr(resource));
// `resource` now owns the allocated handle.

std::shared_ptr<MyHandle> resource2(nullptr);
err = c_api_recreate_handle(std::inout_ptr(resource2), resource_deleter{});
```

### Monadic operations for `std::optional`
`and_then`, `transform`, and `or_else` compose `std::optional`-returning operations. **Replaces:** chains of manual `if (opt.has_value())` checks.
```c++
std::optional<double> stringToSqrtDouble(const std::string& input) {
  return parse_int(input)
    .and_then(ensure_non_negative)
    .transform([](int x) { return std::sqrt(x); })
    .or_else(default_value_or_empty);
}
```
This is a control-flow rewrite, not a one-liner substitution — call it out separately from cosmetic findings in a modernization report.

### `std::expected<T, E>`
Represents a value-or-error in a single type. `std::unexpected` constructs the error case. Supports the same monadic operations as `std::optional`. **Replaces:** out-parameter error codes or hand-rolled `{T value; bool ok; E error;}` structs.
```c++
enum class Error { ParseError, NegativeNumber };

std::expected<double, Error> stringToSqrtDouble(const std::string& input) {
    auto parsed = parse_int(input);
    if (!parsed) return parsed;
    if (*parsed < 0) return std::unexpected(Error::NegativeNumber);
    return std::sqrt(*parsed);
}
```
**Important:** this is an API design change when adopted at a function boundary — callers must handle the new return type. Flag distinctly from mechanical findings in a modernization report.

### `std::unreachable()`
**Replaces:** `__builtin_unreachable()` (GCC/Clang).
```c++
int convert(MyEnum e) {
    switch (e) {
        case MyEnum::A: return 0;
        case MyEnum::B: return 1;
        case MyEnum::C: return 2;
        default: std::unreachable();
    }
}
```

## Modernization-audit checklist

When evaluating a file for a C++23 migration report, for each candidate finding confirm:

- The construct genuinely predates C++23 — don't flag C++17/C++20 features that are already idiomatic (e.g. `std::span`, concepts, ranges, `<format>` are C++20, not pre-C++23 cruft).
- The C++23 replacement is supported by both the compiler **and** the standard library implementation in use for that file (host vs. device code matters — see toolchain table).
- Library-feature swaps (e.g. error codes -> `std::expected`) are design changes, not mechanical ones — note this distinctly from pure syntax simplifications (e.g. `contains()`, `std::to_underlying`).
- Any documentation/comments referencing the old pattern will go stale and need updating — flag once per finding without restating it elsewhere in the report.

## Source & license

Adapted from Anthony Calandra's [modern-cpp-features](https://github.com/AnthonyCalandra/modern-cpp-features) (MIT License), reorganized as a modernization-oriented reference and supplemented with current (mid-2026) toolchain support notes.
