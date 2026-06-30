---
name: c23-features
description: Reference for C23 (ISO/IEC 9899:2024) language and library features compared to C17/C11/C99 — what each feature replaces, a short example, and current toolchain support across GCC, Clang, nvcc (CUDA) and hipcc/amdclang++ (ROCm/HIP). Use this skill whenever writing new C code, reviewing or auditing existing C code for modernization, deciding whether a specific construct should be upgraded to C23 (e.g. NULL -> nullptr, repeated #define constants -> constexpr, vendor __attribute__ -> [[...]] attributes, manual type duplication -> auto, _Generic dispatch -> typeof), or producing a C modernization report. Also trigger when the user asks "is this available in C23", "what's the C23 equivalent of X", "can I use _BitInt/#embed/nullptr on this compiler", or needs to validate a C23 recommendation against the actual build toolchain (e.g. nvcc host compiler, cross-compilation targets) before suggesting it.
license: MIT
---

# C23 Features

A working reference for modernizing C code to C23. Use it both to write new C23 code and to evaluate whether existing pre-C23 code should be flagged for modernization.

## How to use this skill

1. Check the **toolchain support** table below before recommending any C23 feature — a feature being "in the standard" does not mean it's usable on every compiler/target in a given project. This matters most for projects that cross-compile for multiple backends (e.g. a CPU path via GCC/Clang and a GPU path via nvcc or hipcc).
2. Use the **feature reference** to identify what a given C23 feature replaces and whether it's a drop-in change or a semantic change worth flagging in review.
3. When auditing code (not just writing it), a finding is only valid if: (a) the construct has a real C23 replacement, (b) the replacement is supported by every compiler the file is actually built with, and (c) the change doesn't alter observable behavior unless that's the explicit goal (e.g. `nullptr` is not convertible to integer types, unlike `NULL` — that's a real semantic difference, not just syntax).

## Toolchain support (current as of mid-2026)

C23 support is uneven and still evolving — verify against the actual compiler/flags a target uses before relying on a feature.

| Toolchain | Status |
|---|---|
| **GCC** | C23 is the default dialect since GCC 15 (`gnu23`); explicit selection via `-std=c23` (or `-std=gnu23` for GNU extensions). Substantial C23 support since GCC 14, including `#embed`. Older GCC versions only expose it experimentally via `-std=c2x`. |
| **Clang** | Partial C23 support via `-std=c23`. Coverage trails GCC on some library/runtime pieces; check per-feature before depending on something niche (e.g. decimal floating-point, `_BitInt` corner cases). |
| **nvcc (CUDA, host code)** | Host-code C dialect follows the host compiler (GCC/Clang) used by nvcc, subject to nvcc's own host-compiler version policy. Recent CUDA toolkits track modern host compilers reasonably well, but always confirm the actual host compiler version nvcc is invoking — nvcc rejects host compiler versions outside its supported range outright. |
| **hipcc / amdclang++ (ROCm/HIP)** | Both are Clang-based, so C/C++ standard support generally tracks upstream Clang. Device code has reduced standard-library availability independent of the language standard selected — a feature being "language-supported" in C23 doesn't guarantee its library counterpart works in device code. |

**Practical rule for multi-backend projects:** if a source file is compiled identically for CPU, CUDA, and HIP paths, the safe C23 subset is the intersection of GCC/Clang language support — not the union. Features absent or partial on Clang (the common denominator for both Clang-proper and hipcc/amdclang++) should be flagged as "needs a fallback" rather than unconditionally recommended.

## Language features

### `auto`
Deduces the type of a variable from its initializer.
```c
auto f = 123.0f; // deduced to `float`

#include <tgmath.h>
auto c = cos(x); // deduced depending on the type of `x`
```
Unlike C++, C23's `auto` only applies to object definitions — it cannot be used to infer function return or parameter types.

### `constexpr`
Declares a typed, scoped compile-time constant. **Replaces:** `#define` constants, `enum` tricks for compile-time values.
```c
constexpr size_t cache_line_size_bytes = 64;
```
Restricted to scalars: no functions, structs, unions, or arrays, and no pointer/atomic/volatile-qualified types.

### Decimal floating-point types
`_Decimal32`, `_Decimal64`, `_Decimal128` provide IEEE-754 base-10 floating-point semantics, avoiding the representation error of binary floats for decimal-sensitive arithmetic.
```c
_Decimal32 decsum = 0.0df;
for (int i = 0; i < 10; i++)
    decsum += 0.1df;
```
Compiler support is still limited — check before relying on this.

### Bit-precise integers — `_BitInt(N)`
Declares an exact-`N`-bit signed or unsigned integer type. **Replaces:** hand-rolled bitfields/masking for non-standard integer widths.
```c
_BitInt(4) sbi;          // 4-bit signed
unsigned _BitInt(4) ubi; // 4-bit unsigned
```

### Binary literals
**Replaces:** hex/manual-shift workarounds to express base-2 values.
```c
0b110       // == 6
0b1111'1111 // == 255, `'` as a digit separator
```

### `char8_t` and UTF-8 character literals
`char8_t` is an unsigned type for 8-bit-wide characters; a `u8`-prefixed character literal is of type `char8_t`. **Replaces:** plain `unsigned char` for UTF-8.
```c
char8_t x = u8'x';
```

### Unicode string literals
`u8`, `u`, and `U` prefixes produce UTF-8, UTF-16, and UTF-32 string literals respectively.

### Empty initializer `{}`
Default-initializes an object: pointers to `NULL`/`nullptr`, arithmetic types to zero, decimal floats to positive zero, aggregates member-by-member, unions via their first named member. **Replaces:** the `= {0}` idiom. Arrays of unknown size cannot use it.
```c
char c = {};   // == 0
struct { int x; int y; } s = {}; // == { x: 0, y: 0 }
int ia[5] = {};                  // == [0, 0, 0, 0, 0]
```

### Attributes
Standardizes a `[[...]]` syntax. **Replaces:** `__attribute__(...)`, `__declspec`, and similar vendor extensions.
```c
[[noreturn]] void f(void) { exit(0); }
```
Standard attributes: `[[deprecated("reason")]]`, `[[fallthrough]]`, `[[nodiscard("reason")]]`, `[[maybe_unused]]`, `[[noreturn]]`. Compiler-specific directives (e.g. `[[clang::no_sanitize]]`) remain available alongside the standard set.

### New keywords
`true`, `false`, `thread_local`, and `static_assert` are now keywords rather than macros from `<stdbool.h>`/`<threads.h>`/`<assert.h>`. `static_assert` also no longer requires the message string argument.

### `nullptr`
A dedicated null-pointer constant of type `nullptr_t`, implicitly convertible to pointer types and `bool` — but, unlike `NULL`, **not** convertible to integral types. **Replaces:** `NULL`.
```c
void foo(int);
foo(NULL);    // valid
foo(nullptr); // error — this is a real semantic difference, not cosmetic
```

### `#embed`
Preprocessor directive to embed binary or text resources directly into source. **Replaces:** external tools (`xxd`, `objcopy`, custom scripts) that convert resources into C byte arrays.
```c
const uint8_t image_bytes[] = {
#embed "image.bmp"
};
```
Supports `prefix`, `suffix`, and `if_empty` parameters.

### Enums with underlying type
**Replaces:** compiler-dependent enum storage sizing.
```c
enum e : unsigned short
{
    x // `x` is an `unsigned short`
};
```

### `typeof` / `typeof_unqual`
Gets the type of an expression (similar to C++'s `decltype`). **Replaces:** the GCC/Clang `__typeof__` extension. `typeof_unqual` strips cv-qualifiers.
```c
int a;
const volatile int b;
typeof(a) c;        // int
typeof_unqual(b) d;  // int
```

### Improved compatibility for tagged types
Tagged types (`struct`/`union`/`enum`) with the same tag name and content are now compatible both across and within translation units, and redeclaration of an identical tagged type is allowed. **Replaces:** macro-generated struct definitions duplicated across headers as a workaround.
```c
#define PRODUCT(A, B) struct prod { A a; B b; }

void foo(PRODUCT(int, float) x) { /* ... */ }
void bar(PRODUCT(int, float) y) { foo(y); } // compiles: compatible type
```

## Library features

### Floating-point formatting — `strfromf` / `strfromd` / `strfroml`
Direct float-to-string conversion for `float`, `double`, `long double`. **Replaces:** `snprintf(..., "%f", ...)` workarounds.
```c
char buf[BUFFER_SIZE] = {};
strfromf(buf, BUFFER_SIZE, "%f", 123.0f);
```

### `fscanf`/`fprintf` format specifiers
- `%wN`, `%wfN` modifiers for `uintN_t`, `intN_t`, `uint_fastN_t`, `int_fastN_t`.
- `H`, `D`, `DD` modifiers for `_Decimal32`/`_Decimal64`/`_Decimal128`.
- `b`, `B` specifiers for unsigned binary output.
```c
uint64_t num = 1234;
printf("%w64u\n", num);
printf("%w64b\n", num);
```

### `memset_explicit`
Like `memset`, but guaranteed not to be optimized away by the compiler — use this for zeroing sensitive memory (keys, secrets, passwords).
```c
char str[] = "foo";
memset_explicit(str, 0, sizeof(str));
```

### `unreachable()` macro
Standardizes what was previously compiler-specific. **Replaces:** `__builtin_unreachable()` (GCC/Clang).
```c
if (1 > 0) { /* ... */ }
else unreachable();
```

### `memccpy`
Copies bytes until either a terminating byte value is found (and copied) or a byte limit is reached. **Replaces:** manual copy loops with a terminator check.
```c
char dest[MAX_LEN] = {};
memccpy(dest, src, 0, MAX_LEN - 1);
```

### `strdup` / `strndup`
Allocate-and-copy string duplication, now standardized rather than POSIX/compiler extensions. Caller must `free()` the result.
```c
char* a = strdup("foobarbaz");      // "foobarbaz"
char* b = strndup("foobarbaz", 3);  // "foo"
free(a); free(b);
```

### `gmtime_r` / `localtime_r`
Thread-safe variants of `gmtime`/`localtime`, writing into a caller-provided buffer instead of a static internal one. Returns `NULL`/`nullptr` on error.
```c
time_t t = time(NULL);
struct tm buf;
struct tm* ret = gmtime_r(&t, &buf);
```

### `timespec_getres`
Stores the resolution of the time source for a given base; returns zero on failure.
```c
struct timespec ts;
if (timespec_getres(&ts, TIME_UTC) == TIME_UTC) { /* ... */ }
```

## C11 baseline (for context)

C23 builds on C11. If a codebase predates C11 idioms entirely, these are worth flagging too — they're not C23-specific, but the same kind of "should be modernized" finding:

- **`_Generic`** — type-based dispatch at compile time.
- **`alignof`/`_Alignof`** and **`alignas`/`_Alignas`** — query/set alignment instead of compiler pragmas.
- **`static_assert`/`_Static_assert`** — compile-time assertions.
- **`noreturn`/`_Noreturn`** — superseded by `[[noreturn]]` in C23.
- **Anonymous structs/unions**.
- **`<stdatomic.h>`** — atomic types/flags/variables with explicit memory ordering, replacing hand-rolled spinlocks.
- **`<threads.h>`** — OS-agnostic threads/mutexes/condition variables. Historically poorly supported — verify before depending on it over a platform-native threading API.
- **Bounds-checked functions (`_s` suffix)** — e.g. `fopen_s`, `gets_s`.
- **`aligned_alloc`**, **`char16_t`/`char32_t`**, wide string literals (`u"..."`, `U"..."`), **`timespec_get`**, **quick exiting** (`quick_exit`/`at_quick_exit`), **exclusive-mode file opening** (`"wx"`/`"w+x"`).

## Modernization-audit checklist

When evaluating a file for a C23 migration report, for each candidate finding confirm:

- The current construct genuinely predates C23 (don't flag C11/C17 features that are already fine — e.g. `_Generic`, `_Static_assert`, `<stdatomic.h>` are C11, not pre-C23 cruft).
- The C23 replacement is supported on every compiler this file is actually built with (see toolchain table above).
- Any documentation/comments referencing the old construct (e.g. mentions of `NULL`, a removed macro, a now-implicit conversion) will go stale and need updating — flag this once per finding without restating it redundantly elsewhere in the report.
- The change is either behavior-preserving, or any behavior change (like `nullptr`'s stricter typing) is called out explicitly rather than presented as "just a rename."

## Source & license

Adapted from Anthony Calandra's [modern-c-features](https://github.com/AnthonyCalandra/modern-c-features) (MIT License), reorganized as a modernization-oriented reference and supplemented with current (mid-2026) toolchain support notes.
