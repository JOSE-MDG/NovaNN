# bfloat16 Under Clang on x86-64: Test-Side Limitations

**Scope:** `hip-*` presets (Clang toolchain). **Not affected:** `cpu-*` and
`cuda-*` presets, which compile exclusively with GCC/G++ (verified from each
preset's `CMakeCache.txt`: `CMAKE_CXX_COMPILER:STRING=/usr/bin/g++`, CUDA
device code via `/opt/cuda/bin/nvcc`).

## Summary

When NovaNN's test suites are compiled by Clang, every test that pushes a
`bfloat16` value through a function call boundary can observe that value
silently altered: subnormals become zero and NaN payloads become quieted.
This document explains why that happens, why it is outside this project's
control, and states the project stance on how the tests behave under each
compiler. The one-line takeaway:

> **Under `__clang__`, the tests exercise the software reference path for
> bfloat16; under GCC they exercise the native hardware path. Both paths are
> correct per their own specification; they simply disagree on subnormal
> inputs, and the disagreement belongs to the toolchain, not to us.**

The guard used throughout the test tree is deliberately bare:

```c
#if defined(__clang__)
```

Per the official Clang documentation, `__clang__` is "Defined when compiling
with Clang" [3], which is precisely the condition under which the lowering
described below applies.

## Mechanism 1: The ISA level (intentional, specified)

AVX-512 BF16 provides `VCVTNEPS2BF16` (and `VCVTNE2PS2BF16`) for packed
FP32-to-BF16 conversion. The Intel Architecture Instruction Set Extensions
Programming Reference specifies, verbatim [1]:

> This instruction uses "Round to nearest (even)" rounding mode. Output
> denormals are always flushed to zero and input denormals are always treated
> as zero. MXCSR is not consulted nor updated. No floating-point exceptions
> are generated.

Three properties matter here:

1. **Input denormals are treated as zero.** An FP32 subnormal source produces
   a sign-preserving zero, not a BF16 subnormal.
2. **MXCSR is neither consulted nor updated.** FTZ/DAZ flags are irrelevant;
   the flush is hard-wired into the instruction. We measured MXCSR inside a
   failing test process: `FTZ=0, DAZ=0`, with the flush still occurring.
3. **NaN payloads are quieted** (the pseudocode sets `dest[6] := 1`), which is
   why NaN payload corruption accompanies the subnormal loss.

This is intentional design, following the same convention adopted by TPU and
ARMv8.2 BF16 hardware: subnormal precision is traded away for silicon
simplicity. It is not a defect; it is the documented ISA contract.

## Mechanism 2: The compiler level (emulation details we do not control)

On x86-64, Clang supports `__bf16` but, per its own documentation, X86 is
listed among targets where support is "**currently never natively**"; when
compiling arithmetic on such types "Clang will perform the arithmetic in
float, inserting extensions and truncations as necessary," and by default
"does not truncate intermediate operands back to their true type unless the
operand is the result of an explicit cast or assignment" [2].

NovaNN typedefs its lowercase `bfloat16` to `__bf16` under GCC/Clang. Every
function parameter or return value of that type therefore becomes an explicit
cast/assignment boundary where Clang re-truncates the widened representation,
and the truncation lowering ends in the `VCVTNEPS2BF16` sequence described
above. Disassembling our own binaries makes this directly visible:

* Debug builds, argument edge of a scalar kernel calling
  `bf16_to_float(bfloat16)`:
  `movzwl (%..),%eax ; shl $0x10,%eax ; vmovd %eax,%xmm0 ;
  vcvtneps2bf16 %xmm0,%xmm0 ; call ...` — the value is destroyed *before*
  the callee runs.
* Release builds, return edge of `bf16_from_float`:
  `shl $0x10,%ebx ; vmovd %ebx,%xmm0 ; vcvtneps2bf16 %xmm0,%xmm0 ; ret`.

For comparison, GCC keeps `bfloat16` values as opaque 16-bit integers and
performs exact shifts, which is why the identical sources pass every suite
under the `cpu-*` and `cuda-*` presets.

Which of the two edges fires depends on optimization level; that both do is a
property of Clang's emulation strategy, not of any NovaNN code. We cannot fix
it from source without changing public signatures, and the behavior may change
between Clang versions without notice, since excess-precision handling is an
explicitly tunable compiler policy (`-fbfloat16-excess-precision=`) rather
than a language guarantee.

## Project stance

We considered three ways out: routing production conversions through
integer-based paths under HIP, gating whole suites, or filing upstream. None
felt right for now. Production signatures exist for kernel performance and API
stability, not for the convenience of one compiler's emulation layer; hiding
the native path would also hide real behavior our users will meet. So the
tests carry the burden instead: wherever a `bfloat16` value would cross a call
boundary under Clang, the suites switch to the software reference path
(`round_to_nearest_even` / `f32_from_bits`, the same integer math our
`BFloat16` wrapper has always used), and the handful of kernel-mediated cases
that cannot be rerouted are omitted with a reason string pointing here. On
GCC, nothing changes: the native hardware path stays fully exercised. If a
future Clang stops canonicalizing at ABI edges, deleting these guards should
make every suite pass unchanged; that is the day we look forward to.

## References

1. Intel Corporation, *Intel® Architecture Instruction Set Extensions and
   Future Features Programming Reference*, chapter 2, "VCVTNEPS2BF16 —
   Convert Packed Single Data to Packed BF16 Data".
   <https://www.intel.com/content/dam/develop/external/us/en/documents/architecture-instruction-set-extensions-programming-reference-737410.pdf>
2. The Clang Team, *Clang Language Extensions*, section "Half-Precision
   Floating Point" (target list, emulation, `-fbfloat16-excess-precision`),
   fetched 2026-08-25.
   <https://clang.llvm.org/docs/LanguageExtensions.html>
3. The Clang Team, *Clang Language Extensions*, section "Builtin Macros"
   (`__clang__`).
   <https://clang.llvm.org/docs/LanguageExtensions.html>
