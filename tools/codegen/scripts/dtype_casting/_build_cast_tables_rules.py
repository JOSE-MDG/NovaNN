"""Build cast-table rules from the CPU dtype casting source of truth."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from tools.codegen.engine import PROJECT_ROOT, RULES_DIR

BUILD_PRIORITY = 0

if TYPE_CHECKING:
    from pathlib import Path

_SOURCE_PATH = (
    PROJECT_ROOT / "ncore" / "native" / "cpu" / "dtype" / "DTypeCasting.c"
)
RULES_PATH = RULES_DIR / "dtype_casting" / "cast_tables_rules.json"

FLOAT_TYPES = {
    "fp4e2m1",
    "fp8e4m3",
    "fp8e5m2",
    "fp16",
    "bf16",
    "f32",
    "f64",
}
INT_TYPES = {
    "s8",
    "s16",
    "s32",
    "s64",
    "u8",
    "u16",
    "u32",
    "u64",
}

FUNC_RE = re.compile(
    r"(?P<comment>/\*\*(?:.|\n)*?\*/)\s*"
    r"(?P<attribute>\[\[gnu::target\([^\n]*\)\]\]\s*)?"
    r"void\s+(?P<name>[A-Za-z0-9_]+)\s*\(",
    re.DOTALL,
)
TARGET_RE = re.compile(r'\[\[gnu::target\("([^"]*)"\)\]\]')

# Requires token -> SimdIsa bit names.
REQUIRES_TO_BITS = {
    "SSE4.2": ("SSE4_2",),
    "SSE4_2": ("SSE4_2",),
    "AVX/AVX2": ("AVX", "AVX2"),
    "AVX": ("AVX",),
    "AVX2": ("AVX2",),
    "F16C": ("F16C",),
    "FMA": ("FMA3",),
    "FMA3": ("FMA3",),
    "AVX512F": ("AVX512F",),
    "AVX512BW": ("AVX512BW",),
    "AVX512DQ": ("AVX512DQ",),
    "AVX512VL": ("AVX512VL",),
    "AVX512VNNI": ("AVX512VNNI",),
    "AVX512FP16": ("AVX512FP16",),
    "AVX512BF16": ("AVX512BF16",),
    "AVX10.1": ("AVX10_1",),
    "AVX10.2": ("AVX10_2",),
    "AMX-TILE": ("AMX",),
    "AMX-FP16": ("AMXFP16",),
    "AMX-BF16": ("AMXBF16",),
    "AMX-INT8": ("AMXINT8",),
}

# gnu::target spelling -> SimdIsa bit names.
TARGET_TO_BITS = {
    "sse4.2": ("SSE4_2",),
    "avx": ("AVX",),
    "avx2": ("AVX2",),
    "f16c": ("F16C",),
    "fma": ("FMA3",),
    "fma3": ("FMA3",),
    "avx512f": ("AVX512F",),
    "avx512bw": ("AVX512BW",),
    "avx512dq": ("AVX512DQ",),
    "avx512vl": ("AVX512VL",),
    "avx512vnni": ("AVX512VNNI",),
    "avx512fp16": ("AVX512FP16",),
    "avx512bf16": ("AVX512BF16",),
    "avx10.1": ("AVX10_1",),
    "avx10.2": ("AVX10_2",),
    "amx-tile": ("AMX",),
    "amx-fp16": ("AMXFP16",),
    "amx-bf16": ("AMXBF16",),
    "amx-int8": ("AMXINT8",),
}

# Bits whose instructions are VEX encoded: the compiler accepts the
# target alone, but the CPU still needs AVX state to execute them.
IMPLIED_BITS = {
    "AVX2": ("AVX",),
    "F16C": ("AVX",),
    "FMA3": ("AVX",),
}

BIT_ORDER = [
    "SSE4_2",
    "AVX",
    "AVX2",
    "AVX2Int8",
    "AVX2VNNI",
    "F16C",
    "VNNI",
    "FMA3",
    "AVX512F",
    "AVX512BW",
    "AVX512DQ",
    "AVX512VL",
    "AVX512VNNI",
    "AVX512FP16",
    "AVX512BF16",
    "AVX10_1",
    "AVX10_2",
    "AMX",
    "AMXFP16",
    "AMXBF16",
    "AMXINT8",
]


def _expand_bits(bits: list[str]) -> set[str]:
    """Add the bits implied by VEX/EVEX encoding."""
    expanded: set[str] = set()
    for bit in bits:
        expanded.add(bit)
        expanded.update(IMPLIED_BITS.get(bit, ()))
    return expanded


def _canonical_requires(requires: list[str], name: str) -> set[str]:
    """Canonicalize a Requires: list into SimdIsa bit names."""
    bits: list[str] = []
    for token in requires:
        key = token.strip().upper()
        if key not in REQUIRES_TO_BITS:
            raise ValueError(f"{name}: unknown Requires token {token!r}")
        bits.extend(REQUIRES_TO_BITS[key])
    return _expand_bits(bits)


def _canonical_target(target: str, name: str) -> set[str]:
    """Canonicalize a gnu::target list into SimdIsa bit names."""
    bits: list[str] = []
    for token in target.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in TARGET_TO_BITS:
            raise ValueError(f"{name}: unknown gnu::target token {token!r}")
        bits.extend(TARGET_TO_BITS[key])
    return _expand_bits(bits)


def _sorted_bits(bits: set[str]) -> list[str]:
    """Return bit names in mask order."""
    return sorted(bits, key=BIT_ORDER.index)


def _extract_requires(comment: str) -> list[str]:
    """Extract a comma-separated ISA list from a function doc comment."""
    lines = comment.splitlines()
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if "Requires:" not in line:
            continue

        text = line.split("Requires:", 1)[1].strip()
        parts = [text] if text else []
        for tail in lines[idx + 1 :]:
            stripped = tail.strip()
            if not stripped.startswith("*"):
                break
            if stripped.startswith("*/"):
                break

            body = stripped.lstrip("*").strip()
            if not body:
                break
            if body.startswith("@param") or body.startswith("@brief"):
                break
            parts.append(body)

        requires = " ".join(" ".join(parts).split())
        return [item.strip() for item in requires.split(",") if item.strip()]

    return []


def _parse_tables(source_text: str) -> list[dict[str, Any]]:
    """Parse DTypeCasting.c into cast-table rule objects."""
    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for match in FUNC_RE.finditer(source_text):
        name = match.group("name")
        comment = match.group("comment")
        attribute = match.group("attribute")

        if name.endswith("_scalar"):
            if attribute is not None:
                raise ValueError(f"{name}: scalar kernels take no target")
            base = name[:-7]
        else:
            scalar_candidates = [
                key for key in entries if name.startswith(f"{key}_")
            ]
            if not scalar_candidates:
                raise ValueError(f"cannot resolve scalar base for {name}")
            base = max(scalar_candidates, key=len)

        entry = entries.get(base)
        if entry is None:
            if "_to_" not in base:
                raise ValueError(f"unexpected cast base name: {base}")
            src, dst = base[1:].split("_to_", 1)
            entry = {
                "signature": f"lookup_{base}",
                "src_type_abbr": src,
                "dst_type_abbr": dst,
                "guard_required": False,
                "scalar_name": None,
                "variants": [],
            }
            entries[base] = entry
            order.append(base)

        if name.endswith("_scalar"):
            entry["scalar_name"] = name
            continue

        if attribute is None:
            raise ValueError(f"{name}: SIMD kernel is missing [[gnu::target]]")
        target_match = TARGET_RE.fullmatch(attribute.strip())
        if target_match is None:
            raise ValueError(f"{name}: cannot parse {attribute!r}")
        target = target_match.group(1)

        requires = _extract_requires(comment)
        requires_bits = _canonical_requires(requires, name)
        target_bits = _canonical_target(target, name)
        if requires_bits != target_bits:
            raise ValueError(
                f"{name}: Requires {sorted(requires_bits)} do not match "
                f'[[gnu::target("{target}")]] {sorted(target_bits)}'
            )

        entry["variants"].append({
            "name": name,
            "requires": requires,
            "isa": _sorted_bits(requires_bits),
            "is_scalar": False,
        })
        entry["guard_required"] = True

    tables: list[dict[str, Any]] = []
    for base in order:
        entry = entries[base]
        scalar_name = entry["scalar_name"]
        if scalar_name is None:
            raise ValueError(f"missing scalar implementation for {base}")

        variants = [
            *entry["variants"],
            {
                "name": scalar_name,
                "requires": [],
                "isa": [],
                "is_scalar": True,
            },
        ]
        tables.append({
            "signature": entry["signature"],
            "src_type_abbr": entry["src_type_abbr"],
            "dst_type_abbr": entry["dst_type_abbr"],
            "guard_required": entry["guard_required"],
            "variants": variants,
        })

    return tables


def build_rules() -> dict[str, Any]:
    """Build the JSON object consumed by the cast-table generator."""
    if not _SOURCE_PATH.is_file():
        raise FileNotFoundError(f"source file not found: {_SOURCE_PATH}")

    source_text = _SOURCE_PATH.read_text()
    tables = _parse_tables(source_text)
    return {
        "tables": tables,
    }


def write_rules(path: Path = RULES_PATH) -> None:
    """Write the cast-table rules JSON file.

    The file is rewritten only when its bytes change: the codegen runs
    during the CMake configure step, and touching the file on every
    configure keeps the source tree dirty and, on Windows, rewrites it
    with CRLF endings.
    """
    rules = build_rules()
    payload = (json.dumps(rules, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    path.write_bytes(payload)


def main() -> None:
    """Entry point: generate and write cast_tables_rules.json."""
    write_rules()


if __name__ == "__main__":
    main()
