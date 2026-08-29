"""Build cast_funcs_rules.json from cast_tables_rules.json + dtype_casting_rules.json.

Derives the dispatch rules (ISA conditions, lookup tables, variant indices) needed
to generate the header-only cast.h file. The output JSON contains only dispatch
information: function names, lookup table references, guard requirements, and
pre-computed C condition expressions for each SIMD variant.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tools.codegen.engine import RULES_DIR

BUILD_PRIORITY = 1

if TYPE_CHECKING:
    from pathlib import Path


CAST_TABLES_RULES = RULES_DIR / "dtype_casting" / "cast_tables_rules.json"
DTYPE_CASTING_RULES = RULES_DIR / "dtype_casting" / "dtype_casting_rules.json"
OUTPUT_PATH = RULES_DIR / "dtype_casting" / "cast_funcs_rules.json"

FLOAT_TYPES = {"fp4e2m1", "fp8e4m3", "fp8e5m2", "fp16", "bf16", "f32", "f64"}
SINT_TYPES = {"s8", "s16", "s32", "s64"}
UINT_TYPES = {"u8", "u16", "u32", "u64"}

ISA_TO_SIMD = {
    "AVX512F": "SIMD->avx512f_",
    "AVX512FP16": "SIMD->avx512_fp16_",
    "AVX512BF16": "SIMD->avx512_bf16_",
    "AVX512BW": "SIMD->avx512_bw_",
    "AVX512VL": "SIMD->avx512_vl_",
    "AVX512DQ": "SIMD->avx512_dq_",
    "AVX": "SIMD->avx_",
    "AVX2": "SIMD->avx2_",
    "AVX/AVX2": "SIMD->avx_ || SIMD->avx2_",
    "F16C": "SIMD->f16c_",
    "SSE4.2": "SIMD->sse4_2_",
    "AVX10.2": "SIMD->avx10_2_",
}


def categorize_cast(src: str, dst: str) -> str:
    """Classify a cast pair into its dispatch category."""
    src_is_float = src in FLOAT_TYPES
    src_is_sint = src in SINT_TYPES
    src_is_uint = src in UINT_TYPES
    dst_is_float = dst in FLOAT_TYPES
    dst_is_sint = dst in SINT_TYPES
    dst_is_uint = dst in UINT_TYPES

    if src_is_float and dst_is_float:
        return "floating_to_floating"
    if src_is_float and dst_is_sint:
        return "floating_to_sinteger"
    if src_is_float and dst_is_uint:
        return "floating_to_uinteger"
    if src_is_sint and dst_is_float:
        return "sinteger_to_floating"
    if src_is_uint and dst_is_float:
        return "uinteger_to_floating"
    if src_is_sint and dst_is_sint:
        return "sinteger_to_sinteger"
    if src_is_sint and dst_is_uint:
        return "sinteger_to_uinteger"
    if src_is_uint and dst_is_sint:
        return "uinteger_to_sinteger"
    if src_is_uint and dst_is_uint:
        return "uinteger_to_uinteger"
    return "unknown"


def requires_to_condition(requires: list[str]) -> str:
    """Convert a list of ISA requirement strings into a C condition expression."""
    if not requires:
        return "1"
    conds = []
    for req in requires:
        if req in ISA_TO_SIMD:
            conds.append(f"({ISA_TO_SIMD[req]})")
        else:
            conds.append(f"/* UNKNOWN_ISA: {req} */ 0")
    return " && ".join(conds)


def build_rules() -> dict:
    """Build the complete cast_funcs_rules.json structure.

    Loads both input rule files, merges dispatch information from
    cast_tables_rules with scalar metadata from dtype_casting_rules, and
    produces a dict organized by cast category with pre-computed conditions.
    """
    with open(CAST_TABLES_RULES) as f:
        cast_tables = json.load(f)

    with open(DTYPE_CASTING_RULES) as f:
        dtype_casting = json.load(f)

    scalar_names = set()
    for cat_data in dtype_casting.get("funcs", {}).get("types", {}).values():
        for sig in cat_data.get("signatures", []):
            scalar_names.add(sig["name"])

    categorized: dict[str, list] = {}

    for table in cast_tables.get("tables", []):
        src = table["src_type_abbr"]
        dst = table["dst_type_abbr"]
        cat = categorize_cast(src, dst)

        cast_name = f"t{src}_to_{dst}"
        lookup = table["signature"]
        guard = table["guard_required"]

        variants = []
        for idx, v in enumerate(table["variants"]):
            is_scalar = v.get("is_scalar", False)
            vdata = {"name": v["name"], "idx": idx}
            if not is_scalar:
                reqs = v.get("requires", [])
                vdata["isa_required"] = reqs
                vdata["condition"] = requires_to_condition(reqs)
            else:
                vdata["is_scalar"] = True
            variants.append(vdata)

        scalar_only = not guard

        entry = {
            "name": cast_name,
            "scalar_only": scalar_only,
            "lookup_table": {
                "guard_required": guard,
                "name": lookup,
                "funcs": variants,
            },
        }
        categorized.setdefault(cat, []).append(entry)

    return {
        "rules": {
            "funcs": {
                "return": "void",
                "arguments": "const Tensor *restrict src, Tensor *restrict dst",
                "types": categorized,
            }
        }
    }


def write_rules(path: Path = OUTPUT_PATH) -> None:
    """Write the cast-funcs rules JSON file."""
    rules = build_rules()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, indent=2) + "\n")


def main() -> None:
    """Entry point: generate and write cast_funcs_rules.json."""
    write_rules()


if __name__ == "__main__":
    main()
