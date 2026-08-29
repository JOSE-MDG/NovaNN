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
    r"(?:\[\[gnu::target\([^\n]*\)\]\]\s*)?"
    r"void\s+(?P<name>[A-Za-z0-9_]+)\s*\(",
    re.DOTALL,
)


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

        requires = " ".join(parts).replace("  ", " ").strip()
        return [item.strip() for item in requires.split(",") if item.strip()]

    return []


def _parse_tables(source_text: str) -> list[dict[str, Any]]:
    """Parse DTypeCasting.c into cast-table rule objects."""
    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for match in FUNC_RE.finditer(source_text):
        name = match.group("name")
        comment = match.group("comment")

        if name.endswith("_scalar"):
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

        requires = _extract_requires(comment)
        entry["variants"].append({
            "name": name,
            "requires": requires,
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
        "source": str(_SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "tables": tables,
    }


def write_rules(path: Path = RULES_PATH) -> None:
    """Write the cast-table rules JSON file."""
    rules = build_rules()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, indent=2) + "\n")


def main() -> None:
    """Entry point: generate and write cast_tables_rules.json."""
    write_rules()


if __name__ == "__main__":
    main()
