from __future__ import annotations
import io
from pathlib import Path
from typing import Any

def load(f: str | Path | io.BufferedIOBase, *, weights_only: bool = True) -> Any: ...
def _load_from_file(file: io.BufferedIOBase, weights_only: bool = True) -> Any: ...
