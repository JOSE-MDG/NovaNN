from __future__ import annotations
import pickle
import io
from pathlib import Path
from typing import Any

def save(
    obj: Any, f: str | Path | io.BufferedIOBase, protocol: int = pickle.HIGHEST_PROTOCOL
) -> None: ...
