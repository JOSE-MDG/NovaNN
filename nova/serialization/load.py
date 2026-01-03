import traceback
import pickle
import io
from nova.utils.logger import logger


def load(f: str | io.BufferedIOBase, *, weights_only: bool = True):
    try:
        res = None
        if isinstance(f, str):
            with open(f, "rb") as file:
                res = _load_from_file(file, weights_only=weights_only)
        else:
            res = _load_from_file(f, weights_only=weights_only)

        logger.info("✅ Successfully loaded")
        return res
    except Exception as e:
        lines = [line for line in traceback.format_exception(e)]
        logger.error("An error occurred during loading.")
        print(*lines)


def _load_from_file(file: str, weights_only: bool = True):
    from ._safe_load import SafeUnpickler

    if weights_only:
        return SafeUnpickler(file).load()
    else:
        return pickle.load(file)
