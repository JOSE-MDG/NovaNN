import traceback
import pickle
import io
from nova.utils.logger import logger
from typing import Any


def save(obj: Any, f: str | io.BufferedIOBase, protocol: int = pickle.HIGHEST_PROTOCOL):

    try:
        if isinstance(f, str):
            with open(f, "wb") as file:
                pickle.dump(obj, file, protocol=protocol)
        else:
            pickle.dump(obj, f, protocol=protocol)

        logger.info("✅ Saved successfully")
    except Exception as e:
        lines = [line for line in traceback.format_exception(e)]
        logger.error("An error occurred during saving.")
        print(*lines)
