def _single(input: int | tuple[int, int] | str) -> int:
    """
    Ensures the given input is a integer.

    Used internally by 1D operations (e.g., Conv1d, AvgPool1d) to handle
    parameters that can be provided as a pair or a tuple.

    Args:
        input: Integer, tuple of two integers or a string.

    Returns:
        A integer.

    Raises:
        ValueError: If a string like "same" is provided (not supported yet).
    """
    if isinstance(input, tuple):
        return input[0]

    elif isinstance(input, str):
        if input == "valid":
            return 0
        elif input == "same":
            raise ValueError(f"The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")

    return int(input)


def _pair(input: int | tuple[int, int]) -> tuple[int, int]:
    """
    Ensures the given input is a tuple of two integers.

    Used internally by 2D operations (e.g., Conv2d, AvgPool2d) to handle
    parameters that can be provided as a single int or a tuple.

    Args:
        input: Integer, tuple of two integers or a string.

    Returns:
        Tuple of two integers.

    Raises:
        ValueError: If a string like "same" is provided (not supported yet).
    """
    if isinstance(input, int):
        return (input, input)

    elif isinstance(input, str):
        if input == "valid":
            return (0, 0)
        elif input == "same":
            raise ValueError(f"The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")

    return tuple(input)


def _triple(input: int | tuple[int, int, int] | str) -> tuple[int, int, int]:
    """
    Ensures the given input is a tuple of three integers.

    Used internally by 3D operations (e.g., Conv3d, AvgPool3d) to handle
    parameters that can be provided as a single int or a tuple.

    Args:
        input: Integer, tuple of three integers or a string.

    Returns:
        Tuple of three integers.

    Raises:
        ValueError: If a string like "same" is provided (not supported yet).
    """
    if isinstance(input, int):
        return (input, input, input)

    elif isinstance(input, str):
        if input == "valid":
            return (0, 0, 0)
        elif input == "same":
            raise ValueError(f"The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")
    return tuple(input)
