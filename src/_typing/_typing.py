from typing import Tuple, Any, Callable, List, Optional
from src.module import Parameters

Shape = Tuple[int, ...]

InitFn = Callable[[Tuple[Shape]], Any]
ListOfParameters = List[Parameters]
InitFnArg = Callable[[Shape], Any]
IntOrPair = int | Tuple[int, int]
ActivAndParams = Tuple[Optional[str], Optional[float]]
