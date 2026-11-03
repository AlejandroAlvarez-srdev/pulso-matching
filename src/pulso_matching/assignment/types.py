from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

CostMatrix = Sequence[Sequence[float | None]]
FloatMatrix = NDArray[np.float64]
Pairs = list[tuple[int, int]]
