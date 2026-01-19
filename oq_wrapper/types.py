"""Supporting types for oq_wrapper modules"""

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = float | npt.NDArray[np.floating] | pd.Series
