import numpy as np
import pandas as pd


def estimate_width_ASK14(dip: pd.Series, mag: pd.Series):
    """Estimate a width for ASK_14 model
    The equation is from NGA-West 2 spreadsheet
    dip: pd.Series
    mag: pd.Series
    """
    return np.minimum(18 / np.sin(np.radians(dip)), 10 ** (-1.75 + 0.45 * mag))
