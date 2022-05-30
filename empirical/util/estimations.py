import numpy as np
import pandas as pd


def estimate_width_ASK14(rupture_df: pd.DataFrame):
    """Estimate a width for ASK_14 model
    The equation is from NGA-West 2 spreadsheet
    rupture_df: pd.DataFrame
        Rupture Context(for OQ but in DF form) that contains
        site, distance and rupture information.
    """
    rupture_df["width"] = np.minimum(
        18 / np.sin(np.radians(rupture_df["dip"])),
        10 ** (-1.75 + 0.45 * rupture_df["mag"]),
    )

    return rupture_df
