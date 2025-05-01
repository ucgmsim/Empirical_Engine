from collections.abc import Callable, Sequence
from typing import Union

import pandas as pd
import numpy as np
from scipy import interpolate

from openquake.hazardlib import const as oq_const
from openquake.hazardlib import contexts, gsim, imt

from . import constants





def convert_im_label(im: imt.IMT):
    """Convert OQ's IM term into the internal term.
    E.g:
        pSA_period (OQ uses SA(period))
        Ds575 (OQ uses RSD575)
        Ds595 (OQ uses RSD595)
    im: imt.IMT
    """
    imt_tuple = imt.imt2tup(im.string)
    if len(imt_tuple) == 1:
        return (
            imt_tuple[0].replace("RSD", "Ds")
            if imt_tuple[0].startswith("RSD")
            else imt_tuple[0]
        )

    return f"pSA_{imt_tuple[-1]}"



