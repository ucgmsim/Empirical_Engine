[![Build Status](https://quakecoresoft.canterbury.ac.nz/jenkins/job/Empirical_Engine/badge/icon?build=last:${params.ghprbActualCommit=master})](https://quakecoresoft.canterbury.ac.nz/jenkins/job/Empirical_Engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Empirical_Engine
Contains codes to calculate Empirical IMs using the openquake engine

## To calculate empirical IMs

Call the function oq_run in util.openquake_wrapper_vectorised.py with the following parameters:

    model_type: GMM
        OQ model
    tect_type: TectType
        One of the tectonic types from
        ACTIVE_SHALLOW, SUBDUCTION_SLAB and SUBDUCTION_INTERFACE
    rupture_df: Rupture DF
        Columns for properties. E.g., vs30, z1pt0, rrup, rjb, mag, rake, dip....
        Rows be the separate site-fault pairs
    im: string
        intensity measure
    periods: Sequence[Union[int, float]]
        for spectral acceleration, openquake tables automatically
        interpolate values between specified values, fails if outside range
    meta_config: Dict
        A dictionary contains models and its weight
    kwargs: pass extra (model specific) parameters to models
    