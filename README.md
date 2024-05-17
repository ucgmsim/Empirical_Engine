[![Build Status](https://quakecoresoft.canterbury.ac.nz/jenkins/job/Empirical_Engine/badge/icon?build=last:${params.ghprbActualCommit=master})](https://quakecoresoft.canterbury.ac.nz/jenkins/job/Empirical_Engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Empirical_Engine
Contains codes to calculate Empirical IMs using the openquake engine

## To calculate empirical IMs

calculate_empirical.py is the main script to calculate empirical IMs. 
It uses the openquake engine to calculate the IMs. The script takes the following parameters:

```
positional arguments:
  output                output directory

options:
  -h, --help            show this help message and exit
  --ll_ffp LL_FFP       Path to the .ll file
  --vs30_ffp VS30_FFP   Path to the .vs30 file
  --z_ffp Z_FFP         Path to the .z file that contains Z1.0 and Z2.5. If not available, estimate from vs30
                        utilizing relations in z_model_calculations.py. (eg. chiou_young_08_calc_z1p0). The file
                        should have columns: station, z1p0, z2p5, sigma
  --srf_ffp SRF_FFP     Path to the SRF file
  --nhm_ffp NHM_FFP     Path to the NHM file. If srf_ffp is not provided, this is used to get the fault data. Get
                        one from https://github.com/ucgmsim/Empirical_Engine/files/15256612/NZ_FLTmodel_2010_v18p6.txt
  --srfdata_ffp SRFDATA_FFP
                        Path to the SRF .info or .csv file
  -rm MAX_RUPTURE_DISTANCE, --max_rupture_distance MAX_RUPTURE_DISTANCE
                        Only calculate empiricals for stations that are within X distance to rupture
  --nz_gmdb_source_ffp NZ_GMDB_SOURCE_FFP
                        NZ GMDB source CSV. Required for historical events when srfdata is missing. Use
                        earthquake_source_table.csv contained in GMDB.zip from https://osf.io/q9yrg/?view_only=05337ba1ebc744fc96b9924de633ca0e 
  --model_config_ffp MODEL_CONFIG_FFP
                        Path to the model_config file. Found in Empirical util.
  --meta_config_ffp META_CONFIG_FFP
                        Path to the meta_config weight file. Found in Empirical util.
  -e, --extended_period
                        Indicate the use of extended(100) pSA periods
  -p PERIODS [PERIODS ...], --periods PERIODS [PERIODS ...]
                        pSA period(s) separated by a " " space. eg: 0.02 0.05 0.1.
  -m IM [IM ...], --im IM [IM ...]
                        Intensity measure(s) separated by a " " space(if more than one). eg: PGV PGA CAV.
  -comp {090,000,ver,H1,H2,geom,rotd50,rotd100,rotd100_50,norm,EAS}, --component {090,000,ver,H1,H2,geom,rotd50,rotd100,rotd100_50,norm,EAS}
                        The component you want to calculate.

```
This is designed to accommodate fairly flexible situations. In practice, we often find .srf, .info and some .csv files missing.
- If srf_ffp is not supplied, and it is a known fault in NHM, it will extract the srf data directly from NHM.
- srfdata_ffp can be either .csv or .info.
- If neither .csv nor .info is supplied, and if it is a historical event (found in NZ_GMDB) , it will find the event info from NZ_GMDB and carry on.

Data files can be downloaded from the following links:
- earthquake_source_table.csv can be found in the [GMDB.zip](https://osf.io/q9yrg/?view_only=05337ba1ebc744fc96b9924de633ca0e)
- NHM file : [NZ_FLTmodel_2010_v18p6.txt](https://github.com/ucgmsim/Empirical_Engine/files/15256612/NZ_FLTmodel_2010_v18p6.txt) (Credit: [GNS Science](https://www.gns.cri.nz/data-and-resources/2010-national-seismic-hazard-model/))

Note that  
- Z-values must be supplied. Z1.0 and Z2.5 can be estimated from vs30 using the relations in z_model_calculations.py.
- If the model_config_ffp is not supplied, it will use the default model_config found in Empirical util.
- If the meta_config_ffp is not supplied, it will use the default meta_config found in Empirical util.


Internally, it calls the function oq_run in util.openquake_wrapper_vectorised.py with the following parameters:

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
    
