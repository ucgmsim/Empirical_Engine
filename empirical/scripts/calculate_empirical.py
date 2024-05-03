import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from qcore import constants, nhm, formats, utils
from empirical.util import empirical, z_model_calculations

RJB_MAX = 200
IM_LIST = [
    "PGA",
    "PGV",
    "pSA",
    "CAV",
    "Ds575",
    "Ds595",
]  # Other IMs: AI (CB_12 keyerror)

DEFAULT_COMP = constants.Components.crotd50.str_value

DEFAULT_MODEL_CONFIG_FFP = Path(__file__).parents[1] / "util" / "model_config.yaml"
DEFAULT_META_CONFIG_FFP = Path(__file__).parents[1] / "util" / "meta_config.yaml"
NZ_GMDB_SOURCE_PATH = Path(__file__).parents[1] / "data" / "earthquake_source_table.csv"
NHM_PATH = Path(__file__).parents[1] / "data" / "NZ_FLTmodel_2010_v18p6.txt"


def run_emp(
    output_dir: Path,
    ll_ffp: Path,
    vs30_ffp: Path,
    z_ffp: Path,
    srf_ffp: Path,
    nhm_ffp: Path = NHM_PATH,
    srfdata_ffp: Path = None,
    nz_gmdb_source_ffp: Path = NZ_GMDB_SOURCE_PATH,
    model_config_ffp: Path = DEFAULT_MODEL_CONFIG_FFP,
    meta_config_ffp: Path = DEFAULT_META_CONFIG_FFP,
    rjb_max: float = RJB_MAX,
    im_list: list[str] = IM_LIST,
    component: str = DEFAULT_COMP,
    periods=constants.DEFAULT_PSA_PERIODS,
    extended_period=False,
):
    """
    Run empirical calculations for a given historical event or a future event from a specified fault.

    Parameters
    ----------
    output_dir  : Path
        Path to the output directory
    ll_ffp  : Path
        lat lon file path for the stations
    vs30_ffp    : Path
        vs30 file path for vs30 values at the stations
    z_ffp   : Path
        z file path for z1.0 and z2.5 values at the stations
    srf_ffp : Path
        srf file path for the fault data.
    nhm_ffp : Path
        nhm file path for the collection of faults data (default: NZ_FLTmodel_2010_v18p6.txt)
    srfdata_ffp : Path
        srfdata file path for the fault data. Can be either realisation .csv or .info file
    nz_gmdb_source_ffp  : Path
        nz_gmdb_source file path for the source data
    model_config_ffp    : Path
        model_config file path for the empirical model. prescribes the model to be used for tectonic class, IM and component
    meta_config_ffp : Path
        meta_config file path for the empirical model. prescribes the weight of the model for IM and Tectonic class
    rjb_max : float
        Maximum rupture distance
    im_list : list
        List of intensity measures. Currently supported: PGA, PGV, pSA, CAV, Ds575, Ds595
    component   :str
        Component of the IM (eg. geom, rotd50) to calculate empirical for
    periods : list
        List of periods for pSA. Default is qcore.constants.DEFAULT_PSA_PERIODS
    extended_period : bool
        Indicate the use of extended(100) pSA periods

    """

    if extended_period:
        periods = constants.EXT_PERIOD

    ### Data loading
    if model_config_ffp.exists():
        model_config = utils.load_yaml(model_config_ffp)
    # Using Full Loader for the meta config due to the python tuple pSA/PGA
    meta_config = None
    if meta_config_ffp.exists():
        with open(meta_config_ffp) as f:
            meta_config = yaml.load(f, Loader=yaml.FullLoader)

    stations_df = formats.load_station_file(ll_ffp)
    vs30_df = formats.load_vs30_file(vs30_ffp)

    z_df = formats.load_z_file(z_ffp)
    z_df = z_df.rename(columns={"z1p0": "z1pt0", "z2p5": "z2pt5"})

    site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
    del stations_df, vs30_df, z_df

    nz_gmdb_source_df = pd.read_csv(
        nz_gmdb_source_ffp, index_col=0
    )  # Not useful if this is not a historical event, but we load it anyway

    if srfdata_ffp is None:
        # If srfdata_ffp is not supplied, but if srf_ffp is supplied and this is a historical event,we can still proceed
        print(f"INFO: srfdata_ffp not provided.")
        if srf_ffp is None:
            raise RuntimeError(f"Either srfdata_ffp or srf_ffp should be specified.")

        else:  # srf_ffp is provided. check if this is a valid historical event
            event = srf_ffp.stem
            event_name = event.split("_")[0]
            # Load source info. Only useful when this is a historical event
            try:
                fault_df = nz_gmdb_source_df.loc[
                    event_name, empirical.NZ_GMDB_SOURCE_COLUMNS
                ]
            except KeyError:
                print(f"Unknown event {event_name}")
                raise
            else:
                print(f"INFO: Found {event_name} in NZGMDB.")

    else:
        # srfdata_ffp is supplied. This can be either .csv or .info
        event = srfdata_ffp.stem
        fault_name = event.split("_")[0]
        if srfdata_ffp.suffix == ".info":
            fault_df = empirical.load_srf_info(srfdata_ffp, fault_name)
        else:  # .csv
            fault_df = empirical.load_rel_csv(srfdata_ffp, fault_name)
        # Either .csv or .info, fault_df has consistent columns defined in NZ_GMDB_SOURCE_COLUMNS

        if srf_ffp is None:
            # Even if srf_ffp is not supplied, if it is a valid fault, we can use NHM to get the fault data and proceed
            nhm_data = nhm.load_nhm(str(nhm_ffp))
            # Get fault data
            try:
                # We are reconstructing missing srf_ffp from nhm.
                srf_ffp = nhm_data[fault_name]
            except KeyError:
                print(f"ERROR: Unknown fault {fault_name}")
                raise
            else:
                print(f"INFO: Found {fault_name} in NHM.")

    # at this point, we have valid fault_df, and srf_ffp (can be either Path or NHMFault)

    tect_type = empirical.TECT_CLASS_MAPPING[fault_df.tect_class]

    # Get site source data. srf_ffp is either Path or NHM.
    rrup_df = empirical.get_site_source_data(srf_ffp, site_df[["lon", "lat"]].values)

    # Each model (determined by model_config, tect_type, im, component) has different set of required columns
    # Let's make oq_rupture_df from site_df, rrup_df and fault_df, and hopefully(!) it has all required columns
    oq_rupture_df = empirical.get_oq_rupture_df(
        site_df, rrup_df, fault_df, rjb_max=rjb_max
    )

    # In reality,some model specific columns may be still missing.
    # Such exception handling is done by openquake_wrapper_vectorized.oq_prerun_exception_handle()
    # If any column is missing, it will raise an error with detailed info.
    # You could add more exception rules there.

    empirical.create_emp_rel_csv(
        event,
        periods,
        oq_rupture_df,
        im_list,
        component,
        tect_type,
        model_config,
        meta_config,
        output_dir,
    )


def load_args():
    parser = argparse.ArgumentParser(
        description="Script to calculate IMs for empirical models."
        "Produces one .csv for each IM containing "
        "all specified sites."
    )

    parser.add_argument(
        "--ll_ffp",
        required=True,
        type=Path,
        help="Path to the .ll file",
    )
    parser.add_argument(
        "--vs30_ffp",
        required=True,
        type=Path,
        help="Path to the .vs30 file",
    )
    parser.add_argument(
        "--z_ffp",
        required=True,
        type=Path,
        help="Path to the .z file that contains Z1.0 and Z2.5. "
        "If you don't have this file, estimate from vs30 utilizing relations in z_model_calculations.py. (eg. chiou_young_08_calc_z1p0)"
        "The file should have columns: station, z1p0, z2p5, sigma",
    )

    parser.add_argument(
        "--srf_ffp",
        help="Path to the SRF file",
        type=Path,
    )

    parser.add_argument(
        "--nhm_ffp",
        help="Path to the NHM file",
        default=NHM_PATH,
        type=Path,
    )

    parser.add_argument(
        "--srfdata_ffp",
        help="Path to the SRF .info or .csv file",
        type=Path,
    )

    parser.add_argument(
        "-rm",
        "--max_rupture_distance",
        type=float,
        default=RJB_MAX,
        help="Only calculate empiricals for stations "
        "that are within X distance to rupture",
    )

    parser.add_argument(
        "--nz_gmdb_source_ffp", help="NZ GMDB source CSV", default=NZ_GMDB_SOURCE_PATH
    )

    parser.add_argument(
        "--model_config_ffp",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_FFP,
        help="Path to the model_config file. Found in Empirical util.",
    )

    parser.add_argument(
        "--meta_config_ffp",
        type=Path,
        default=DEFAULT_META_CONFIG_FFP,
        help="Path to the meta_config weight file. Found in Empirical util.",
    )
    parser.add_argument(
        "-e",
        "--extended_period",
        action="store_true",
        help="Indicate the use of extended(100) pSA periods",
    )
    parser.add_argument(
        "-p",
        "--periods",
        nargs="+",
        default=constants.DEFAULT_PSA_PERIODS,
        type=float,
        help='pSA period(s) separated by a " " space. eg: 0.02 0.05 0.1.',
    )
    parser.add_argument(
        "-m",
        "--im",
        nargs="+",
        default=IM_LIST,
        help='Intensity measure(s) separated by a " " space(if more than one). eg: PGV PGA CAV.',
    )
    #    # TODO: Put common argparse arguments between IM_calc and empirical in shared file
    parser.add_argument(
        "-comp",
        "--component",
        choices=list(constants.Components.iterate_str_values()),
        default=constants.Components.cgeom.str_value,
        help="The component you want to calculate. Available components are: [%(choices)s]. Default is %(default)",
    )

    parser.add_argument("output", type=Path, help="output directory")
    args = parser.parse_args()

    return args


def main():
    args = load_args()
    utils.setup_dir(args.output)
    run_emp(
        args.output,
        args.ll_ffp,
        args.vs30_ffp,
        args.z_ffp,
        args.srf_ffp,
        args.nhm_ffp,
        args.srfdata_ffp,
        args.nz_gmdb_source_ffp,
        args.model_config_ffp,
        args.meta_config_ffp,
        args.max_rupture_distance,
        args.im,
        args.component,
        args.periods,
        args.extended_period,
    )


if __name__ == "__main__":
    main()
