import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import yaml

from qcore import constants, nhm, formats, utils
from empirical.util.openquake_wrapper_vectorized import oq_run
from empirical.util.classdef import GMM


from empirical.util.empirical import (
    create_emp_df,
    get_site_source_data,
    load_srf_info,
    load_rel_csv,
    nhm_flt_to_df,
    TECT_CLASS_MAPPING,
)

RJB_MAX = 200
# IM_LIST = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "pSA"]
IM_LIST = ["PGA", "PGV", "pSA", "CAV"]
# IM_LIST = ["AI"] # CB_12 keyerror


DEFAULT_MODEL_CONFIG_FFP = Path(__file__).parents[1] / "util" / "model_config.yaml"
DEFAULT_META_CONFIG_FFP = Path(__file__).parents[1] / "util" / "meta_config.yaml"
NZ_GMDB_SOURCE_PATH = Path(__file__).parents[1] / "data" / "earthquake_source_table.csv"
NHM_PATH = Path(__file__).parents[1] / "data" / "NZ_FLTmodel_2010_v18p6.txt"

OQ_INPUT_COLUMNS = [
    "vs30",
    "rrup",
    "rjb",
    "z1pt0",
    "z2pt5",
    "mag",
    "rake",
    "dip",
    "vs30measured",
    "ztor",
    "zbot",
    "rx",
    "hypo_depth",
]

FAULT_DF_COLUMNS = [
    "mag",
    "tect_class",
    "z_tor",
    "z_bor",
    "rake",
    "dip",
    "depth",
]  # following NZ_GMDB_SOURCE column names


def read_model_dict(config=None):
    if config is None:
        config = DEFAULT_MODEL_CONFIG_FFP

    model_dict = yaml.safe_load(open(config))
    return model_dict


def estimate_z1p0(vs30):
    return (
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30**8.0 + 378.7**8.0)) / 1000.0
    )  # CY08 estimate in KM


def estimate_z2p5(z1p0=None, z1p5=None):
    if z1p5 is not None:
        return 0.636 + 1.549 * z1p5
    elif z1p0 is not None:
        return 0.519 + 3.595 * z1p0
    else:
        print("no z2p5 able to be estimated")
        exit()


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
    im_list=IM_LIST,
    component=None,
    periods=constants.DEFAULT_PSA_PERIODS,
    extended_period=False,
):

    ### Data loading
    model_config = utils.load_yaml(model_config_ffp)
    # Using Full Loader for the meta config due to the python tuple pSA/PGA
    meta_config = None
    if meta_config_ffp is not None:
        with open(meta_config_ffp) as f:
            meta_config = yaml.load(f, Loader=yaml.FullLoader)

    stations_df = formats.load_station_file(ll_ffp)
    vs30_df = formats.load_vs30_file(vs30_ffp)
    if z_ffp is not None and z_ffp.exists():
        z_df = formats.load_z_file(z_ffp)
        z_df = z_df.rename(columns={"z1p0": "z1pt0", "z2p5": "z2pt5"})
    else:
        z1p0_df = estimate_z1p0(vs30_df)
        z2p5_df = estimate_z2p5(z1p0_df)
        z_df = pd.concat(
            [
                z1p0_df.rename(columns={"vs30": "z1pt0"}),
                z2p5_df.rename(columns={"vs30": "z2pt5"}),
            ],
            axis=1,
        )

    site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
    del stations_df, vs30_df, z_df

    source_df = pd.read_csv(nz_gmdb_source_ffp, index_col=0)

    if srfdata_ffp is None:
        print(f"srfdata_ffp not provided.")
        if srf_ffp is None:
            print(f"Error: At least srf_ffp should be specified.")
            sys.exit()
        else:
            event = srf_ffp.stem
            event_name = event.split("_")[0]  # ie. fault_name for Cybershake
            # Load source info. Only useful when this is a historical event
            try:
                fault_df = source_df.loc[event_name, FAULT_DF_COLUMNS]
            except:
                print(f"Error: Unknown event {event_name}")
                sys.exit()
            else:
                print(f"Found {event_name} in NZGMDB.")

    else:
        event = srfdata_ffp.stem
        fault_name = event.split("_")[0]
        if srfdata_ffp.suffix == ".info":
            fault_df = load_srf_info(srfdata_ffp, fault_name)
        else:  # .csv
            fault_df = load_rel_csv(srfdata_ffp, fault_name)

        if srf_ffp is None:
            nhm_data = nhm.load_nhm(str(nhm_ffp))
            # Get fault data
            nhm_flt_info = nhm_data[fault_name]
            srf_ffp = nhm_flt_to_df(nhm_flt_info)  # use NHM instead of srf_ffp later

    # from now on, we have fault_df, and srf_ffp (can be either Path or NHM)

    tect_type = TECT_CLASS_MAPPING[fault_df.tect_class]

    locations_df = site_df.copy(True)

    rrup_df = get_site_source_data(srf_ffp, site_df[["lon", "lat"]].values)

    locations_df.loc[:, ["rrup", "rjb", "rx", "ry"]] = rrup_df.values

    # Enforce distance threshold
    locations_df = locations_df.loc[locations_df.rjb <= rjb_max]
    locations_df["site"] = locations_df.index.values
    locations_df["event"] = str(event)

    # Add event data
    locations_df.loc[
        :,
        [
            "mag",
            "tect_class",
            "ztor",
            "zbot",
            "rake",
            "dip",
            "hypo_depth",
        ],  # OQ_INPUT_COLUMNS corresponding FAULT_DF_COLUMNS
    ] = fault_df.values

    locations_df["vs30measured"] = False

    create_emp_df(
        event,
        periods,
        locations_df,
        im_list,
        component,
        tect_type,
        model_config,
        meta_config,
        output_dir,
        convert_mean=np.exp,
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
        # required=True,
        type=Path,
        help="Path to the .z file that contains Z1.0",
    )
    # parser.add_argument(
    #     "-r",
    #     "--rupture_distance",
    #     required=True,
    #     help="Path to the rupture distance csv file",
    # )

    parser.add_argument(
        "--srf_ffp",
        help="Path to the SRF file",
        # required=True,
        type=Path,
    )

    parser.add_argument(
        "--nhm_ffp",
        help="Path to the NHM file",
        default=NHM_PATH,
        # required=True,
        type=Path,
    )

    parser.add_argument(
        "--srfdata_ffp",
        help="Path to the SRF .info or .csv file",
        # required=False,
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
        #        default=DEFAULT_META_CONFIG_FFP,
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
        help="pSA period(s) separated by a " "space. eg: 0.02 0.05 0.1.",
    )
    parser.add_argument(
        "-m",
        "--im",
        nargs="+",
        default=IM_LIST,
        help="Intensity measure(s) separated by a "
        "space(if more than one). eg: PGV PGA CAV.",
    )
    #    # TODO: Put common argparse arguments between IM_calc and empirical in shared file
    parser.add_argument(
        "-comp",
        "--component",
        nargs="+",
        choices=list(constants.Components.iterate_str_values()),
        default=constants.Components.cgeom.str_value,
        help="The component(s) you want to calculate."
        " Available components are: [%(choices)s]. Default is %(default)s",
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
