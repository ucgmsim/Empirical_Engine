import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import yaml

from qcore import srf, constants
from qcore.utils import setup_dir
from empirical.util.openquake_wrapper_vectorized import oq_run
from empirical.util.classdef import TectType, GMM
from IM_calculation.source_site_dist import src_site_dist

RJB_MAX = 200
# IM_LIST = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "pSA"]
IM_LIST = ["PGA", "PGV", "pSA", "CAV"]
# IM_LIST = ["AI"] # CB_12 keyerror


DEFAULT_GMM_CONFIG_NAME = Path(__file__).parents[1] / "util" / "model_config.yaml"
NZ_GMDB_SOURCE_PATH = Path(__file__).parents[1] / "data" / "earthquake_source_table.csv"

TECT_CLASS_MAPPING = {
    "Crustal": TectType.ACTIVE_SHALLOW,
    "Slab": TectType.SUBDUCTION_SLAB,
    "Interface": TectType.SUBDUCTION_INTERFACE,
    "Undetermined": TectType.ACTIVE_SHALLOW,
}

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


def get_tect_type_name(tect_type):
    found = None
    for key, val in TECT_CLASS_MAPPING.items():
        if tect_type == val:
            found = key
            break
    return found


def read_model_dict(config=None):
    if config is None:
        config = DEFAULT_GMM_CONFIG_NAME

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


def load_srf_info(srf_info, event_name):
    """Create fault parameters"""
    fault = {}
    f = h5py.File(srf_info, "r")
    attrs = f.attrs

    fault["mag"] = np.max(attrs["mag"])

    if "tect_type" in attrs:
        try:
            tect_type = TectType[attrs["tect_type"]]  # ok if attrs['tect_type'] is str
        except KeyError:  # bytes
            tect_type = TectType[attrs["tect_type"].decode("utf-8")]

    else:
        print("tect_type not found assuming 'ACTIVE_SHALLOW'")
        tect_type = TectType.ACTIVE_SHALLOW
    fault["tect_class"] = get_tect_type_name(tect_type)

    if "dtop" in attrs:
        fault["z_tor"] = np.min(attrs["dtop"])
    else:
        fault["z_tor"] = attrs["hdepth"]

    if "dbottom" in attrs:
        fault["z_bor"] = np.min(attrs["dbottom"])

    else:
        fault["z_bor"] = attrs["hdepth"]

    rake = attrs["rake"]
    if np.max(rake) == np.min(rake):
        fault["rake"] = np.min(rake)
    else:
        print("unexpected rake value")
        exit()

    dip = attrs["dip"]
    if np.max(dip) == np.min(dip):
        fault["dip"] = np.min(dip)
    else:
        print("unexpected dip value")
        exit()

    fault["depth"] = attrs["hdepth"]

    return pd.Series(fault, name=event_name)


def run_emp_gmms(
    output_dir: Path,
    ll_ffp: Path,
    vs30_ffp: Path,
    z_ffp: Path,
    srf_ffp: Path,
    srfinfo_ffp: Path = None,
    nz_gmdb_source_ffp: Path = NZ_GMDB_SOURCE_PATH,
    rjb_max: float = RJB_MAX,
    config_file: Path = None,
    im_list=IM_LIST,
    components=None,
    periods=constants.DEFAULT_PSA_PERIODS,
    extended_period=False,
):
    """
    Computes the empirical GMM parameters for all
        specified sites and sources

    Parameters
    ----------
    output_ffp: Path
    ll_ffp: Path
        Path to .ll file, make sure .vs30 and .z files are present in the same directory
    srf_dir: Path
        Directory that contains the srf files
    nz_gmdb_source_ffp: Path
        Path to the NZ-GMDB source file
    rjb_max: float
        RJB distance threshold

    Returns
    -------
    result_df: DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
    """
    if extended_period:
        periods = np.unique(np.append(periods, constants.EXT_PERIOD))

    tect_type_model_dict = read_model_dict(config_file)

    ### Data loading
    # Get all srf files
    # srf_ffps = list(srf_dir.rglob("*.srf"))

    # events = [cur_ffp.stem for cur_ffp in srf_ffps]
    event = srf_ffp.stem
    # TODO: consider if this is necessary when each event is Cybershake realisation
    event_name = event.split("_")[0]
    # Load source info
    source_df = pd.read_csv(nz_gmdb_source_ffp, index_col=0)

    if srfinfo_ffp is None:
        fault_df = source_df.loc[
            event_name, ["mag", "tect_class", "z_tor", "z_bor", "rake", "dip", "depth"]
        ]
    else:  # this will supercede source_df
        fault_df = load_srf_info(srfinfo_ffp, event_name)

    # Load srf data
    # srf_points, plane_infos = {}, {}
    # for cur_srf_ffp in srf_ffps:
    #     srf_points[cur_srf_ffp.stem] = srf.read_srf_points(str(cur_srf_ffp))
    #     plane_infos[cur_srf_ffp.stem] = srf.read_header(str(cur_srf_ffp), idx=True)
    srf_points = srf.read_srf_points(str(srf_ffp))
    plane_infos = srf.read_header(str(srf_ffp), idx=True)

    # Load the site_data
    site_dir = ll_ffp.parent
    # vs30_ffp = ll_ffp.with_suffix(".vs30")
    # z_ffp = ll_ffp.with_suffix(".z")

    # assert ll_ffp.exists()
    # assert vs30_ffp.exists()
    # assert z_ffp.exists()

    stations_df = pd.read_csv(
        ll_ffp,
        sep=r"\s+",
        index_col=2,
        header=None,
        names=["lon", "lat"],
    )
    vs30_df = pd.read_csv(
        vs30_ffp,
        sep=r"\s+",
        index_col=0,
        header=None,
        names=["vs30"],
    )
    if z_ffp is not None and z_ffp.exists():
        z_df = pd.read_csv(z_ffp, index_col=0)
        z_df = z_df.rename(columns={"Z_1.0(km)": "z1pt0", "Z_2.5(km)": "z2pt5"})

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

    ### Data merging/re-naming and tidy up
    assert np.all(stations_df.index == vs30_df.index) and np.all(
        stations_df.index == z_df.index
    )
    site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
    del stations_df, vs30_df, z_df

    ### Distance calculation
    site_locs = np.concatenate(
        (site_df[["lon", "lat"]].values, np.zeros((site_df.shape[0], 1))), axis=1
    )
    data_dfs = []

    cur_data_df = site_df.copy(True)
    cur_data_df["rrup"], cur_data_df["rjb"] = src_site_dist.calc_rrup_rjb(
        srf_points, site_locs
    )

    cur_data_df["rx"], cur_data_df["ry"] = src_site_dist.calc_rx_ry(
        srf_points, plane_infos, site_locs
    )
    # Enforce distance threshold
    cur_data_df = cur_data_df.loc[cur_data_df.rjb <= rjb_max]
    cur_data_df["site"] = cur_data_df.index.values
    cur_data_df["event"] = str(event)
    # cur_data_df.index = np.add(f"{event}_", cur_data_df.index.values)

    # Add event data
    cur_data_df[
        ["mag", "tect_class", "ztor", "zbot", "rake", "dip", "hypo_depth"]
    ] = fault_df

    data_dfs.append(cur_data_df)

    data_df = pd.concat(data_dfs, axis=0)
    data_df["vs30measured"] = False

    ### GM prediction
    dfs = []
    sites = np.unique(data_df.site)
    for site_ix, cur_site in enumerate(sites):
        print(f"Processing site {cur_site}, {site_ix + 1}/{len(sites)}")

        cur_site_mask = data_df.site.values == cur_site

        for cur_tect_class in np.unique(data_df.loc[cur_site_mask].tect_class):
            cur_tect_mask = cur_site_mask & (data_df.tect_class == cur_tect_class)

            if cur_tect_class not in TECT_CLASS_MAPPING:
                continue

            cur_tect_type = TECT_CLASS_MAPPING[cur_tect_class]
            im_comp_model = tect_type_model_dict[cur_tect_type.name]

            im_results = []
            for im in im_list:
                for comp in components:
                    model_str = im_comp_model.get(im).get(comp)[0]
                    if model_str is None:
                        continue

                    model = GMM[model_str]

                    if im == "pSA":
                        result = oq_run(
                            model,
                            cur_tect_type,
                            data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                            im,
                            periods,
                            convert_mean=np.exp,
                        )
                    else:
                        result = oq_run(
                            model,
                            cur_tect_type,
                            data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                            im,
                            convert_mean=np.exp,
                        )

                    im_results.append(result)

            temp_df = pd.concat(im_results, axis=1)
            temp_df.index = data_df.loc[cur_tect_mask].index
            cur_df = pd.concat([temp_df, data_df[["event", "site"]]], axis=1)

            dfs.append(cur_df)

    result_df = pd.concat(dfs, axis=0)
    output_ffp = output_dir / f"{event}.csv"
    result_df.to_csv(output_ffp, index_label="id")


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
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--srfinfo_ffp",
        help="Path to the SRF info file",
        required=False,
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
        "-c",
        "--config",
        help="configuration file to " "select which model is being used",
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
        "--components",
        nargs="+",
        choices=list(constants.Components.iterate_str_values()),
        default=[constants.Components.cgeom.str_value],
        help="The component(s) you want to calculate."
        " Available components are: [%(choices)s]. Default is %(default)s",
    )
    #
    #    parser.add_argument(
    #        "--gmm_param_config",
    #        default=None,
    #        help="the file that contains the extra parameters for models",
    #    )

    parser.add_argument("output", type=Path, help="output directory")
    args = parser.parse_args()

    return args


def main():
    args = load_args()
    setup_dir(args.output)
    run_emp_gmms(
        args.output,
        args.ll_ffp,
        args.vs30_ffp,
        args.z_ffp,
        args.srf_ffp,
        args.srfinfo_ffp,
        args.nz_gmdb_source_ffp,
        args.max_rupture_distance,
        args.config,
        args.im,
        args.components,
        args.periods,
        args.extended_period,
    )


if __name__ == "__main__":
    main()
