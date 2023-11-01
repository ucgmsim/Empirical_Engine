from pathlib import Path
import pandas as pd
import numpy as np


from qcore import srf, constants
from empirical.util.openquake_wrapper_vectorized import oq_run
from empirical.util.classdef import TectType, GMM
from IM_calculation.source_site_dist import src_site_dist

RJB_MAX = 200

NZ_GMDB_SOURCE_PATH = Path(__file__).parent / "Data" / "earthquake_source_table.csv"

def run_emp_gmms(
    output_ffp: Path,
    site_dir: Path,
    srf_dir: Path,
    nz_gmdb_source_ffp: Path,
    rjb_max: float = RJB_MAX,
):
    """
    Computes the empirical GMM parameters for all
        specified sites and sources

    Parameters
    ----------
    output_ffp: Path
    site_dir: Path
        Directory that contains all the site
        information files (i.e. vs30, ll, and z)
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


    ### Constants
    GMM_MAPPING = {
        TectType.ACTIVE_SHALLOW: GMM.Br_10,
        TectType.SUBDUCTION_SLAB: GMM.ZA_06,
        TectType.SUBDUCTION_INTERFACE: GMM.ZA_06,
    }

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
        "mag",
        "rake",
        "dip",
        "vs30measured",
        "ztor",
        "rx",
        "hypo_depth",
    ]

    ### Data loading
    # Get all srf files
    srf_ffps = list(srf_dir.rglob("*.srf"))
    events = [cur_ffp.stem for cur_ffp in srf_ffps]

    # Load source info
    source_df = pd.read_csv(nz_gmdb_source_ffp, index_col=0)

    # Load srf data
    srf_points, plane_infos = {}, {}
    for cur_srf_ffp in srf_ffps:
        srf_points[cur_srf_ffp.stem] = srf.read_srf_points(str(cur_srf_ffp))
        plane_infos[cur_srf_ffp.stem] = srf.read_header(str(cur_srf_ffp), idx=True)

    # Load the site_data
    stations_df = pd.read_csv(
        site_dir / f"{constants.STATION_FN_NAME}.ll",
        sep=" ",
        index_col=2,
        header=None,
        names=["lon", "lat"],
    )
    vs30_df = pd.read_csv(
        site_dir / f"{constants.STATION_FN_NAME}.vs30",
        sep=" ",
        index_col=0,
        header=None,
        names=["vs30"],
    )
    z_df = pd.read_csv(site_dir / f"{constants.STATION_FN_NAME}.z", index_col=0)

    ### Data merging/re-naming and tidy up
    assert np.all(stations_df.index == vs30_df.index) and np.all(
        stations_df.index == z_df.index
    )
    site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
    site_df = site_df.rename(columns={"Z_1.0(km)": "z1pt0"})
    del stations_df, vs30_df, z_df

    ### Distance calculation
    site_locs = np.concatenate(
        (site_df[["lon", "lat"]].values, np.zeros((site_df.shape[0], 1))), axis=1
    )
    data_dfs = []
    for cur_event in events:
        cur_data_df = site_df.copy(True)
        cur_data_df["rrup"], cur_data_df["rjb"] = src_site_dist.calc_rrup_rjb(
            srf_points[cur_event], site_locs
        )

        cur_data_df["rx"], cur_data_df["ry"] = src_site_dist.calc_rx_ry(
            srf_points[cur_event], plane_infos[cur_event], site_locs
        )
        # Enforce distance threshold
        cur_data_df = cur_data_df.loc[cur_data_df.rjb <= rjb_max]
        cur_data_df["site"] = cur_data_df.index.values
        cur_data_df["event"] = str(cur_event)
        cur_data_df.index = np.add(f"{cur_event}_", cur_data_df.index.values)

        # Add event data
        cur_data_df[
            ["mag", "tect_class", "ztor", "rake", "dip", "hypo_depth"]
        ] = source_df.loc[
            cur_event, ["mag", "tect_class", "z_tor", "rake", "dip", "depth"]
        ]

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
            pga_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "PGA",
            )

            psa_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "pSA",
                constants.PERIODS,
            )

            cur_df = pd.concat((pga_result, psa_result), axis=1)
            cur_df.index = data_df.loc[cur_tect_mask].index
            cur_df[["event", "site"]] = data_df[["event", "site"]]

            dfs.append(cur_df)

    result_df = pd.concat(dfs, axis=0)
    result_df.to_csv(output_ffp, index_label="id")


def load_args():
    parser = argparse.ArgumentParser(
        description="Script to calculate IMs for empirical models."
        "Produces one .csv for each IM containing "
        "all specified sites."
    )
    parser.add_argument(
        "--vs30_file",
        "-v",
        required=True,
        help="vs30 file. Default value is 250 if" " station/file not present",
    )
    # parser.add_argument(
    #     "-r",
    #     "--rupture_distance",
    #     required=True,
    #     help="Path to the rupture distance csv file",
    # )
    parser.add_argument(
        "-srf", "--srf_info", help="Path to srf-info file", required=True
    )
    parser.add_argument(
        "--vs30_default",
        default=classdef.VS30_DEFAULT,
        help="Sets the default value for the vs30",
    )
    parser.add_argument(
        "-s",
        "--stations",
        nargs="+",
        help="List of stations to calculate empiricals for",
    )
    parser.add_argument(
        "-rm",
        "--max_rupture_distance",
        type=float,
        help="Only calculate empiricals for stations "
        "that are within X distance to rupture",
    )
    parser.add_argument("-i", "--identifier", help="run-name for run")
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
        "--period",
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
    # TODO: Put common argparse arguments between IM_calc and empirical in shared file
    parser.add_argument(
        "-comp",
        "--components",
        nargs="+",
        choices=list(constants.Components.iterate_str_values()),
        default=[constants.Components.cgeom.str_value],
        help="The component(s) you want to calculate."
        " Available components are: [%(choices)s]. Default is %(default)s",
    )

    parser.add_argument(
        "--gmm_param_config",
        default=None,
        help="the file that contains the extra parameters for models",
    )

    parser.add_argument("output", help="output directory")
    args = parser.parse_args()
    return args


def main():
    args = load_args()
    setup_dir(args.output)
    calculate_empirical(
        args.identifier,
        args.srf_info,
        args.output,
        args.config,
        args.stations,
        args.vs30_file,
        args.vs30_default,
        args.im,
        args.rupture_distance,
        args.max_rupture_distance,
        args.period,
        args.extended_period,
        args.components,
        args.gmm_param_config,
    )


if __name__ == "__main__":
    main()
