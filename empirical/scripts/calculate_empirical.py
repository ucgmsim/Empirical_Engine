import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml

from empirical.util import empirical
from qcore import constants, utils

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


def run_emp(
    output_dir: Path,
    event_name: str,
    sites_info_ffp: Path,
    srfdata_ffp: Optional[Path] = None,
    nz_gmdb_source_ffp: Optional[Path] = None,
    model_config_ffp: Path = DEFAULT_MODEL_CONFIG_FFP,
    meta_config_ffp: Path = DEFAULT_META_CONFIG_FFP,
    im_list: list[str] = IM_LIST,
    component: str = DEFAULT_COMP,
    periods: list[float] = constants.DEFAULT_PSA_PERIODS,
    extended_period: bool = False,
):
    """
    Run empirical calculations for a given historical event or a future event from a specified fault.

    Parameters
    ----------
    output_dir : Path
        Path to the output directory.
    event_name : str
        Name of the event (or fault).
    sites_info_ffp : Path
        Path to the site CSV file obtained from running collect_sites_info.py.
    srfdata_ffp : Optional[Path], optional
        SRF data file path for the fault data. Can be either realisation .csv or .info file.
        If this is None, we could still proceed if nz_gmdb_source_ffp is provided and it is a known historical event.
    nz_gmdb_source_ffp : Optional[Path], optional
        NZ GMDB source file path for the source data. Must be provided for historical events when srfdata_ffp is missing.
    model_config_ffp : Path, optional
        Model config file path for the empirical model. Prescribes the model to be used for tectonic class, IM, and component.
    meta_config_ffp : Path, optional
        Meta config file path for the empirical model. Prescribes the weight of the model for IM and tectonic class.
    im_list : list[str], optional
        List of intensity measures. Currently supported: PGA, PGV, pSA, CAV, Ds575, Ds595.
    component : str, optional
        Component of the IM (e.g., geom, rotd50) to calculate empirical for.
    periods : list[float], optional
        List of periods for pSA. Default is qcore.constants.DEFAULT_PSA_PERIODS.
    extended_period : bool, optional
        Indicate the use of extended (100) pSA periods.

    """

    if extended_period:
        periods = constants.EXT_PERIOD

    ### Data loading
    model_config = None
    if model_config_ffp.exists():
        with open(model_config_ffp, "r") as f:
            model_config = yaml.safe_load(f)

    # Using Full Loader for the meta config due to the python tuple pSA/PGA
    meta_config = None
    if meta_config_ffp.exists():
        with open(meta_config_ffp, "r") as f:
            meta_config = yaml.load(f, Loader=yaml.FullLoader)

    if srfdata_ffp is not None:
        event = srfdata_ffp.stem
        event_name = event.split("_")[0]
        if srfdata_ffp.suffix == ".info":
            fault_df = empirical.load_srf_info(srfdata_ffp, event_name)
        else:  # .csv
            fault_df = empirical.load_rel_csv(srfdata_ffp, event_name)
        # Either .csv or .info, fault_df has consistent columns defined in NZ_GMDB_SOURCE_COLUMNS
    else:
        # If srfdata_ffp is not supplied, but if it is a known historical event,we can still proceed
        print(f"INFO: srfdata_ffp not provided.")

        # Load source info and check if it is a valid event
        if nz_gmdb_source_ffp is not None:
            nz_gmdb_source_df = pd.read_csv(nz_gmdb_source_ffp, index_col=0)
            try:
                fault_df = nz_gmdb_source_df.loc[
                    event_name, empirical.NZ_GMDB_SOURCE_COLUMNS
                ]
            except KeyError:
                raise ValueError(f"Unknown event {event_name}")
            else:
                print(f"INFO: Found {event_name} in NZGMDB.")
        else:
            raise RuntimeError(
                f"nz_gmdb_source_ffp is required for historical events. Use earthquake_source_table.csv from GMDB.zip"
            )

    # at this point, we have valid fault_df

    tect_type = empirical.TECT_CLASS_MAPPING[fault_df.tect_class]

    # Load sites_info CSV file produced by collect_sites_info.py
    oq_rupture_df = pd.read_csv(sites_info_ffp, index_col=0)

    # Each model (determined by model_config, tect_type, im, component) has different set of required columns
    # Let's make oq_rupture_df from site_df, and fault_df, and hopefully(!) it has all required columns

    oq_rupture_df.loc[
        :,
        empirical.OQ_RUPTURE_COLUMNS,  # rename columns to follow OQ_RUPTURE_COLUMNS
    ] = fault_df[
        empirical.NZ_GMDB_SOURCE_COLUMNS
    ].values  # fault_df has NZ_GMDB_SOURCE_COLUMNS

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
        "Produces one .csv for all specified sites."
        "Run collect_sites_info.py first to generate the CSV file for the 'sites_info_ffp' argument"
    )

    parser.add_argument(
        "--srfdata_ffp",
        help="Path to the SRF .info or .csv file",
        type=Path,
    )

    parser.add_argument(
        "--event_name",
        help="Event name. Required if srfdata_ffp is not provided.",
        type=Path,
    )

    parser.add_argument(
        "--nz_gmdb_source_ffp",
        type=Path,
        help="NZ GMDB source CSV. Required for historical events when srfdata is missing. Use earthquake_source_table.csv "
        "contained in GMDB.zip from https://osf.io/q9yrg/?view_only=05337ba1ebc744fc96b9924de633ca0e",
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
    parser.add_argument(
        "-comp",
        "--component",
        choices=list(constants.Components.iterate_str_values()),
        default=constants.Components.cgeom.str_value,
        help="The component you want to calculate.",
    )

    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument(
        "sites_info_ffp",
        help="Path to the site csv file obtained from running collect_sites_info.py. Contains [station, lon, lat, rrup, rjb,rx,ry, vs30, z1p0, z2p5].",
        type=Path,
    )

    args = parser.parse_args()

    if args.srfdata_ffp is None and args.nz_gmdb_source_ffp is None:
        parser.error("Either srfdata_ffp or nz_gmdb_source_ffp is required.")

    if args.event_name is None:
        if args.srfdata_ffp is not None:
            args.event_name = args.srfdata_ffp.stem
        else:
            parser.error("event_name is required if srfdata_ffp is not provided.")
    else:
        if args.nz_gmdb_source_ffp is None:
            parser.error(
                "nz_gmdb_source_ffp is required to be able to find event info from GMDB."
            )

    return args


def main():
    args = load_args()
    utils.setup_dir(args.output)
    run_emp(
        args.output,
        args.event_name,
        args.sites_info_ffp,
        args.srfdata_ffp,
        args.nz_gmdb_source_ffp,
        args.model_config_ffp,
        args.meta_config_ffp,
        args.im,
        args.component,
        args.periods,
        args.extended_period,
    )


if __name__ == "__main__":
    main()
