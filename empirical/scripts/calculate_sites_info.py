import argparse
from pathlib import Path

import pandas as pd

from empirical.util import empirical
from qcore import nhm, formats, utils

RJB_MAX = 200


def calculate_source_to_site_distance(
    output_dir: Path,
    event_name: str,
    ll_ffp: Path,
    vs30_ffp: Path,
    z_ffp: Path,
    srf_ffp: Path = None,
    nhm_ffp: Path = None,
    rjb_max: float = RJB_MAX,
):
    """
    Calculate source-site distances for stations in ll_ffp and put them alongside vs30 and z values into a single CSV file.
    Can work either with SRF or NHM for a valid fault

    Parameters
    ----------
    output_dir  : Path
        Path to the output directory
    event_name  : str
        Name of the event (or fault)
    ll_ffp  : Path
        lat lon file path for the stations
    vs30_ffp    : Path
        vs30 file path for vs30 values at the stations
    z_ffp   : Path
        z file path for z1.0 and z2.5 values at the stations
    srf_ffp  : Path
        Path to the SRF file
    nhm_ffp  : Path (default=None)
        Path to the NHM file. If srf_ffp is not provided, this is used to get the fault data.
    rjb_max  : float, optional (default=RJB_MAX)
        Only calculate for stations whose rjb is below this value
    """

    stations_df = formats.load_station_file(ll_ffp)
    vs30_df = formats.load_vs30_file(vs30_ffp)
    z_df = formats.load_z_file(z_ffp)
    z_df = z_df.rename(columns={"z1p0": "z1pt0", "z2p5": "z2pt5"})

    site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
    del stations_df, vs30_df, z_df

    if srf_ffp is None:
        # If srf_ffp is not supplied, but it is a valid fault, we can use NHM to get the fault data and proceed
        if nhm_ffp is not None:
            nhm_data = nhm.load_nhm(str(nhm_ffp))
            # Get fault data
            try:
                # We are reconstructing the missing srf_ffp from nhm.
                srf_ffp = nhm_data[event_name]
            except KeyError:
                raise ValueError(f"Unknown fault {event_name}")
            else:
                print(f"INFO: Found {event_name} in NHM.")
        else:
            raise RuntimeError(f"nhm_ffp is required if srf_ffp is not provided.")

    rrup_df = empirical.get_site_source_data(srf_ffp, site_df[["lon", "lat"]].values)

    # Combine lon_lat_df with relevant columns from rrup_df
    rrup_to_export = pd.concat([site_df, rrup_df.set_index(site_df.index)], axis=1)

    # Filter site_df to only include sites with rjb <= rjb_max
    rrup_to_export = rrup_to_export[rrup_to_export.rjb <= rjb_max]
    rrup_to_export[
        ["lon", "lat", "rrup", "rjb", "rx", "ry", "vs30", "z1pt0", "z2pt5"]
    ].to_csv(output_dir / f"sites_info_{event_name}.csv")


def load_args():
    parser = argparse.ArgumentParser(
        description="Script to produce a CSV file containing site info (vs30, z1.0, z2.5) and rupture-source distances "
        "and rupture-source distances (rrup, rjb, rx, ry) within a specified maximum distance."
        "This script is a pre-requisite for calculate_empirical.py as the .csv file produced is required to "
        "calculate IMs for empirical models."
    )

    parser.add_argument(
        "--srf_ffp",
        help="Path to the SRF file",
        type=Path,
    )

    parser.add_argument(
        "--nhm_ffp",
        help="Path to the NHM file. If srf_ffp is not provided, this is used to get the fault data. "
        "Get one from https://github.com/ucgmsim/Empirical_Engine/files/15256612/NZ_FLTmodel_2010_v18p6.txt",
        type=Path,
    )

    parser.add_argument(
        "--event_name",
        help="Name of the fault. Required if srf_ffp is not provided.",
    )

    parser.add_argument(
        "-rm",
        "--max_rupture_distance",
        type=float,
        default=RJB_MAX,
        help="Only collect sites within this distance to the source",
    )

    parser.add_argument("output", type=Path, help="output directory")

    parser.add_argument(
        "ll_ffp",
        type=Path,
        help="Path to the .ll file",
    )

    parser.add_argument(
        "vs30_ffp",
        type=Path,
        help="Path to the .vs30 file."
        "The file should have columns: station, vs30, (sigma)",
    )

    parser.add_argument(
        "z_ffp",
        type=Path,
        help="Path to the .z file that contains Z1.0 and Z2.5."
        "The file should have columns: station, z1p0, z2p5",
    )

    args = parser.parse_args()

    if args.srf_ffp is None and args.nhm_ffp is None:
        parser.error("Either srf_ffp or nhm_ffp is required.")

    if args.event_name is None:
        if args.srf_ffp is not None:
            args.event_name = args.srf_ffp.stem  # can use srf name as the event_name
        else:
            parser.error("event_name is required if srf_ffp is not provided.")
    else:
        if args.nhm_ffp is None:
            parser.error(
                "nhm_ffp is required to be able to re-construct the missing srf."
            )

    return args


def main():
    args = load_args()
    utils.setup_dir(args.output)
    calculate_source_to_site_distance(
        args.output,
        args.event_name,
        args.ll_ffp,
        args.vs30_ffp,
        args.z_ffp,
        args.srf_ffp,
        args.nhm_ffp,
        args.max_rupture_distance,
    )


if __name__ == "__main__":
    main()
