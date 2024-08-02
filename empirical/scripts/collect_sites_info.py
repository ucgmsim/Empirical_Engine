"""
Script to produce a CSV file containing site info (vs30, z1.0, z2.5) and rupture-source distances (rrup, rjb, rx, ry)
within a specified maximum distance.
This script is a pre-requisite for calculate_empirical.py as the .csv file produced is a required input.
"""

from pathlib import Path

import pandas as pd
import typer
from typing_extensions import Annotated

from empirical.util import empirical
from qcore import nhm, formats

RJB_MAX = 200

app = typer.Typer()


@app.command()
def collect_sites_info_(
    source_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to the SRF or NHM file. NHM file can be obtained from https://github.com/ucgmsim/Empirical_Engine/files/15256612/NZ_FLTmodel_2010_v18p6.txt",
            file_okay=True,
            exists=True,
            readable=True,
        ),
    ],
    ll_ffp: Annotated[
        Path,
        typer.Argument(help="Path to the .ll file", file_okay=True, exists=True),
    ],
    vs30_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to the .vs30 file. The file should have columns: station, vs30, (sigma)",
            file_okay=True,
            exists=True,
        ),
    ],
    z_ffp: Annotated[
        Path,
        typer.Argument(
            help="Path to the .z file that contains Z1.0 and Z2.5. The file should have columns: station, z1p0, z2p5",
            file_okay=True,
            exists=True,
            readable=True,
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            help="output directory",
            dir_okay=True,
            writable=True,
        ),
    ] = Path().cwd(),
    nhm_fault_name: Annotated[
        str,
        typer.Option(
            help="Fault name to look up from NHM. Required if source-ffp is an NHM file"
        ),
    ] = "",
    rjb_max: Annotated[
        float,
        typer.Option(help="Only calculate for stations whose rjb is below this value"),
    ] = RJB_MAX,
):
    """
    Calculate source-site distances for stations in ll_ffp and put them alongside vs30 and z values into a single CSV file.
    Can work either with SRF or NHM for a valid fault

    """
    print(source_ffp.suffix)
    if source_ffp.suffix == ".srf":
        event_name = source_ffp.stem  # use srf name as the event_name
    else:
        # srf_ffp is not an SRF. May be an NHM file?
        print(f"INFO: {source_ffp} is not an SRF file")

        assert (
            len(nhm_fault_name) > 0
        ), f"--nhm-fault-name must be specified if source-ffp is not SRF."

        try:
            nhm_data = nhm.load_nhm(str(source_ffp))
        except:  # can be a random file causing random exception
            print(f"ERROR: {source_ffp} is not a valid NHM file.")
            return

        try:
            # See if nhm_data has info for this fault
            source_ffp = nhm_data[nhm_fault_name]
        except KeyError:
            raise ValueError(f"Unknown fault {nhm_fault_name}")
        else:
            print(f"INFO: Found {nhm_fault_name} in NHM.")
            event_name = nhm_fault_name  # use this fault name as the event_name

    out_dir.mkdir(exist_ok=True)

    stations_df = formats.load_station_file(ll_ffp)
    vs30_df = formats.load_vs30_file(vs30_ffp)
    z_df = formats.load_z_file(z_ffp)
    z_df = z_df.rename(columns={"z1p0": "z1pt0", "z2p5": "z2pt5"})

    site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
    del stations_df, vs30_df, z_df

    rrup_df = empirical.get_site_source_data(source_ffp, site_df[["lon", "lat"]].values)

    # Combine lon_lat_df with relevant columns from rrup_df
    combined_sites_info_df = pd.concat(
        [site_df, rrup_df.set_index(site_df.index)], axis=1
    )

    # Filter site_df to only include sites with rjb <= rjb_max
    combined_sites_info_df = combined_sites_info_df[
        combined_sites_info_df.rjb <= rjb_max
    ]
    outpath = out_dir / f"sites_info_{event_name}.csv"
    combined_sites_info_df[
        ["lon", "lat", "rrup", "rjb", "rx", "ry", "vs30", "z1pt0", "z2pt5"]
    ].to_csv(outpath)
    print(f"Success: {outpath}")


if __name__ == "__main__":
    app()
