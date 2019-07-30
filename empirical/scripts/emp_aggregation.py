#!/usr/bin/env python2
"""Script to aggregate individual IM csv files into a single file.

Sites in the individual IMs files have to match.
Resulting csv file will have consistent column order.
"""
import argparse
import os
import numpy as np
import pandas as pd

from datetime import datetime

from qcore.im import order_im_cols_df

STATION_COL_NAME = "station"
COMPONENT_COL_NAME = "component"


def aggregate_data():
    parser = argparse.ArgumentParser(
        description="Script to aggregate individual IM " "csv files into a single file."
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="path to output folder that stores the " "aggregated measures",
    )
    parser.add_argument("-i", "--identifier", help="run-name for run")
    parser.add_argument(
        "-r",
        "--rupture",
        default="unknown",
        help="Please specify the rupture name of the " "simulation. eg.Albury",
    )
    parser.add_argument(
        "-v", "--version", default="XXpY", help="The version of the simulation. eg.18p4"
    )
    parser.add_argument("im_files", nargs="+")
    args = parser.parse_args()

    dfs = [pd.read_csv(im_file) for im_file in args.im_files]

    # Check that they all have the same number of rows
    # and that the stations match
    ref_n_rows, ref_stations = None, None
    for ix, df in enumerate(dfs):
        if ref_n_rows is None:
            ref_n_rows = df.shape[0]
            ref_stations = np.sort(df[STATION_COL_NAME].values)
        else:
            if ref_n_rows != df.shape[0] or np.any(
                np.sort(df[STATION_COL_NAME].values) != ref_stations
            ):
                raise Exception(
                    "Input files {} and {} are incompatible. Either different "
                    "number of entries, or stations/sites don't match".format(
                        args.im_files[0], args.im_files[ix]
                    )
                )

    # Concatenate the dataframes
    result_df = pd.concat(dfs, axis=1, sort=False, join="inner")

    # Drop duplicate station and component rows
    tmp_stations = result_df[STATION_COL_NAME].iloc[:, 0]
    tmp_components = result_df[COMPONENT_COL_NAME].iloc[:, 0]
    result_df.drop([STATION_COL_NAME, COMPONENT_COL_NAME], axis=1, inplace=True)
    result_df[STATION_COL_NAME] = tmp_stations
    result_df[COMPONENT_COL_NAME] = tmp_components

    if result_df.shape[0] != ref_n_rows:
        raise Exception("Resulting dataframe has the inccorect shape!")

    # Order the columns
    result_df = order_im_cols_df(result_df)

    result_df.to_csv(
        os.path.join(args.output_dir, "{}.csv".format(args.identifier)),
        index=False,
        float_format="%.6f",
    )

    # Metadata file
    metadata_fname = "{}_empirical.info".format(args.identifier)
    metadata_path = os.path.join(args.output_dir, metadata_fname)
    with open(metadata_path, "w") as f:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")

        f.write("identifier,rupture,type,date,version\n")
        f.write(
            "{},{},empirical,{},{}".format(
                args.identifier, args.rupture, date, args.version
            )
        )


if __name__ == "__main__":
    aggregate_data()
