from pathlib import Path
from typing import List, Dict
import argparse
import json

import pandas as pd
import numpy as np
import geojson
from turfpy.measurement import points_within_polygon

from empirical.util.classdef import TectType, GMM
from empirical.util import openquake_wrapper_vectorized
from mera.mera_pymer4 import run_mera


DEFAULT_IMS = ["PGA", "pSA"]
DEFAULT_MODELS = [
    "S_22",
    "A_22",
    "ASK_14",
    "CY_14",
    "BSSA_14",
    "CB_14",
    "Br_13",
    "AG_20",
    "P_20",
    "K_20",
]


def convert_tect_class(tect_type):
    if tect_type == "Crustal":
        return TectType.ACTIVE_SHALLOW
    elif tect_type == "Interface":
        return TectType.SUBDUCTION_INTERFACE
    elif tect_type == "Slab":
        return TectType.SUBDUCTION_SLAB
    elif tect_type == "Outer-rise" or tect_type == "Undetermined":
        # Ignore these tectonic types
        return None


def get_backarc_mask(backarc_json_ffp: Path, locs: np.ndarray):
    """
    Computes a mask identifying each location
    that requires the backarc flag based on
    if it is inside the backarc polygon or not

    locs: array of floats
        [lon, lat]

    """

    # Determine if backarc needs to be enabled for each loc
    points = geojson.FeatureCollection(
        [
            geojson.Feature(geometry=geojson.Point(tuple(cur_loc[::-1]), id=ix))
            for ix, cur_loc in enumerate(locs)
        ]
    )
    with backarc_json_ffp.open("r") as f:
        poly_coords = np.flip(json.load(f)["geometry"]["coordinates"][0], axis=1)

    polygon = geojson.Polygon([poly_coords.tolist()])
    backarc_ind = (
        [
            cur_point["geometry"]["id"]
            for cur_point in points_within_polygon(points, polygon)["features"]
        ],
    )
    backarc_mask = np.zeros(shape=locs.shape[0], dtype=bool)
    backarc_mask[backarc_ind] = True

    return backarc_mask


def filter_gm_csv(gm_csv: Path):
    """
    Filter the given gm csv to only include the rows that have valid data
    :param gm_csv: The path to the gm csv
    :return: Ground Motion DataFrame containing only the valid rows
    """
    gm_df = pd.read_csv(gm_csv, dtype={"gmid": str, "evid": str})
    gm_df = gm_df.loc[gm_df.loc[:, ["sta_lon", "mag"]].notnull().all(axis=1)]
    return gm_df


def calc_empirical(
    gm_csv: Path,
    backarc_json_ffp: Path,
    output_dir: Path,
    models: List[str] = None,
    ims: List[str] = None,
    periods: List[float] = None,
):
    """
    Calculate the empirical IMs for the given gm csv and output the results to the given output directory.
    :param gm_csv: The path to the csv containing the ground motion data
    :param backarc_json_ffp: The path to the backarc json file
    :param output_dir: The path to the output directory
    :param models: The list of models to use
    :param ims: The list of IMs to calculate
    :param periods: The list of periods to calculate
    :return: Dict of IM DataFrames per model
    """
    # Read gm csv and get periods if needed
    gm_df = filter_gm_csv(gm_csv)
    if periods is None:
        periods = sorted([float(col[4:]) for col in gm_df.columns if "pSA" in col])

    # Get models and IM's if needed
    if models is None:
        models = DEFAULT_MODELS
    if ims is None:
        ims = DEFAULT_IMS

    # Calculate the backarc mask
    print("Calculating backarc mask")
    backarc_mask = get_backarc_mask(
        backarc_json_ffp, gm_df[["sta_lon", "sta_lat"]].values
    )
    print("Backarc mask calculated")

    # Create the rupture dataframe for open quake
    rupture_df = pd.DataFrame(
        {
            "gmid": gm_df["gmid"],
            "evid": gm_df["evid"],
            "sta": gm_df["sta"],
            "mag": gm_df["mag"],
            "dip": gm_df["dip"],
            "rake": gm_df["rake"],
            "dbot": gm_df["z_bor"],
            "zbot": gm_df["z_bor"],
            "ztor": gm_df["z_tor"],
            "tect_type": [convert_tect_class(t) for t in gm_df["tect_class"]],
            "rjb": gm_df["r_jb"],
            "rrup": gm_df["r_rup"],
            "rx": gm_df["r_x"],
            "ry": gm_df["r_y"],
            "rtvz": gm_df["r_tvz"],
            "lon": gm_df["sta_lon"],
            "lat": gm_df["sta_lat"],
            "vs30": gm_df["Vs30"],
            "z1pt0": gm_df["Z1.0"],
            "z2pt5": gm_df["Z2.5"],
            "vs30measured": False,
            "hypo_depth": gm_df["ev_depth"],
            "backarc": backarc_mask,
        }
    )

    # Make the model output directory
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Calculate im csvs for each of the GMMs / TectTypes using the data in the gm_df
    model_outputs = dict()
    for model in models:
        model = GMM[model]
        # Get the tect types for the model
        tect_types = openquake_wrapper_vectorized.OQ_MODELS[model].keys()
        tect_df_list = []
        for tect_type in tect_types:
            # Extract the rupture data information that has the same tect type
            tect_rup_df = rupture_df.copy(deep=True).loc[
                rupture_df["tect_type"] == tect_type
            ]
            im_df_list = []
            print(f"Calculating for {model.name} {tect_type}")
            for im in ims:
                im_df = openquake_wrapper_vectorized.oq_run(
                    model,
                    tect_type,
                    tect_rup_df,
                    im,
                    periods if im == "pSA" else None,
                )
                im_df = np.exp(im_df.loc[:, im_df.columns.str.contains("mean")])
                im_df = im_df.rename(
                    {
                        col_name: col_name.rstrip("_mean")
                        for col_name in im_df.columns.values
                    },
                    axis="columns",
                )
                im_df_list.append(im_df)

            # Combine all the different IM DataFrames into one
            tect_result_df = pd.concat(im_df_list, axis=1)
            # Insert back the gmid, evid and station to the dataframe
            tect_result_df.index = tect_rup_df.index
            tect_result_df.insert(0, "sta", tect_rup_df["sta"])
            tect_result_df.insert(0, "evid", tect_rup_df["evid"])
            tect_result_df.insert(0, "gmid", tect_rup_df["gmid"])
            tect_df_list.append(tect_result_df)
        # Combine all the tect type results into one dataframe
        model_result_df = pd.concat(tect_df_list, axis=0).sort_index()
        print(f"Writing {model.name} to {model_dir / f'{model.name}.csv'}")
        model_result_df.to_csv(model_dir / f"{model.name}.csv", index=False)
        model_outputs[model] = model_result_df
    return model_outputs


def calc_residuals(
    gm_csv: Path,
    model_outputs: Dict[GMM, pd.DataFrame],
    output_dir: Path,
    ims: List[str] = None,
):
    """
    Calculate the residuals between the observed IMs and the model IMs
    :param gm_csv: The path to the gm csv
    :param model_outputs: The model outputs with the given model as the dictionary key
    :param output_dir: The path to the output directory
    :param ims: The IMs to calculate the residuals for
    """
    # Make the residual output directory
    residual_dir = output_dir / "residuals"
    residual_dir.mkdir(exist_ok=True)

    # Read gm csv for observed data
    gm_df = filter_gm_csv(gm_csv)

    # Get default IM's if needed
    if ims is None:
        ims = DEFAULT_IMS

    # Loop over each model output and calculate the residuals
    for model, model_df in model_outputs.items():
        # Extract the same IMs from the observed data
        im_columns = [
            cur_im
            for cur_im in np.intersect1d(model_df.columns, gm_df.columns)
            if any(im in cur_im for im in ims)
        ]

        # Compute the residual
        res_df = np.log(gm_df.loc[model_df.index, im_columns] / model_df[im_columns])

        # Add event id and station id columns
        res_df.insert(0, "sta", model_df["sta"])
        res_df.insert(0, "evid", model_df["evid"])
        res_df.insert(0, "gmid", model_df["gmid"])

        # Run MER
        print(f"Running MER for {model.name}")
        event_res_df, site_res_df, rem_res_df, bias_std_df = run_mera(
            res_df, list(im_columns), "evid", "sta"
        )

        # rename the IM columns in each DataFrame to reflect their residual information
        event_res_df = event_res_df.rename(
            {col: f"{col}_event" for col in event_res_df.columns}, axis="columns"
        )
        site_res_df = site_res_df.rename(
            {col: f"{col}_site" for col in site_res_df.columns}, axis="columns"
        )
        rem_res_df = rem_res_df.rename(
            {col: f"{col}_rem" for col in rem_res_df.columns}, axis="columns"
        )
        bias_std_df = bias_std_df.T.unstack().to_frame().sort_index(level=1).T
        bias_std_df.columns = bias_std_df.columns.map("_".join)

        # Get event data from the model_df
        full_res_df = pd.DataFrame(model_df[["gmid", "evid", "sta"]])

        # Merge the event, site and rem residual dataframes
        full_res_df = full_res_df.merge(site_res_df, left_on="sta", right_index=True)
        full_res_df = full_res_df.merge(event_res_df, left_on="evid", right_index=True)
        full_res_df = full_res_df.merge(rem_res_df, left_index=True, right_index=True)
        full_res_df = full_res_df.merge(bias_std_df, how="cross")

        # Save the full residual dataframe
        print(f"Writing residual results to {residual_dir}/{model.name}.csv")
        full_res_df.to_csv(residual_dir / f"{model.name}.csv", index=False)


def load_args():
    parser = argparse.ArgumentParser(
        description="Calculate the empirical IMs for the given gm csv"
        "and output the results to the given output directory."
    )
    parser.add_argument(
        "gm_csv",
        type=Path,
        help="The path to the csv containing the ground motion data",
    )
    parser.add_argument(
        "backarc_json_ffp",
        type=Path,
        help="The path to the json file containing the backarc polygon information",
    )
    parser.add_argument(
        "output_dir", type=Path, help="The path to the output directory"
    )
    parser.add_argument(
        "--models",
        type=List[str],
        default=DEFAULT_MODELS,
        help="List of models to use. Defaults to all models used in thr NZSHM",
    )
    parser.add_argument(
        "--ims",
        type=List[str],
        default=DEFAULT_IMS,
        help="List of IMs to calculate. Defaults to PGA and pSA",
    )
    parser.add_argument(
        "--periods",
        type=List[float],
        default=None,
        help="List of periods to calculate. Defaults to all periods in the gm csv",
    )

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    model_outputs = calc_empirical(
        args.gm_csv,
        args.backarc_json_ffp,
        args.output_dir,
        args.models,
        args.ims,
        args.periods,
    )
    calc_residuals(args.gm_csv, model_outputs, args.output_dir)


if __name__ == "__main__":
    main()
