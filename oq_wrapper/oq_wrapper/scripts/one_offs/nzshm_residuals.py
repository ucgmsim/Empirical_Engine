import argparse
import json
from pathlib import Path

import geojson
import numpy as np
import pandas as pd
from turfpy.measurement import points_within_polygon

from empirical.util import openquake_wrapper_vectorized
from empirical.util.classdef import GMM, TectType
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
TECT_CLASS_MAP = {
    "Crustal": TectType.ACTIVE_SHALLOW,
    "Interface": TectType.SUBDUCTION_INTERFACE,
    "Slab": TectType.SUBDUCTION_SLAB,
    "Outer-rise": None,
    "Undetermined": None,
}
TECT_CLASS_MAP_REV = {v: k for k, v in TECT_CLASS_MAP.items()}


def sort_period_columns(df: pd.DataFrame):
    """
    Sort the columns of the given dataframe by period
    Manages the IM's if they are labelled using the . or the p notation
    :param df: The dataframe to sort
    """
    sort_order = sorted(
        df.columns,
        key=lambda col: float(col[4:].replace("p", ".")) if "pSA" in col else 0,
    )
    df = df.reindex(sort_order, axis=1)
    return df


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
    # Remove rows with NaNs in the station lon lat values and the magnitude value
    gm_df = gm_df.loc[gm_df.loc[:, ["sta_lon", "mag"]].notnull().all(axis=1)]
    # Only include Accelerometer channels
    gm_df = gm_df.loc[gm_df["chan"].isin(["HN", "BN"])]
    # Only include rows with accepted rrup values per tectonic class
    gm_df = gm_df.loc[
        (
            (
                (gm_df["tect_class"] == "Crustal")
                & (gm_df["r_rup"] <= 300)
                & (gm_df["mag"] >= 3.5)
            )
            | gm_df["tect_class"]
            != "Crustal"
        )
    ]
    gm_df = gm_df.loc[
        (
            (
                (gm_df["tect_class"].isin(["Interface", "Slab"]))
                & (gm_df["r_rup"] <= 500)
                & (gm_df["mag"] >= 4.5)
            )
            | ~gm_df["tect_class"].isin(["Interface", "Slab"])
        )
    ]
    return gm_df


def generate_period_mask(gm_df: pd.DataFrame):
    """
    Get a mask for the gm_df filtering for T < Tmax, where Tmax is 1/fmin
    fmin is defined as np.sqrt(event_site["f_min_X"] * event_site["f_min_Y"])
    :param gm_df: The dataframe to filter
    :return: A mask for the gm_df where values are True if they are outside the filter
    """
    # Create a dataframe with the period values for each column that's pSA in gm_df
    psa_str_cols = [col for col in gm_df.columns if "pSA" in col]
    psa_cols = sorted([float(col[4:].replace("p", ".")) for col in psa_str_cols])
    period_values = pd.DataFrame(
        {col: col for col in psa_cols}, index=gm_df.index, columns=psa_cols
    )
    period_values.columns = psa_str_cols

    # Calculate f_min and t_max
    f_min = np.sqrt(gm_df["fmin_mean_X"] * gm_df["fmin_mean_Y"])
    t_max = 1 / f_min

    # Extend t_max by duplicating each row of the t_max values for each col in psa_cols
    t_max_expanded = pd.concat([t_max] * len(psa_cols), axis=1)
    t_max_expanded.columns = psa_str_cols

    # Compare the expanded t_max values to the period values to get the final mask
    mask = period_values > t_max_expanded
    return mask


def calc_empirical(
    gm_csv: Path,
    backarc_json_ffp: Path,
    output_dir: Path,
    models: list[str] = None,
    ims: list[str] = None,
    periods: list[float] = None,
    period_specific_ffp: Path = None,
):
    """
    Calculate the empirical IMs for the given gm csv and output the results to the given output directory.
    :param gm_csv: The path to the csv containing the ground motion data
    :param backarc_json_ffp: The path to the backarc json file
    :param output_dir: The path to the output directory
    :param models: The list of models to use
    :param ims: The list of IMs to calculate
    :param periods: The list of periods to calculate
    :param period_specific_ffp: The path to the period specific file
    :return: Dict of IM DataFrames per model and filtered gm csv as a dataframe
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
    # Get unique locations
    locs = gm_df[["sta", "sta_lon", "sta_lat"]].drop_duplicates()
    locs["backarc"] = get_backarc_mask(
        backarc_json_ffp, locs[["sta_lon", "sta_lat"]].values
    )

    # Merge backarc mask into gm_df on each unique location
    gm_df = gm_df.merge(locs[["sta", "backarc"]], on="sta", how="left")

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
            "tect_type": [TECT_CLASS_MAP[t] for t in gm_df["tect_class"]],
            "rjb": gm_df["r_jb"],
            "rrup": gm_df["r_rup"],
            "rx": gm_df["r_x"],
            "ry": gm_df["r_y"],
            "rtvz": gm_df["r_tvz"],
            "lon": gm_df["sta_lon"],
            "lat": gm_df["sta_lat"],
            "vs30": gm_df["Vs30"],
            "z1pt0": gm_df["Z1.0"] / 1000,  # Convert to km
            "z2pt5": gm_df["Z2.5"],
            "vs30measured": False,
            "hypo_depth": gm_df["ev_depth"],
            "backarc": gm_df["backarc"],
        }
    )

    # Calculate the fmin mask
    period_mask = generate_period_mask(gm_df)

    # Filter the gm_df by the period mask
    gm_df[period_mask.columns] = gm_df[period_mask.columns].mask(period_mask)

    # Grab the period specific data
    if period_specific_ffp is not None:
        period_specific_df = pd.read_csv(period_specific_ffp, delimiter=" ")
        period_specific_df["TectonicType"] = [
            TECT_CLASS_MAP[t] for t in period_specific_df["TectonicType"]
        ]
    else:
        period_specific_df = None

    # Make the model output directory
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Calculate im csvs for each of the GMMs / TectTypes using the data in the gm_df
    model_outputs = dict()
    for str_model in models:
        model = GMM[str_model]
        # Get the tect types for the model
        tect_types = openquake_wrapper_vectorized.OQ_MODELS[model].keys()
        for tect_type in tect_types:
            # Extract the rupture data information that has the same tect type
            tect_rup_df = rupture_df.loc[rupture_df["tect_type"] == tect_type]
            im_df_list = []
            print(f"Calculating for {model.name} {tect_type}")
            for im in ims:
                ps_tect_df = None
                if im == "pSA" and period_specific_df is not None:
                    # Filter down by model and tect type
                    ps_model_df = period_specific_df.loc[
                        period_specific_df["gmmID"] == str_model
                    ]
                    ps_tect_df = ps_model_df.loc[
                        ps_model_df["TectonicType"] == tect_type
                    ]

                    # Renaming
                    ps_tect_df = ps_tect_df.rename({"stat_id": "sta"}, axis="columns")
                    new_column_names = {
                        col: f"adj_{col.replace('.', 'p')}"
                        for col in ps_tect_df.columns
                        if "pSA" in col
                    }
                    ps_tect_df = ps_tect_df.rename(columns=new_column_names)

                    # Fill in missing stations with zeros for adjustment factors
                    ps_tect_df = (
                        tect_rup_df.loc[:, ["sta"]]
                        .merge(
                            ps_tect_df[["sta", *new_column_names.values()]],
                            on="sta",
                            how="left",
                        )
                        .fillna(0)
                    )

                im_df = openquake_wrapper_vectorized.oq_run(
                    model,
                    tect_type,
                    tect_rup_df,
                    im,
                    periods if im == "pSA" else None,
                    kwargs={"period_specific_df": ps_tect_df},
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
            # Get the tect type string
            tect_type_str = TECT_CLASS_MAP_REV[tect_type]

            # Filter the tect_result_df by the period mask
            tect_result_df[period_mask.columns] = tect_result_df[
                period_mask.columns
            ].mask(period_mask)

            # Save the tect type results to a csv
            print(
                f"Writing {model.name} {tect_type_str} to {model_dir / f'{model.name}_{tect_type_str}.csv'}"
            )
            tect_result_df.to_csv(
                model_dir / f"{model.name}_{tect_type_str}.csv", index=False
            )
            model_outputs[f"{model.name}_{tect_type_str}"] = tect_result_df
    return model_outputs, gm_df


def calc_residuals(
    gm_df: pd.DataFrame,
    model_outputs: dict[str, pd.DataFrame],
    output_dir: Path,
    ims: list[str] = None,
):
    """
    Calculate the residuals between the observed IMs and the model IMs
    :param gm_df: The filtered ground motion dataframe
    :param model_outputs: The model outputs with the given model and tect_type pairing as the dictionary key
    :param output_dir: The path to the output directory
    :param ims: The IMs to calculate the residuals for
    """
    # Make the residual output directory
    residual_dir = output_dir / "residuals"
    residual_dir.mkdir(exist_ok=True, parents=True)

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

        # Get the non nan mask
        non_nan_mask = res_df.notna()

        # Run MER
        print(f"Running MER for {model}")
        event_res_df, site_res_df, rem_res_df, bias_std_df = run_mera(
            res_df, list(im_columns), "evid", "sta", mask=non_nan_mask
        )

        # Sort and save each of the residual dataframes to a csv
        print(f"Writing residual results for {model} in {residual_dir}")
        sort_period_columns(event_res_df).to_csv(
            residual_dir / f"{model}_event.csv", index=False
        )
        sort_period_columns(site_res_df).to_csv(
            residual_dir / f"{model}_site.csv", index=False
        )
        sort_period_columns(rem_res_df).to_csv(
            residual_dir / f"{model}_rem.csv", index=False
        )
        sort_period_columns(bias_std_df.T).T.to_csv(
            residual_dir / f"{model}_bias_std.csv", index=False
        )


def load_args():
    """Load command line arguments for nzshm_residuals.

    Returns
    -------
    ArgumentParser
        An argument parser for nzshm_residuals script.
    """
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
        type=list[str],
        default=DEFAULT_MODELS,
        help="List of models to use. Defaults to all models used in thr NZSHM",
    )
    parser.add_argument(
        "--ims",
        type=list[str],
        default=DEFAULT_IMS,
        help="List of IMs to calculate. Defaults to PGA and pSA",
    )
    parser.add_argument(
        "--periods",
        type=list[float],
        default=None,
        help="List of periods to calculate. Defaults to all periods in the gm csv",
    )
    parser.add_argument(
        "--period_specific_ffp",
        type=Path,
        default=None,
        help="The path to the period specific txt file",
    )

    args = parser.parse_args()
    return args


def main():
    """Main script for nzshm_residuals."""
    args = load_args()
    model_outputs, gm_df = calc_empirical(
        args.gm_csv,
        args.backarc_json_ffp,
        args.output_dir,
        args.models,
        args.ims,
        args.periods,
        args.period_specific_ffp,
    )
    calc_residuals(gm_df, model_outputs, args.output_dir)


if __name__ == "__main__":
    main()
