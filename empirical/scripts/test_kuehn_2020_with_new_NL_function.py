from pathlib import Path
from typing import List, Dict
import argparse
import json
import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import geojson
from turfpy.measurement import points_within_polygon

from empirical.util.classdef import TectType, GMM
from empirical.util import openquake_wrapper_vectorized
# from mera.mera_pymer4 import run_mera

#change plotting settings
plt.style.use('classic')
plt.rcParams["font.family"] = "Times New Roman"

# create a list of default IMs to calculate
# DEFAULT_IMS = ["PGA", "pSA"]
DEFAULT_IMS = ["PGA"]
# DEFAULT_IMS = ["pSA"]

# create a list of GMM to loop through for predictions
# DEFAULT_MODELS = [
#     "S_22",
#     "A_22",
#     "ASK_14",
#     "CY_14",
#     "BSSA_14",
#     "CB_14",
#     "Br_13",
#     "AG_20",
#     "P_21",
#     "K_20",
# ]

# DEFAULT_MODELS = [
#     "K_20",
#     "K_20_new_NL"
# ]

DEFAULT_MODELS = [
    "K_20"
]

DEFAULT_MODELS = [
    "K_20_new_NL"
]

# DEFAULT_MODELS = [
#     "AG_20"
# ]
#
DEFAULT_MODELS = [
    "AG_20_new_NL"
]

TECT_CLASS_MAP = {
    "Crustal": TectType.ACTIVE_SHALLOW,
    "Interface": TectType.SUBDUCTION_INTERFACE,
    "Slab": TectType.SUBDUCTION_SLAB,
    "Outer-rise": None,
    "Undetermined": None,
}

TECT_CLASS_MAP_REV = {v: k for k, v in TECT_CLASS_MAP.items()}

# Create the rupture dataframe for open quake
rupture_df = pd.DataFrame(
    {
        "mag": [8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5],
        "ztor": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        "tect_type": [TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_INTERFACE,\
                      TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_INTERFACE,\
                      TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_INTERFACE,\
                      TectType.SUBDUCTION_INTERFACE],
        "rrup": [1,5,10,20,30,50,100, 300, 500, 1000],
        # "vs30": [760, 760, 760, 760, 760, 760, 760, 760, 760, 760],
        "vs30": [225, 225, 225, 225, 225, 225, 225, 225, 225, 225],
        "z1pt0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
)


def calc_empirical(rupture_df: pd.DataFrame,
    output_dir: Path,
    models: List[str] = None,
    ims: List[str] = None,
    periods: List[float] = None
):
    """
    Calculate the empirical IMs for the given gm csv and output the results to the given output directory.
    :param gm_csv: The path to the csv containing the ground motion data
    :param site_csv: The path to the csv containing the site data
    :param backarc_json_ffp: The path to the backarc json file
    :param output_dir: The path to the output directory
    :param models: The list of models to use
    :param ims: The list of IMs to calculate
    :param periods: The list of periods to calculate
    :param period_specific_ffp: The path to the period specific file
    :return: Dict of IM DataFrames per model and filtered gm csv as a dataframe
    """

    # Get models and IM's if needed
    if models is None:
        models = DEFAULT_MODELS
    if ims is None:
        ims = DEFAULT_IMS


    # Make the model output directory
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    # Calculate im csvs for each of the GMMs / TectTypes using the data in the gm_df
    model_outputs = dict()
    for str_model in models:
        model = GMM[str_model]
        # Get the tect types for the model
        tect_types = openquake_wrapper_vectorized.OQ_MODELS[model].keys()
        # for tect_type in tect_types:
        for tect_type in [TectType.SUBDUCTION_INTERFACE]:
            # Extract the rupture data information that has the same tect type
            tect_rup_df = rupture_df.loc[rupture_df["tect_type"] == tect_type]
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

                mean_cols = im_df.columns[im_df.columns.str.contains("mean")].values.astype(str)
                im_df[mean_cols] = np.exp(im_df[mean_cols])
                # im_df = np.exp(im_df.loc[:, im_df.columns.str.contains("mean")])
                im_df = im_df.rename(
                    {
                        col_name: col_name.rstrip("_mean")
                        for col_name in mean_cols
                    },
                    axis="columns",
                )
                im_df_list.append(im_df)

                '''
                # to save only the mean column without stddevs
                im_df = np.exp(im_df.loc[:, im_df.columns.str.contains("mean")])
                im_df = im_df.rename(
                    {
                        col_name: col_name.rstrip("_mean")
                        for col_name in im_df.columns.values
                    },
                    axis="columns",
                )
                im_df_list.append(im_df)
                '''

            # Combine all the different IM DataFrames into one
            tect_result_df = pd.concat(im_df_list, axis=1)
            # Insert back the gmid, evid and station to the dataframe
            tect_result_df.index = tect_rup_df.index
            tect_result_df.insert(0, "vs30", tect_rup_df["vs30"])
            tect_result_df.insert(0, "mag", tect_rup_df["mag"])
            tect_result_df.insert(0, "rrup", tect_rup_df["rrup"])

            # Get the tect type string
            tect_type_str = TECT_CLASS_MAP_REV[tect_type]


            # Save the tect type results to a csv
            print(
                f"Writing {model.name} {tect_type_str} to {model_dir / f'{model.name}_{tect_type_str}.csv'}"
            )
            tect_result_df.to_csv(
                model_dir / f"{model.name}_{tect_type_str}.csv")
            model_outputs[f"{model.name}_{tect_type_str}"] = tect_result_df
    return model_outputs

def load_args():
    parser = argparse.ArgumentParser(
        description="Calculate the empirical IMs for the given gm csv"
        "and output the results to the given output directory."
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
    parser.add_argument(
        "--period_specific_ffp",
        type=Path,
        default=None,
        help="The path to the period specific txt file",
    )

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    model_outputs = calc_empirical(rupture_df,
        args.output_dir,
        args.models,
        args.ims,
        args.periods,
    )


if __name__ == "__main__":
    main()

