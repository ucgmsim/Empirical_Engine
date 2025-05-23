from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd

import qcore.src_site_dist as ssd
from empirical.util import classdef
from empirical.util import openquake_wrapper_vectorized as oq_wrapper
from qcore import nhm, srf

TECT_CLASS_MAPPING = {
    "Crustal": classdef.TectType.ACTIVE_SHALLOW,
    "Slab": classdef.TectType.SUBDUCTION_SLAB,
    "Interface": classdef.TectType.SUBDUCTION_INTERFACE,
    "Undetermined": classdef.TectType.ACTIVE_SHALLOW,
}
REVERSE_TECT_CLASS_MAPPING = {value: key for key, value in TECT_CLASS_MAPPING.items()}

NZ_GMDB_SOURCE_COLUMNS = [
    "mag",
    "tect_class",
    "z_tor",
    "z_bor",
    "rake",
    "dip",
    "depth",
    "ev_depth",
    "r_rup",
    "r_jb",
    "r_x",
    "r_y",
    "Vs30",
    "Z1.0",
    "Z2.5",
] 

OQ_RUPTURE_COLUMNS = [
    "mag",
    "tect_class",
    "ztor",
    "zbot",
    "rake",
    "dip",
    "hypo_depth",
    "hypo_depth",
    "rrup",
    "rjb",
    "rx",
    "ry0",
    "vs30",
    "z1pt0",
    "z2pt5",
]

NZGMDB_OQ_COL_MAPPING = dict(zip(NZ_GMDB_SOURCE_COLUMNS, OQ_RUPTURE_COLUMNS))


def get_site_source_data(flt: Union[nhm.NHMFault, Path], stations: np.ndarray):
    """
    Computes site-source data rrup, rjb, rx, ry
    for a single fault and a set of stations

    Parameters
    ----------
    flt : nhm.NHMFault or Path
    stations : np.ndarray
        (eg. [[lon1, lat1], [lon2, lat2], ...])

    Returns
    -------
    pd.DataFrame
        containing ["rrup", "rjb", "rx", "ry"] for a single fault and a set of stations
    """
    # Add depth for the stations (hardcoded to 0)
    filtered_station_np = np.concatenate(
        (
            stations,
            np.zeros((stations.shape[0], 1), dtype=stations.dtype),
        ),
        axis=1,
    )

    # Get fault header and point data
    if isinstance(flt, nhm.NHMFault):
        plane_info, flt_points = nhm.get_fault_header_points(flt)
    else:
        flt_points = srf.read_srf_points(str(flt))
        plane_info = srf.read_header(str(flt), idx=True)

    # Calculate source to site distances
    r_rup, r_jb = ssd.calc_rrup_rjb(flt_points, filtered_station_np)
    r_x, r_y = ssd.calc_rx_ry(flt_points, plane_info, filtered_station_np)

    rrup_df = pd.DataFrame(
        np.vstack([r_rup, r_jb, r_x, r_y]).T, columns=["rrup", "rjb", "rx", "ry"]
    )

    return rrup_df


def get_model(
    model_config: dict,
    tect_type: classdef.TectType,
    im: str,
    component: str,
):
    """
    Gets the appropriate model based on the model config given the tect_type, im and component

    Parameters
    ----------
    model_config: dict
        Contains the empirical model to be used for a given tect_type, im and component
    tect_type:  classdef.TectType
        Tectonic type of the fault
    im : str
        Intensity Measure to calculate
    component:  str
        Component to calculate results for

    Returns
    -------
    classdef.GMM
        The GMM model to be used. Returns None if no model is found

    """

    try:
        return classdef.GMM[model_config[tect_type.name][im][component][0]]
    except KeyError:
        pass


def load_srf_info(srf_info: Path, event_name: str):
    """Load srf_info file in HDF5 format and return a pandas Series with the fault parameters

    Parameters
    ----------
    srf_info: Path
        Path to the srf_info file
    event_name: str
        Name of the event

    Returns
    -------
    pd.Series : containing fault parameters
    """

    fault = {}

    with h5py.File(srf_info, "r") as f:
        attrs = dict(f.attrs)

    fault["mag"] = np.max(attrs["mag"])

    if "tect_type" in attrs:
        tect_type_key = attrs[
            "tect_type"
        ]  # tect_type is sometimes stored as bytes, sometimes as string
        tect_type = classdef.TectType[
            (
                tect_type_key
                if isinstance(tect_type_key, str)
                else tect_type_key.decode("utf-8")
            )
        ]

    else:
        print("INFO: tect_type not found.  Default 'ACTIVE_SHALLOW' is used.")
        tect_type = classdef.TectType.ACTIVE_SHALLOW
    fault["tect_class"] = REVERSE_TECT_CLASS_MAPPING.get(tect_type)

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
        raise ValueError("unexpected rake value")

    dip = attrs["dip"]
    if np.max(dip) == np.min(dip):
        fault["dip"] = np.min(dip)
    else:
        raise ValueError("unexpected dip value")

    fault["depth"] = attrs["hdepth"]

    return pd.Series(fault, name=event_name)


def load_rel_csv(source_csv: Path, event_name: str):
    """
    Load realisation csv file and return a pandas Series with the fault parameters

    Parameters
    ----------
    source_csv: Path
        Path to the source csv file
    event_name: str
        Name of the event

    Returns
    -------
    pd.Series : containing fault parameters

    """
    rel_csv_columns = [  # columns in the source csv file
        "dtop",
        "dbottom",
        "dip",
        "dhypo",
        "magnitude",
        "tect_type",
        "rake",
        "dip",
    ]

    csv_info = pd.read_csv(source_csv)
    missing_cols = set(rel_csv_columns) - set(csv_info.columns)

    if len(missing_cols) > 0:
        raise KeyError(
            f"Error: Column {missing_cols} are not found. Consider using SRF .info file instead."
        )

    rel_info = csv_info.loc[0]

    hypo_depth = (
        rel_info["dtop"] + np.sin(np.radians(rel_info["dip"])) * rel_info["dhypo"]
    )
    fault = {}

    fault["mag"] = rel_info["magnitude"]
    fault["tect_class"] = rel_info["tect_type"]
    fault["z_tor"] = rel_info["dtop"]
    fault["z_bor"] = rel_info["dbottom"]
    fault["rake"] = rel_info["rake"]
    fault["dip"] = rel_info["dip"]
    fault["depth"] = hypo_depth

    return pd.Series(fault, name=event_name)


def create_emp_rel_csv(
    rel_name: str,
    periods: list[str],
    rupture_df: pd.DataFrame,
    ims: list,
    component: str,
    tect_type: classdef.TectType,
    model_config: dict,
    meta_config: dict,
    output_flt_dir: Path,
):
    """
    Calculates and saves a single empirical realisation csv file based on IM's specified.

    Parameters
    ----------
    rel_name: str
        Name of the realisation
    periods: list
        List of periods to calculate pSA for
    rupture_df: pd.DataFrame
        Dataframe containing rupture data that should include all columns required for OpenQuake calculations
    ims: list
        list of IM's to calculate
    component: str
        component to calculate results for
    tect_type: classdef.TectType
        Tectonic Type of the fault
    model_config: Dict
        Loaded model config yaml dict which contains the models to use for
        the given TectType, IM, Component
    meta_config: Dict
        loaded meta config yaml dict which contains the weightings for each model
        categorized by IM, TectType then Model
    output_flt_dir: Path
        Output path to generate results in
    """
    im_df_list = []
    for im in ims:
        if im == "AI":
            # Ignoring AI IM due to non vectorised GMM CB_12
            continue

        # Volcanic types are very similar to Active shallow - Brendon
        if tect_type == classdef.TectType.VOLCANIC:

            print(
                f"INFO: ({rel_name},{tect_type.name},{im},{component}): Will be treated as Active Shallow"
            )
            tect_type = classdef.TectType.ACTIVE_SHALLOW

        model = get_model(model_config, tect_type, im, component)

        if model is None:
            print(
                f"WARNING: ({rel_name},{tect_type.name},{im},{component}): No model found, to be skipped."
            )
            continue

        # CB_10, CB_12 and AS_16 are for AI, CAV, Ds575, Ds595, but only available for Active Shallow
        # So we consider them as Active Shallow even if the tect_type is not.
        # https: // wiki.canterbury.ac.nz / display / QuakeCore / Empirical + Engine
        if tect_type != classdef.TectType.ACTIVE_SHALLOW and model.name in (
            "CB_10",
            "CB_12",
            "AS_16",
        ):
            print(
                f"INFO: ({rel_name},{tect_type.name},{im},{component}): Will be treated as Active Shallow using {model.name}"
            )
            tect_type = classdef.TectType.ACTIVE_SHALLOW

        im_meta_config = None
        if meta_config is not None:
            for meta_key in meta_config:
                if im in meta_key:
                    im_meta_config = meta_config[meta_key][tect_type.name]

        im_df = oq_wrapper.oq_run(
            model,
            tect_type,
            rupture_df,
            im,
            periods=periods if im == "pSA" else None,
            meta_config=im_meta_config,
        )
        im_df_list.append(im_df)

    result_df = pd.concat(im_df_list, axis=1)

    result_df.insert(0, "component", component)
    result_df.insert(0, "station", rupture_df.index.values)

    output_flt_dir.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(output_flt_dir / f"{rel_name}.csv", index=False)


def nhm_flt_to_df(nhm_flt: nhm.NHMFault):
    """
    Custom nhm fault to dataframe to do renaming
    and ensure only one load of the NHM

    Parameters
    ----------
    nhm_flt : nhm.NHMFault
        NHM fault object

    Returns
    -------
    pd.DataFrame : containing fault parameters
    """
    rupture_dict = {
        nhm_flt.name: [
            nhm_flt.name,
            nhm_flt.mw,
            nhm_flt.dip,
            nhm_flt.rake,
            nhm_flt.dbottom,
            nhm_flt.dtop,
            nhm_flt.tectonic_type,
            nhm_flt.recur_int_median,
        ]
    }
    return pd.DataFrame.from_dict(
        rupture_dict,
        orient="index",
        columns=[
            "fault_name",
            "mag",
            "dip",
            "rake",
            "dbot",
            "ztor",
            "tect_class",
            "recurrance_rate",
        ],
    ).reset_index()
