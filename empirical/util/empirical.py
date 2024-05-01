from typing import List, Union
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from empirical.util import classdef, openquake_wrapper_vectorized as oq_wrapper

import IM_calculation.source_site_dist.src_site_dist as ssd

from qcore import nhm, srf, constants


TECT_CLASS_MAPPING = {
    "Crustal": classdef.TectType.ACTIVE_SHALLOW,
    "Slab": classdef.TectType.SUBDUCTION_SLAB,
    "Interface": classdef.TectType.SUBDUCTION_INTERFACE,
    "Undetermined": classdef.TectType.ACTIVE_SHALLOW,
}

NZ_GMDB_SOURCE_COLUMNS = [
    "mag",
    "tect_class",
    "z_tor",
    "z_bor",
    "rake",
    "dip",
    "depth",
]  # following NZ_GMDB_SOURCE column names

OQ_RUPTURE_COLUMNS = [
    "mag",
    "tect_class",
    "ztor",
    "zbot",
    "rake",
    "dip",
    "hypo_depth",
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

    Returns
    -------
    pd.DataFrame : containing ["rrup", "rjb", "rx", "ry"] for a single fault and a set of stations
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
    classdef.GMM : The GMM model to be used

    """
    model = None
    try:
        model = classdef.GMM[model_config[tect_type.name][im][component][0]]
    except KeyError:
        print(
            f"Warning: No model found for {tect_type.name}, {im}, {component} in model_config file."
        )
        raise
    return model


def get_tect_type_name(tect_type):
    """
    Get the tectonic type name from the classdef.TectType (eg. Crustal, Slab, Interface, Undetermined)
    ----------
    tect_type: classdef.TectType

    Returns
    -------
    str : Tectonic type name
    """
    found = None
    for key, val in TECT_CLASS_MAPPING.items():
        if tect_type == val:
            found = key
            break
    return found


def load_srf_info(srf_info, event_name):
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
    f = h5py.File(srf_info, "r")
    attrs = f.attrs

    fault["mag"] = np.max(attrs["mag"])

    if "tect_type" in attrs:
        try:
            tect_type = classdef.TectType[
                attrs["tect_type"]
            ]  # ok if attrs['tect_type'] is str
        except KeyError:  # bytes
            tect_type = classdef.TectType[attrs["tect_type"].decode("utf-8")]

    else:
        print("INFO: tect_type not found.  Default 'ACTIVE_SHALLOW' is used.")
        tect_type = classdef.TectType.ACTIVE_SHALLOW
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


def load_rel_csv(source_csv: Path, event_name: str):
    """

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
    rel_name,
    periods: List[str],
    rupture_df: pd.DataFrame,
    ims: list,
    component: str,
    tect_type: classdef.TectType,
    model_config: dict,
    meta_config: dict,
    output_flt_dir: Path,
    convert_mean=lambda x: x,
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
    convert_mean: function to convert mean values, eg) convert_mean=np.exp for log space. by default no conversion
    """
    im_df_list = []
    for im in ims:
        if im == "AI":
            # Ignoring AI IM due to non vectorised GMM CB_12
            continue

        # Volcanic types are very similar to Active shallow - Brendon
        if tect_type == classdef.TectType.VOLCANIC:
            tect_type = classdef.TectType.ACTIVE_SHALLOW

        model = get_model(model_config, tect_type, im, component)

        if tect_type != classdef.TectType.ACTIVE_SHALLOW and model.name in (
            "CB_10",
            "CB_12",
            "AS_16",
        ):
            tect_type = classdef.TectType.ACTIVE_SHALLOW

        if model is None:
            continue

        im_meta_config = None
        if meta_config is not None:
            for meta_key in meta_config.keys():
                if im in meta_key:
                    im_meta_config = meta_config[meta_key][tect_type.name]

        im_df = oq_wrapper.oq_run(
            model,
            tect_type,
            rupture_df,
            im,
            periods=periods if im == "pSA" else None,
            meta_config=im_meta_config,
            convert_mean=convert_mean,
        )
        im_df_list.append(im_df)

    result_df = pd.concat(im_df_list, axis=1)

    result_df.insert(0, "component", component)
    result_df.insert(0, "station", rupture_df.index.values)

    Path.mkdir(output_flt_dir, exist_ok=True, parents=True)
    result_df.to_csv(output_flt_dir / f"{rel_name}.csv", index=False)


def nhm_flt_to_df(nhm_flt: nhm.NHMFault):
    """
    Custom nhm fault to dataframe to do renaming
    and ensure only one load of the NHM
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


def get_oq_rupture_df(
    site_df: pd.DataFrame,
    rrup_df: pd.DataFrame,
    fault_df: pd.Series,
    rjb_max: float = None,
):
    """
    Get the rupture dataframe for OpenQuake calculations. Assumes site_df and rrup_df are in alignment (ie. same length, same order)

    Parameters
    ----------
    site_df: pd.DataFrame
        Dataframe containing site data (columns: (lon, lat, vs30, z1pt0, z2pt5))
    rrup_df: pd.DataFrame
        Dataframe containing site-source distances (columns: (rrup, rjb, rx, ry))
    fault_df: pd.Series
        Series containing fault data (columns: NZ_GMDB_SOURCE_COLUMNS)
    rjb_max: float
        Maximum Rjb distance to consider

    Returns
    -------
    oq_rupture_df: pd.DataFrame

    """
    oq_rupture_df = site_df.copy()
    oq_rupture_df["site"] = oq_rupture_df.index.values

    # Merge site_df and rrup_df
    oq_rupture_df = pd.concat([site_df, rrup_df.set_index(site_df.index)], axis=1)

    # Filter site_df to only include sites with rjb <= rjb_max
    if rjb_max is not None:
        oq_rupture_df = oq_rupture_df.loc[oq_rupture_df.rjb <= rjb_max]

    # Add event/fault data
    oq_rupture_df.loc[
        :,
        OQ_RUPTURE_COLUMNS,  # rename columns to follow OQ_RUPTURE_COLUMNS
    ] = fault_df[
        NZ_GMDB_SOURCE_COLUMNS
    ].values  # fault_df has NZ_GMDB_SOURCE_COLUMNS

    return oq_rupture_df
