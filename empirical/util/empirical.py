import math
import functools
import sys
from typing import List, Dict, Union
from pathlib import Path
import h5py
import yaml
import numpy as np
import pandas as pd
import multiprocessing as mp
from empirical.util.classdef import TectType

import IM_calculation.source_site_dist.src_site_dist as ssd
from IM_calculation.IM.im_calculation import DEFAULT_IMS
from qcore import (
    nhm,
    srf,
    formats,
    simulation_structure,
    utils,
    constants,
    archive_structure,
)
from empirical.util import classdef, openquake_wrapper_vectorized
# from cybershake_investigation import utils as ci_utils
# from cybershake_investigation import site_source

TECT_CLASS_MAPPING = {
    "Crustal": TectType.ACTIVE_SHALLOW,
    "Slab": TectType.SUBDUCTION_SLAB,
    "Interface": TectType.SUBDUCTION_INTERFACE,
    "Undetermined": TectType.ACTIVE_SHALLOW,
}


def get_site_source_data(flt: Union[nhm.NHMFault, Path], stations: np.ndarray):
    """
    Computes site-source data rrup, rjb, rx, ry
    for a single fault and a set of stations
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
    if isinstance(flt,nhm.NHMFault):
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
    model_config: dict, tect_type: classdef.TectType, im: str, component: str
):
    """
    Gets the appropriate model based on the model config given the tect_type, im and component
    """
    model = None
    try:
        model = classdef.GMM[model_config[tect_type.name][im][component][0]]
    except KeyError:
        if component == constants.Components.crotd50.str_value:
            try:
                model = classdef.GMM[
                    model_config[tect_type.name][im][
                        constants.Components.cgeom.str_value
                    ][0]
                ]
            except KeyError:
                pass
    return model


def calc_empirical_simulation(
    cybershake_root: Path,
    fault_list_ffp: Path,
    nhm_ffp: Path,
    meta_config_ffp: Path,
    model_config_ffp: Path,
    output_dir: Path = None,
    component: str = constants.Components.crotd50.str_value,
    n_procs: int = 1,
):
    """
    Calculates empirical im_csv files for a cybershake run for each fault listed in the fault_list.
    Parameters
    ----------
    cybershake_root: Path
        Path to the cybershake root directory
    fault_list_ffp: Path
        Full file path to the list txt file containing all faults to compute for
    nhm_ffp: Path
        Full file path to the fault NHM txt file, typically NZ_FLTmodel_2010
    meta_config_ffp: Path
        Full file path to the meta config yaml file which contains the weightings for each model
        categorized by IM, TectType then Model
    model_config_ffp: Path
        Full file path to the model config yaml file which contains the models to use for
        the given TectType, IM, Component
    output_dir: Path (optional)
        Path to the output directory to place empirical values.
        Defaults to cybershake_root / Data / Empirical
    component: str (optional)
        Component to calculate results for, defualt RotD50
    n_procs: int (optional)
        Number of processes to use for computing
    """
    # Loading configs
    model_config = utils.load_yaml(model_config_ffp)
    # Using Full Loader for the meta config due to the python tuple pSA/PGA
    with open(meta_config_ffp) as f:
        meta_config = yaml.load(f, Loader=yaml.FullLoader)

    # Get IM Info
    root_params = utils.load_yaml(cybershake_root / "Runs" / "root_params.yaml")
    ims = root_params["ims"].get("im", None)
    if ims is None:
        ims = list(DEFAULT_IMS)

    # Load NHM and Fault Info
    nhm_data = nhm.load_nhm(str(nhm_ffp))
    faults = formats.load_fault_selection_file(fault_list_ffp)

    # Load Station Info
    stat_file = Path(root_params["stat_file"])
    ll_parent_stem = stat_file.parent / stat_file.stem
    vs30_df = formats.load_vs30_file(f"{ll_parent_stem}.vs30")
    z_df = formats.load_z_file(f"{ll_parent_stem}.z")
    z_df = z_df.rename({"z1p0": "z1pt0", "z2p5": "z2pt5"}, axis="columns")
    station_df = formats.load_station_file(stat_file)
    station_df = station_df.merge(vs30_df, left_index=True, right_index=True)
    station_df = station_df.merge(z_df, left_index=True, right_index=True)
    station_df["vs30measured"] = False

    if output_dir is None:
        output_dir = cybershake_root / "Data" / "Empirical"
    Path.mkdir(output_dir, exist_ok=True, parents=True)

    mp_create_rel_csv_pre_filled = functools.partial(
        _mp_create_rel_csvs_simulation,
        cybershake_root=cybershake_root,
        nhm_data=nhm_data,
        station_df=station_df,
        ims=ims,
        component=component,
        model_config=model_config,
        meta_config=meta_config,
        output_dir=output_dir,
    )

    with mp.Pool(n_procs) as p:
        p.starmap(mp_create_rel_csv_pre_filled, faults.items())


def _mp_create_rel_csvs_simulation(
    fault: str,
    n_rels: int,
    cybershake_root: Path,
    nhm_data: Dict,
    station_df: pd.DataFrame,
    ims: list,
    component: str,
    model_config: dict,
    meta_config: dict,
    output_dir: Path,
):
    """
    Function for multiprocessing the creation of all realisation csvs for a single fault.
    Under the simulation structure.
    Parameters
    ----------
    fault: str
        Fault name to compute the rel csvs for
    n_rels: int
        Int for how many realisations to compute for the given fault
    cybershake_root: Path
        Path to the cybershake root directory
    nhm_data: Dict
        NHM loaded data for all faults
    station_df: pd.DataFrame
        Dataframe containing the stations with lat, lon, vs30 and z values
    ims: list
        list of IM's to calculate
    component: str
        Component to calculate results for
    model_config: Dict
        Loaded model config yaml dict which contains the models to use for
        the given TectType, IM, Component
    meta_config: Dict
        loaded meta config yaml dict which contains the weightings for each model
        categorized by IM, TectType then Model
    output_dir: Path
        Output path to generate results in
    """
    # Params info
    fault_yaml_ffp = simulation_structure.get_fault_yaml_path(
        cybershake_root / "Runs", fault
    )
    fault_params = utils.load_yaml(fault_yaml_ffp)

    # Specific fault site info
    flt_ll_df = formats.load_station_file(fault_params["FD_STATLIST"])
    ll = flt_ll_df.values
    flt_sites_df = station_df.loc[flt_ll_df.index.values]

    # Get fault data
    nhm_flt_info = nhm_data[fault]
    nhm_flt_data = nhm_flt_to_df(nhm_flt_info)
    tect_type = classdef.TectType[nhm_flt_info.tectonic_type]

    # Site to source data
    site_source_df = get_site_source_data(nhm_flt_info, ll)
    site_source_df.index = flt_sites_df.index.values
    site_source_df = site_source_df.merge(
        flt_sites_df, left_index=True, right_index=True
    )

    output_flt_dir = output_dir / fault

    for n_rel in range(1, n_rels + 1):
        rel_name = simulation_structure.get_realisation_name(fault, n_rel)
        rel_csv = (
            Path(simulation_structure.get_srf_dir(cybershake_root, fault))
            / f"{rel_name}.csv"
        )
        im_csv = Path(
            simulation_structure.get_IM_csv_from_root(cybershake_root, rel_name)
        )
        im_csv_df = pd.read_csv(im_csv, index_col=0)
        periods, _ = ci_utils.get_period_values(im_csv_df)

        create_emp_rel_csv(
            rel_csv,
            periods,
            nhm_flt_data,
            site_source_df,
            ims,
            component,
            tect_type,
            model_config,
            meta_config,
            output_flt_dir,
        )

def get_tect_type_name(tect_type):
    found = None
    for key, val in TECT_CLASS_MAPPING.items():
        if tect_type == val:
            found = key
            break
    return found

def load_srf_info(srf_info, event_name):
    """Create fault parameters"""
    fault = {}
    f = h5py.File(srf_info, "r")
    attrs = f.attrs

    fault["mag"] = np.max(attrs["mag"])

    if "tect_type" in attrs:
        try:
            tect_type = TectType[attrs["tect_type"]]  # ok if attrs['tect_type'] is str
        except KeyError:  # bytes
            tect_type = TectType[attrs["tect_type"].decode("utf-8")]

    else:
        print("tect_type not found assuming 'ACTIVE_SHALLOW'")
        tect_type = TectType.ACTIVE_SHALLOW
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

def load_rel_csv(
        source_csv: Path,
        event_name
):

    required_columns = ["dtop","dbottom","dip","dhypo","magnitude","tect_type","rake","dip"]

    csv_info = pd.read_csv(source_csv)

    missing_cols = []
    for col in required_columns:
        if col not in csv_info.columns:
            missing_cols.append(col)

    if len(missing_cols) > 0:
        print(f"Error: Column {missing_cols} are not found. Consider using SRF .info file instead.")
        sys.exit()


    hypo_depth = (
            csv_info["dtop"] + math.sin(math.radians(csv_info["dip"])) * csv_info["dhypo"]
    )
    fault = {}
    fault["mag"] = csv_info.loc[0, "magnitude"]
    fault["tect_class"] = csv_info["tect_type"]
    fault["z_tor"] = csv_info["dtop"]
    fault["z_bor"] = csv_info["dbottom"]
    fault["rake"] = csv_info["rake"]
    fault["dip"] = csv_info["dip"]
    fault["depth"] = hypo_depth.values[0]

    return pd.Series(fault, name=event_name)


def create_emp_rel_csv(
    source_csv: Path,
    periods: List[str],
    nhm_flt_data: pd.DataFrame,
    filtered_site_source: pd.DataFrame,
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
    source_csv: Path
        Path to the Realisation Source csv file
    periods: List[str]
        List of period columns to calculate as strings
    nhm_flt_data: pd.DataFrame
        NHM loaded data for this specific fault
    filtered_site_source: pd.DataFrame
        The site_source dataframe with rrup data for specific stations for this fault
    ims: list
        list of IM's to calculate
    component: str
        Component to calculate results for
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
    rel_name = source_csv.stem

    csv_info = pd.read_csv(source_csv)
    nhm_flt_data.loc[nhm_flt_data.index.values[0], "mag"] = csv_info.loc[0, "magnitude"]

    hypo_depth = (
        csv_info["dtop"] + math.sin(math.radians(csv_info["dip"])) * csv_info["dhypo"]
    )

    filtered_site_source["hypo_depth"] = hypo_depth.values[0]
    filtered_site_source["zbot"] = csv_info["dbottom"]

    locations_df = nhm_flt_data.join(filtered_site_source, how="outer").ffill().iloc[1:]

    create_emp_df(rel_name, periods, locations_df, ims, component, tect_type, model_config, meta_config, output_flt_dir)

def create_emp_df(
        rel_name,
        periods: List[str],
        locations_df: pd.DataFrame,
        ims: list,
        component: str,
        tect_type: classdef.TectType,
        model_config: dict,
        meta_config: dict,
        output_flt_dir: Path,
    ):

    im_df_list = []
    for im in ims:
        if im == "AI":
            # Ignoring AI IM due to non vectorised GMM CB_12
            continue

        model = get_model(model_config, tect_type, im, component)
        if model is None:
            continue

        im_meta_config = None
        if meta_config is not None:
            for meta_key in meta_config.keys():
                if im in meta_key:
                    im_meta_config = meta_config[meta_key][tect_type.name]

        # Volcanic types are very similar to Active shallow - Brendon
        if tect_type == classdef.TectType.VOLCANIC:
            tect_type = classdef.TectType.ACTIVE_SHALLOW

        im_df = openquake_wrapper_vectorized.oq_run(
            model,
            classdef.TectType.ACTIVE_SHALLOW
            if tect_type != classdef.TectType.ACTIVE_SHALLOW
            and model.name
            in (
                "CB_10",
                "CB_12",
                "AS_16",
            )
            else tect_type,
            locations_df,
            im,
            periods if im == "pSA" else None,
            meta_config=im_meta_config,
            convert_mean=np.exp,
        )
        im_df_list.append(im_df)

    result_df = pd.concat(im_df_list, axis=1)

    result_df.insert(0, "component", component)
    result_df.insert(0, "station", locations_df.index.values)

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
            "rupture_name",
            "mag",
            "dip",
            "rake",
            "dbot",
            "ztor",
            "tect_type",
            "recurrance_rate",
        ],
    ).reset_index()

def _mp_create_rel_csvs_archive(
    fault_dir: Path,
    nhm_data: Dict,
    site_source_df: pd.DataFrame,
    fault_id_df: pd.DataFrame,
    station_df: pd.DataFrame,
    ims: list,
    component: str,
    model_config: dict,
    meta_config: dict,
    output: Path,
):
    """
    Function for multiprocessing the creation of all realisation csvs for a single fault.
    Under the archive structure.
    Parameters
    ----------
    fault_dir: Path
        Path to the fault directory of the archive IM / Source data
    nhm_data: Dict
        NHM loaded data for all faults
    site_source_df: pd.DataFrame
        The site_source dataframe with rrup data for all stations per fault
    fault_id_df: pd.DataFrame
        Dataframe containing the fault names to fault id
    station_df: pd.DataFrame
        Dataframe containing the stations with lat, lon, vs30 and z values
    ims: list
        list of IM's to calculate
    component: str
        Component to calculate results for
    model_config: Dict
        Loaded model config yaml dict which contains the models to use for
        the given TectType, IM, Component
    meta_config: Dict
        loaded meta config yaml dict which contains the weightings for each model
        categorized by IM, TectType then Model
    output: Path
        Output path to generate results in
    """
    fault_name = fault_dir.stem

    # Get Fault data
    nhm_flt_info = nhm_data[fault_name]
    nhm_flt_data = nhm_flt_to_df(nhm_flt_info)
    tect_type = classdef.TectType[nhm_flt_info.tectonic_type]

    # Get site source data if available
    if site_source_df is not None:
        fault_id = fault_id_df[fault_id_df["fault_name"] == fault_name].index.values[0]
        flt_site_source = site_source_df[site_source_df["fault_id"] == fault_id]
        flt_site_source = flt_site_source.merge(
            station_df, left_index=True, right_index=True
        )

    # Generate list of rel csv Path files
    median_file = (
        archive_structure.get_fault_source_dir(fault_dir) / f"{fault_name}.csv"
    )

    # Check the median file exists
    if not median_file.exists():
        print(f"No median source file found for fault {fault_name}!")
    else:
        # Ensure sites and periods correlate to what was produced in the IM csvs
        # Using the first realisation file
        im_csv = (
            archive_structure.get_fault_im_dir(fault_dir) / f"{fault_name}_REL01.csv"
        )
        im_csv_df = pd.read_csv(im_csv, index_col=0)
        # Extract rows that have the given component
        im_csv_df = im_csv_df.loc[im_csv_df["component"] == component]
        sites = im_csv_df.index.values
        periods, _ = ci_utils.get_period_values(im_csv_df)

        if site_source_df is None:
            # Compute all Site to source data
            flt_stations = np.asarray(station_df.loc[sites].iloc[:, :2])
            rrup_df = get_site_source_data(nhm_flt_info, flt_stations)
            rrup_df.index = sites
            filtered_site_source = rrup_df.merge(
                station_df.loc[sites], left_index=True, right_index=True
            )
        else:
            # Ensure rrup values are available and flt_site_source is filtered to the correct sites
            try:
                filtered_site_source = flt_site_source.loc[sites]
            except KeyError:
                print(
                    "Could not find all sites rrup information, computing some sites manually"
                )
                # Calculating the different sets of sites in and out of the flt_site_source
                flt_site_source_sites = list(
                    set(flt_site_source.index.values).intersection(set(sites))
                )
                sites_not_in_rdf = list(
                    set(sites).difference(set(flt_site_source.index.values))
                )
                extra_stations = np.asarray(
                    station_df.loc[sites_not_in_rdf].iloc[:, :2]
                )

                # Calculate site source data
                rrup_df = get_site_source_data(nhm_flt_info, extra_stations)

                # Combine dataframes
                rrup_df.index = sites_not_in_rdf
                extra_stations_df = rrup_df.merge(
                    station_df.loc[sites_not_in_rdf], left_index=True, right_index=True
                )
                filtered_site_source = pd.concat(
                    [flt_site_source.loc[flt_site_source_sites], extra_stations_df]
                )

        # Iterate over list of paths and create the empirical realisation csvs
        output_flt_dir = output / fault_name
        create_emp_rel_csv(
            median_file,
            periods,
            nhm_flt_data,
            filtered_site_source,
            ims,
            component,
            tect_type,
            model_config,
            meta_config,
            output_flt_dir,
        )


def calc_empirical_archive(
    archive_dir: Path,
    ll_ffp: str,
    vs30_ffp: str,
    z_ffp: str,
    nhm_ffp: Path,
    meta_config_ffp: Path,
    model_config_ffp: Path,
    output: Path,
    site_source_db_ffp: str = None,
    component: str = constants.Components.crotd50.str_value,
    n_procs: int = 1,
):
    """
    Calculates empirical im_csv files for a cybershake run for each fault listed in the fault_list.
    Parameters
    ----------
    archive_dir: Path
        Path to the archived cybershake root directory that contains at least the IM and Source files
    ll_ffp: str
        The file listing the stations with lat and lon values
    vs30_ffp: str
        The file listing the stations with their vs30 values
    z_ffp: str
        The file listing stations with their z values
    nhm_ffp: Path
        Full file path to the fault NHM txt file, typically NZ_FLTmodel_2010
    meta_config_ffp: Path
        Full file path to the meta config yaml file which contains the weightings for each model
        categorized by IM, TectType then Model
    model_config_ffp: Path
        Full file path to the model config yaml file which contains the models to use for
        the given TectType, IM, Component
    output: Path
        Output path to generate results in
    site_source_db_ffp: str (optional)
        The flt_site_source_db path to reduce calculations
        for site source distances
    component: str (optional)
        Component to calculate results for, default RotD50
    n_procs: int (optional)
        Number of processes to use for computing
    """
    # Loading configs
    model_config = utils.load_yaml(model_config_ffp)
    # Using Full Loader for the meta config due to the python tuple pSA/PGA
    with open(meta_config_ffp) as f:
        meta_config = yaml.load(f, Loader=yaml.FullLoader)

    ims = list(DEFAULT_IMS)
    nhm_data = nhm.load_nhm(str(nhm_ffp))

    # Load Station Info
    vs30_df = formats.load_vs30_file(vs30_ffp)
    z_df = formats.load_z_file(z_ffp)
    z_df = z_df.rename({"z1p0": "z1pt0", "z2p5": "z2pt5"}, axis="columns")
    station_df = formats.load_station_file(ll_ffp)
    station_df = station_df.merge(vs30_df, left_index=True, right_index=True)
    station_df = station_df.merge(z_df, left_index=True, right_index=True)
    station_df["vs30measured"] = False
    station_df["vs30measured"] = station_df["vs30measured"].astype(bool)

    site_source_df, fault_ids = None, None
    if site_source_db_ffp is not None:
        with site_source.SiteSourceDB(
            site_source_db_ffp, writeable=False
        ) as distance_store:
            site_source_df, fault_ids = distance_store.get_all_rrup_data()

    mp_create_rel_csv_pre_filled = functools.partial(
        _mp_create_rel_csvs_archive,
        nhm_data=nhm_data,
        site_source_df=site_source_df,
        fault_id_df=fault_ids,
        station_df=station_df,
        ims=ims,
        component=component,
        model_config=model_config,
        meta_config=meta_config,
        output=output,
    )

    with mp.Pool(n_procs) as p:
        p.map(
            mp_create_rel_csv_pre_filled,
            [flt_dir for flt_dir in archive_dir.iterdir() if flt_dir.is_dir()],
        )
