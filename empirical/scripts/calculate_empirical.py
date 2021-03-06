#!/usr/bin/env python3
"""Script to calculate IMs for empirical models.

Produces one .csv for each IM containing all specified sites.
"""
import os
import argparse
import h5py
import numpy as np
import pandas as pd

from empirical.util import empirical_factory, classdef
from empirical.util.classdef import Site, Fault, TectType
from empirical.GMM_models.Burks_Baker_2013_iesdr import _STRENGTH_REDUCTION_FACTORS

from qcore import constants
from qcore.utils import setup_dir
from qcore.im import order_im_cols_df

IM_LIST = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "pSA"]
MULTI_VALUE_IMS = ("pSA", "IESDR")
PSA_IM_NAME = "pSA"
STATION_COL_NAME = "station"
COMPONENT_COL_NAME = "component"


def create_fault_parameters(srf_info):
    """Create fault parameters"""
    fault = Fault()
    f = h5py.File(srf_info, "r")
    attrs = f.attrs
    dip = attrs["dip"]
    if np.max(dip) == np.min(dip):
        fault.dip = np.min(dip)
    else:
        print("unexpected dip value")
        exit()
    fault.Mw = np.max(attrs["mag"])
    rake = attrs["rake"]
    if np.max(rake) == np.min(rake):
        fault.rake = np.min(rake)
    else:
        print("unexpected rake value")
        exit()
    fault.Mw = attrs["mag"]
    if "dtop" in attrs:
        fault.ztor = np.min(attrs["dtop"])
    else:
        fault.ztor = attrs["hdepth"]
    if "dbottom" in attrs:
        fault.zbot = np.min(attrs["dbottom"])
    else:
        fault.zbot = attrs["hdepth"]
    if "tect_type" in attrs:
        try:
            fault.tect_type = TectType[
                attrs["tect_type"]
            ]  # ok if attrs['tect_type'] is str
        except KeyError:  # bytes
            fault.tect_type = TectType[attrs["tect_type"].decode("utf-8")]
    else:
        print("tect_type not found assuming 'ACTIVE_SHALLOW'")
        fault.tect_type = TectType.ACTIVE_SHALLOW
    fault.hdepth = attrs["hdepth"]
    if "width" in attrs:
        fault.width = np.max(attrs["width"])
    else:
        fault.width = 0
    return fault


def read_rrup_file(rrup_file):
    """Read rupture(?) file"""
    rrups = dict()
    with open(rrup_file) as f:
        next(f)
        for line in f:
            station, __, __, rrup, rjbs, rx, ry = line.rstrip().split(",")
            rrup = float(rrup)
            rjbs = float(rjbs)
            if rx == "X":
                rx = None
                ry = None
            else:
                rx = float(rx)
                ry = float(ry)
            values = (rrup, rjbs, rx, ry)
            rrups[station] = values
    return rrups


def read_vs30_file(vs30_file):
    """Read vs30 file"""
    values = dict()
    with open(vs30_file) as f:
        for line in f:
            station, value = line.split()
            values[station] = float(value)
    return values


def create_site_parameters(
    rrup_file, vs30_file, stations=None, vs30_default=500, max_distance=None
):
    """Create site parameters"""
    rrups = read_rrup_file(rrup_file)
    vs30_values = read_vs30_file(vs30_file)
    sites = list()
    if max_distance is None:
        max_distance = float("inf")
    if stations is None:
        stations = list(rrups.keys())
    for station in stations:
        rrup, rjbs, rx, ry = rrups[station]
        if rrup < max_distance:
            site = Site()
            site.name = station
            if station in vs30_values:
                site.vs30 = vs30_values[station]
            else:
                site.vs30 = vs30_default
            site.Rrup = rrup
            site.Rjb = rjbs
            site.Rx = rx
            site.Ry = ry
            sites.append(site)
    return sites


def calculate_empirical(
    identifier,
    srf_info,
    output_dir,
    config_file,
    stations,
    vs30_file,
    vs30_default,
    ims,
    rupture_distance,
    max_rupture_distance,
    period,
    extended_period,
    components,
):
    """Calculate empirical intensity measures"""

    # Fault & Site parameters
    fault = create_fault_parameters(srf_info)

    sites = create_site_parameters(
        rupture_distance,
        vs30_file,
        stations=stations,
        vs30_default=vs30_default,
        max_distance=max_rupture_distance,
    )

    if extended_period:
        period = np.unique(np.append(period, constants.EXT_PERIOD))

    tect_type_model_dict = empirical_factory.read_model_dict(config_file)
    station_names = [site.name for site in sites] if stations is None else stations

    for im in ims:
        for cur_gmm, component in empirical_factory.determine_all_gmm(
            fault, im, tect_type_model_dict, components
        ):
            # File & column names
            cur_filename = "{}_{}_{}.csv".format(identifier, cur_gmm.name, im)
            cur_cols = []
            if im in MULTI_VALUE_IMS:
                if im == "IESDR":
                    for p in period:
                        for r in _STRENGTH_REDUCTION_FACTORS:
                            cur_cols.append("IESDR_{}_r_{}".format(p, r))
                            cur_cols.append("IESDR_{}_r_{}_sigma".format(p, r))
                else:
                    for p in period:
                        cur_cols.append("{}_{}".format(im, p))
                        cur_cols.append("{}_{}_sigma".format(im, p))
            else:
                cur_cols.append(im)
                cur_cols.append("{}_sigma".format(im))

            # Get & save the data
            cur_data = np.zeros((len(sites), len(cur_cols)), dtype=np.float)
            for ix, site in enumerate(sites):
                values = empirical_factory.compute_gmm(
                    fault, site, cur_gmm, im, period if im in MULTI_VALUE_IMS else None
                )
                if im in MULTI_VALUE_IMS:
                    cur_data[ix, :] = np.ravel(
                        [
                            [im_value, total_sigma]
                            for im_value, (total_sigma, *_) in values
                        ]
                    )
                else:
                    cur_data[ix, :] = [values[0], values[1][0]]

            df = pd.DataFrame(columns=cur_cols, data=cur_data)
            df[STATION_COL_NAME] = station_names
            df[COMPONENT_COL_NAME] = component.str_value

            # Correct column order
            df = order_im_cols_df(df)

            df.to_csv(os.path.join(output_dir, cur_filename), index=False)


def load_args():
    parser = argparse.ArgumentParser(
        description="Script to calculate IMs for empirical models."
        "Produces one .csv for each IM containing "
        "all specified sites."
    )
    parser.add_argument(
        "--vs30_file",
        "-v",
        required=True,
        help="vs30 file. Default value is 250 if" " station/file not present",
    )
    parser.add_argument(
        "-r",
        "--rupture_distance",
        required=True,
        help="Path to the rupture distance csv file",
    )
    parser.add_argument(
        "-srf", "--srf_info", help="Path to srf-info file", required=True
    )
    parser.add_argument(
        "--vs30_default",
        default=classdef.VS30_DEFAULT,
        help="Sets the default value for the vs30",
    )
    parser.add_argument(
        "-s",
        "--stations",
        nargs="+",
        help="List of stations to calculate empiricals for",
    )
    parser.add_argument(
        "-rm",
        "--max_rupture_distance",
        help="Only calculate empiricals for stations "
        "that are within X distance to rupture",
    )
    parser.add_argument("-i", "--identifier", help="run-name for run")
    parser.add_argument(
        "-c",
        "--config",
        help="configuration file to " "select which model is being used",
    )
    parser.add_argument(
        "-e",
        "--extended_period",
        action="store_true",
        help="Indicate the use of extended(100) pSA periods",
    )
    parser.add_argument(
        "-p",
        "--period",
        nargs="+",
        default=constants.DEFAULT_PSA_PERIODS,
        type=float,
        help="pSA period(s) separated by a " "space. eg: 0.02 0.05 0.1.",
    )
    parser.add_argument(
        "-m",
        "--im",
        nargs="+",
        default=IM_LIST,
        help="Intensity measure(s) separated by a "
        "space(if more than one). eg: PGV PGA CAV.",
    )
    # TODO: Put common argparse arguments between IM_calc and empirical in shared file
    parser.add_argument(
        "-comp",
        "--components",
        nargs="+",
        choices=list(constants.Components.iterate_str_values()),
        default=[constants.Components.cgeom.str_value],
        help="Please provide the velocity/acc component(s) you want to calculate eg.geom."
        " Available compoents are: {} components. Default is all components".format(
            ",".join(constants.Components.iterate_str_values())
        ),
    )
    parser.add_argument("output", help="output directory")
    args = parser.parse_args()
    return args


def main():
    args = load_args()
    setup_dir(args.output)
    calculate_empirical(
        args.identifier,
        args.srf_info,
        args.output,
        args.config,
        args.stations,
        args.vs30_file,
        args.vs30_default,
        args.im,
        args.rupture_distance,
        args.max_rupture_distance,
        args.period,
        args.extended_period,
        args.components,
    )


if __name__ == "__main__":
    main()
