#!/usr/bin/env python2
"""Script to calculate IMs for empirical models.

Produces one .csv for each IM containing all specified sites.
"""
import os
import argparse
import h5py
import numpy as np
import pandas as pd

import empirical_factory

from GMM_models.classdef import Site
from GMM_models.classdef import Fault
from GMM_models.classdef import TectType
from GMM_models import classdef

from qcore.utils import setup_dir
from qcore.im import order_im_cols_df

IM_LIST = ['PGA', 'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'pSA']
EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.), num=100,
                         base=10)
PERIOD = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0,
          7.5, 10.0]

PSA_IM_NAME = 'pSA'
STATION_COL_NAME = 'station'
COMPONENT_COL_NAME = 'component'


def create_fault_parameters(srf_info):
    """Create fault parameters"""
    fault = Fault()
    f = h5py.File(srf_info, 'r')
    attrs = f.attrs
    dip = attrs['dip']
    if np.max(dip) == np.min(dip):
        fault.dip = np.min(dip)
    else:
        print("unexpected dip value")
        exit()
    fault.Mw = np.max(attrs['mag'])
    rake = attrs['rake']
    if np.max(rake) == np.min(rake):
        fault.rake = np.min(rake)
    else:
        print("unexpected rake value")
        exit()
    fault.Mw = attrs['mag']
    if 'dtop' in attrs:
        fault.ztor = np.min(attrs['dtop'])
    else:
        fault.ztor = attrs['hdepth']
    if 'tect_type' in attrs:
        fault.tect_type = TectType[attrs['tect_type']]
    else:
        print("tect_type not found assuming 'ACTIVE_SHALLOW'")
        fault.tect_type = TectType.ACTIVE_SHALLOW
    fault.hdepth = attrs['hdepth']
    return fault


def read_rrup_file(rrup_file):
    """Read rupture(?) file"""
    rrups = dict()
    with open(rrup_file) as f:
        next(f)
        for line in f:
            station, __, __, rrup, rjbs, rx = line.rstrip().split(',')
            rrup = float(rrup)
            rjbs = float(rjbs)
            if rx == "X":
                rx = None
            else:
                rx = float(rx)
            values = (rrup, rjbs, rx)
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


def create_site_parameters(rrup_file, vs30_file, stations=None,
                           vs30_default=500, max_distance=None):
    """Create site parameters"""
    rrups = read_rrup_file(rrup_file)
    vs30_values = read_vs30_file(vs30_file)
    sites = list()
    if max_distance is None:
        max_distance = float("inf")
    if stations is None:
        stations = list(rrups.keys())
    for station in stations:
        rrup, rjbs, rx = rrups[station]
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
            sites.append(site)
    return sites


def calculate_empirical():
    """Calculate empirical intensity measures"""
    parser = argparse.ArgumentParser(
        description="Script to calculate IMs for empirical models."
                    "Produces one .csv for each IM containing "
                    "all specified sites.")

    parser.add_argument('--vs30_file', '-v', required=True,
                        help="vs30 file. Default value is 250 if"
                             " station/file not present")
    parser.add_argument('-r', '--rupture_distance', required=True,
                        help="Path to the rupture distance csv file")
    parser.add_argument('-srf', '--srf_info', help="Path to srf-info file",
                        required=True)
    parser.add_argument('--vs30_default', default=classdef.VS30_DEFAULT,
                        help="Sets the default value for the vs30")
    parser.add_argument('-s', '--stations', nargs='+',
                        help="List of stations to calculate empiricals for")
    parser.add_argument('-rm', '--max_rupture_distance',
                        help="Only calculate empiricals for stations "
                             "that are within X distance to rupture")
    parser.add_argument('-i', '--identifier', help="run-name for run")
    parser.add_argument('-c', '--config',
                        help="configuration file to "
                             "select which model is being used")
    parser.add_argument('-e', '--extended_period', action='store_true',
                        help="Indicate the use of extended(100) pSA periods")

    parser.add_argument('-p', '--period', nargs='+', default=PERIOD, type=float,
                        help='pSA period(s) separated by a '
                             'space. eg: 0.02 0.05 0.1.')
    parser.add_argument('-m', '--im', nargs='+', default=IM_LIST,
                        help='Intensity measure(s) separated by a '
                             'space(if more than one). eg: PGV PGA CAV.')

    parser.add_argument('output', help="output directory")
 
    args = parser.parse_args()

    # Fault & Site parameters
    fault = create_fault_parameters(args.srf_info)
    sites = create_site_parameters(args.rupture_distance, args.vs30_file,
                                   stations=args.stations,
                                   vs30_default=args.vs30_default,
                                   max_distance=args.max_rupture_distance)
    
    setup_dir(args.output)

    if args.extended_period:
        period = np.unique(np.append(PERIOD, EXT_PERIOD))
    else:
        period = PERIOD

    model_dict = empirical_factory.read_model_dict(args.config)
    station_names = [site.name for site in
                     sites] if args.stations is None else args.stations
    for im in args.im:
        cur_gmm = empirical_factory.determine_gmm(fault, im, model_dict)

        # File & column names
        cur_filename = '{}_{}_{}.csv'.format(args.identifier, cur_gmm.name, im)
        cur_cols = []
        if im == PSA_IM_NAME:
            for p in period:
                cur_cols.append('{}_{}'.format(im, p))
                cur_cols.append('{}_{}_sigma'.format(im, p))
        else:
            cur_cols.append(im)
            cur_cols.append('{}_sigma'.format(im))

        # Get & save the data
        cur_data = np.zeros((len(sites), len(cur_cols)), dtype=np.float)
        for ix, site in enumerate(sites):
            values = empirical_factory.compute_gmm(fault, site, cur_gmm, im,
                                                   period)
            if im == PSA_IM_NAME:
                cur_data[ix, :] = np.ravel(
                    [[value_tuple[0], value_tuple[1][0]] for value_tuple in
                     values])
            else:
                cur_data[ix, :] = [values[0], values[1][0]]

        df = pd.DataFrame(columns=cur_cols, data=cur_data)
        df[STATION_COL_NAME] = station_names
        df[COMPONENT_COL_NAME] = 'geom'

        # Correct column order
        df = order_im_cols_df(df)

        df.to_csv(os.path.join(args.output, cur_filename), index=False)


if __name__ == '__main__':
    calculate_empirical()
