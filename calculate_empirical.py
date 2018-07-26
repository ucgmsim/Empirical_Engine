import argparse
from GMM_models.classdef import Site
from GMM_models.classdef import Fault
from GMM_models.classdef import TectType
from GMM_models import classdef
import empirical_factory
import h5py
import numpy as np
import os
import yaml

IM_LIST = ['PGA', 'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'pSA']
PERIOD = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]


def create_fault_parameters(srf_info):
    fault = Fault()
    f = h5py.File(srf_info, 'r')
    attrs = f.attrs
    dip = attrs['dip']
    if np.max(dip) == np.min(dip):
        fault.dip = np.min(dip)
    else:
        print("unexpected dip value")
        exit()
    fault.Mw = attrs['mag']
    rake = attrs['rake']
    if np.max(rake) == np.min(rake):
        fault.rake = np.min(rake)
    else:
        print("unexpected rake value")
        exit()
    fault.Mw = attrs['mag']
    if 'dtop' in attrs:
        fault.ztor = min(attrs['dtop'])
    else:
        fault.ztor = attrs['hdepth']
    if 'tect_type' in attrs:
        fault.tect_type = TectType[attrs['tect_type']]
    else:
        print "tect_type not found assuming 'ACTIVE_SHALLOW'"
        fault.tect_type = TectType.ACTIVE_SHALLOW
    fault.hdepth = attrs['hdepth']
    return fault


def read_rrup_file(rrup_file):
    rrups = dict()
    with open(rrup_file) as f:
        f.next()
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
    values = dict()
    with open(vs30_file) as f:
        for line in f:
            station, value = line.split()
            values[station] = float(value)
    return values


def create_site_parameters(rrup_file, stations, vs30_file=None, vs30_default=500, max_distance=None):
    rrups = read_rrup_file(rrup_file)
    vs30_values = read_vs30_file(vs30_file)
    sites = list()
    if max_distance is None:
        max_distance = float("inf")
    if stations is None:
        stations = rrups.keys()
    for station in stations:
        rrup, rjbs, rx = rrups[station]
        if rrup < max_distance:
            site = Site()
            site.name = station
            if station in vs30_values:
                site.V30 = vs30_values[station]
            else:
                site.V30 = vs30_default
            site.Rrup = rrup
            site.Rjb = rjbs
            site.Rx = rx
            sites.append(site)
    return sites


def calculate_empirical():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vs30_file', '-v', help="vs30 file. Default value is 500 if station/file not present")
    parser.add_argument('--vs30_default', default=classdef.VS30_DEFAULT, help="Sets the default value for the vs30")
    parser.add_argument('-r', '--rupture_distance', help="Path to the rupture distance csv file")
    parser.add_argument('-srf', '--srf_info', help="Path to srf-info file")
    parser.add_argument('-s', '--stations', nargs='+', help="List of stations to calculate empiricals for")
    parser.add_argument('-rm', '--max_rupture_distance',
                        help="Only calculate empiricals for stations that are within X distance to rupture")
    parser.add_argument('-i', '--identifier', help="run-name for run")
    parser.add_argument('-c', '--config', help="configuration file to select which model is being used")
    parser.add_argument('output', help="output directory")

    args = parser.parse_args()

    fault = create_fault_parameters(args.srf_info)
    sites = create_site_parameters(args.rupture_distance, args.stations, args.vs30_file, args.vs30_default, args.max_rupture_distance)

    if args.config is None:
        dir = os.path.dirname(__file__)
        config_file = os.path.join(dir, 'model_config.yaml')
    else:
        config_file = args.config

    model_dict = yaml.load(open(config_file))

    files = {}
    GMM = {}
    for im in IM_LIST:
        gmm = empirical_factory.determine_gmm(fault, im, model_dict)
        GMM[im] = gmm

        filename = '{}_{}_{}.csv'.format(args.identifier, im, gmm.name)
        filepath = os.path.join(args.output, filename)
        files[im] = open(filepath, 'w')
        if im == 'pSA':
            files[im].write('station,component')
            for p in PERIOD:
                files[im].write(',{}_{},{}_{}_sigma'.format(im, p, im, p))
            files[im].write('\n')
        else:
            files[im].write('station,component,{},{}_sigma\n'.format(im, im))

    for site in sites:
        for im in IM_LIST:
            values = empirical_factory.compute_gmm(fault, site, GMM[im], im, PERIOD)

            write_data(files[im], site.name, im, values, PERIOD)


def write_data(file, station_name, im, values, period=None):
    if im == 'pSA':
        file.write('{},geom'.format(station_name))
        for value in values:
            im_value = value[0]
            im_sigma = value[1][0]
            file.write(',{},{}'.format(im_value, im_sigma))
        file.write('\n')
    else:
        im_value = values[0]
        im_sigma = values[1][0]
        file.write('{},geom,{},{}\n'.format(station_name, im_value, im_sigma))


if __name__ == '__main__':
    calculate_empirical()
