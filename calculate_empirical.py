import argparse
from GMM_models.classdef import Site
from GMM_models.classdef import Fault
from GMM_models.classdef import TectType
import empirical_factory
import h5py

IM_LIST = ['PGA', 'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'pSA']
PERIOD = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

def create_fault_parameters(srf_info):
    fault = Fault()
    f = h5py.File(srf_info, 'r')
    attrs = f.attrs
    fault.dip = attrs['dip']
    fault.Mw = attrs['mag']
    fault.rake = attrs['rake']
    if 'dtop' in attrs:
        fault.Ztor = min(attrs['dtop'])
    else:
        fault.Ztor = attrs['hdepth']
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
            values[station] = value
    return values


def create_site_parameters(rrup_file, vs30_file=None, vs30_default=500, max_distance=None):
    rrups = read_rrup_file(rrup_file)
    vs30_values = read_vs30_file(vs30_file)
    sites = list()
    if max_distance is None:
        max_distance = float("inf")
    for station in rrups.keys():
        rrup, rjbs, rx = rrups[station]
        if rrup < max_distance:
            site = Site()
            if station in vs30_values:
                site.V30 = vs30_values[station]
                site.V30measured = True
            else:
                site.V30 = vs30_default
                site.V30measured = False
            site.Rrup = rrup
            site.Rjb = rjbs
            site.Rx = rx
            sites.append(site)
    return sites


def calculate_empirical():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vs30_file', '-v', help="vs30 file. Default value is 500 if station/file not present")
    parser.add_argument('--vs30_default', default=500, help="Sets the default value for the vs30")
    parser.add_argument('-r', '--rupture_distance', help="Path to the rupture distance csv file")
    parser.add_argument('-srf', '--srf_info', help="Path to srf-info file")
    parser.add_argument('-s', '--stations', nargs='+', help="List of stations to calculate empiricals for")
    parser.add_argument('-rm', '--max_rupture_distance',
                        help="Only calculate empiricals for stations that are within X distance to rupture")

    args = parser.parse_args()

    fault = create_fault_parameters(args.srf_info)
    sites = create_site_parameters(args.rupture_distance, args.vs30_file, args.vs30_default, args.max_rupture_distance)

    for im in IM_LIST:
        GMM = empirical_factory.determine_gmm(fault, im)
        for site in sites:
            empirical_factory.compute_gmm(fault, site, GMM, im, PERIOD)


if __name__ == '__main__':
    calculate_empirical()
