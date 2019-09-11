#!/usr/bin/env python3
"""Script to calculate IMs for empirical distributed seismicity

Writes to one empirical DB
"""
import argparse
import os
from typing import List

from geopy import distance
import h5py
import numpy as np
import pandas as pd

from empirical.util import empirical_factory, classdef
from empirical.util.classdef import Site, Fault, TectType
from qcore import formats
from qcore.utils import setup_dir

PRINT_FREQUENCY = 100
MAX_RJB = 200
IM = ['PGA', 'pSA']
PERIOD = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]


def read_background_txt(background_file):
    return pd.read_csv(background_file, skiprows=5,
                delim_whitespace=True, header=None,
                names=["a", "b", "M_min", "M_cutoff", "n_mags", "totCumRate", "source_lat", "source_lon",
                       "source_depth", "rake", "dip", "tect_type"])


def calculate_ds(background_file, ll_file, vs30_file, ims, psa_periods, output_dir):
    background_data = read_background_txt(background_file)
    rupture_list = [create_rupture_name(row.source_lat, row.source_lon, mag, row.source_depth, row.tect_type)
                    for __, row in background_data.iterrows()
                    for mag in get_mw_range(row.M_min, row.M_cutoff, row.n_mags)]
    fault_list = [create_fault_name(row.source_lat, row.source_lon, row.source_depth)
                    for __, row in background_data.iterrows()]
    rupture_df = pd.DataFrame(rupture_list, columns=["rupture_name"])
    fault_df = pd.DataFrame(fault_list, columns=["fault_name"])

    site_df = formats.load_station_ll_vs30(ll_file, vs30_file)
    n_stations = len(site_df)

    imdb_path = os.path.join(output_dir, 'emp_ds.imdb')
    emp_ds_path = os.path.join(output_dir, 'emp_ds_ss.db')
    with pd.HDFStore(imdb_path) as im_store,\
            pd.HDFStore(emp_ds_path) as distance_store:
        im_store['sites'] = site_df[["lat", "lon"]]
        distance_store['sites'] = pd.DataFrame(site_df.index)

        for index, station in site_df.iterrows():
            print("Processing site {} / {}".format(site_df.index.get_loc(index), n_stations))
            site_im_df, distance_df = calculate_ds_site(ims, psa_periods, rupture_df, fault_df, background_data, station.lat, station.lon, station.vs30)
            if not site_im_df.empty:
                im_store['IM_data/site_{}'.format(station.name)] = site_im_df
                distance_store['distances/site_{}'.format(station.name)] = distance_df

        im_store['ruptures'] = rupture_df
        distance_store['faults'] = fault_df



def calculate_ds_site(ims, psa_periods, rupture_df, fault_df, background_data, lat, lon, vs30):
    im_df = pd.DataFrame()
    distance_df = pd.DataFrame()


    site = Site()
    site.vs30 = vs30
    site.rtvz = 0
    site.rx = 0

    model_dict = empirical_factory.read_model_dict()

    for index, row in background_data.iterrows():
        fault = Fault()
        fault.hdepth = fault.ztor = row.source_depth
        fault.dip = row.dip
        fault.rake = row.rake
        fault.tect_type = classdef.TectType[row.tect_type]
        rjb = distance.distance((lat, lon), (row.source_lat, row.source_lon)).km
        if rjb < MAX_RJB:
            site.Rjb = rjb
            site.Rrup = np.sqrt(row.source_depth ** 2 + rjb ** 2)
            for mag in get_mw_range(row.M_min, row.M_cutoff, row.n_mags):
                fault.Mw = mag
                rupture_name = create_rupture_name(row.source_lat, row.source_lon, mag, row.source_depth, row.tect_type)
                rupture_id = rupture_df[rupture_df["rupture_name"] == rupture_name].index.item()

                im_result_dict = {'rupture_id': rupture_id}


                for im in ims:
                    GMM = empirical_factory.determine_gmm(fault, im, model_dict)
                    values = empirical_factory.compute_gmm(fault, site, GMM, im, psa_periods)
                    if im is not 'pSA':
                        values = [values]
                    for i, value in enumerate(values):
                        full_im_name = get_full_im_name(im, psa_periods[i])
                        mean = np.log(value[0])
                        stdev = value[1][0]

                        im_result_dict[full_im_name] = mean
                        im_result_dict["{}_sigma".format(full_im_name)] = stdev
                im_df = im_df.append(im_result_dict, ignore_index=True)
                # print(name, mean, stdev)

            fault_name = create_fault_name(row.source_lat, row.source_lon, row.source_depth)
            fault_id = fault_df[fault_df["fault_name"] == fault_name].index.item()
            distance_dict = {"fault_id": fault_id, 'rrup': site.Rrup, 'rjb': rjb, "rx": None, "rtvz": None,}
            distance_df = distance_df.append(distance_dict, ignore_index=True)

    return im_df, distance_df


def create_rupture_name(lat, lon, mag, depth, tect_type):
    return "{}_{}_{}--{}_{}".format(lat, lon, depth, mag, tect_type)


def create_fault_name(lat, lon, depth):
    return "{}_{}_{}".format(lat, lon, depth)

def get_full_im_name(im, psa_period):
    if im == 'pSA':
        return "{}_{}".format(im, psa_period)
    else:
        return im

def calculate_erf(background_file):
    background_data = read_background_txt(background_file)
    for index, row in background_data.iterrows():
        for mag in get_mw_range(row.M_min, row.M_cutoff, row.n_mags):
            print(create_rupture_name(row.source_lat, row.source_lon, mag, row.source_depth, row.tect_type))
        exit()


def get_mw_range(m_min, m_max, n_mags):
    step = (m_max - m_min) / (n_mags - 1)
    return np.around(np.arange(m_min, m_max + step, step), 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("background_txt", help="background txt file")
    parser.add_argument("ll_file")
    parser.add_argument("vs30_file")
    parser.add_argument("output_dir")
    parser.add_argument("--periods", default=PERIOD, nargs='+', help="Which pSA periods to calculate for")
    parser.add_argument("--im", default=IM, nargs='+', help="Which IMs to calculate")
    parser.add_argument("--store_erf", action="store_true", help="writes the erf to a file")
    parser.add_argument("--skip_calculation", action="store_true", help="skip calculations")

    return parser.parse_args()


def calculate_emp_ds():
    args = parse_args()
    if args.store_erf:
        #calculate_erf(args.background_txt)
        pass
    if not args.skip_calculation:
        calculate_ds(args.background_txt, args.ll_file, args.vs30_file, args.im, args.periods, args.output_dir)


if __name__ == '__main__':
    calculate_emp_ds()
