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

def write_data(
    output_file: str,
    rrup_df: pd.DataFrame,
    stations: List[str] = None,
    faults: List[str] = None,
):
    """Writes the data to the output file as a .h5 file
    If stations and/or faults is specified then only those stations and/or faults are written.
    """
    # Filtering
    if stations is not None:
        stations_both = np.intersect1d(rrup_df.index.values, stations)
        rrup_df = rrup_df.loc[stations_both, :]
    if faults is not None:
        faults_both = np.intersect1d(rrup_df.columns.values, faults)
        rrup_df = rrup_df.loc[:, faults_both]

    # Create the output file, will fail if the file already exists
    with h5py.File(output_file, "w") as f:
        # Save the faults and stations
        print("Writing fault and stations data...")
        f.create_dataset("faults", data=rrup_df.columns.values.astype("S"))
        f.create_dataset("stations", data=rrup_df.index.values.astype("S"))

        # Save the data per station (as that is how it will be retrieved)
        print("Writing data per station...")
        for ix, (name, row) in enumerate(rrup_df.iterrows()):
            fault_ind = np.flatnonzero(~row.isna())
            data = np.core.records.fromarrays(
                (fault_ind, row.iloc[fault_ind].values.values),
                dtype=np.dtype([("fault_ix", np.int16), ("rrup", np.float32)]),
            )
            f.create_dataset(str(ix), data=data)

            if ix + 1 % PRINT_FREQUENCY == 0:
                print(
                    "Stations completed {}/{}".format(
                        ix + 1, rrup_df.index.values.shape[0]
                    )
                )


def read_background_txt(background_file):
    return pd.read_csv(background_file, skiprows=5,
                delim_whitespace=True, header=None,
                names=["a", "b", "M_min", "M_cutoff", "n_mags", "totCumRate", "source_lat", "source_lon",
                       "source_depth", "rake", "dip", "tect_type"])


def calculate_ds(background_file, ll_file, vs30_file):
    background_data = read_background_txt(background_file)
    n_faults = np.sum(background_data.n_mags)
    fault_list = [create_rupture_name(row.source_lon, row.source_lat, mag, row.source_depth, row.tect_type) for __, row in background_data.iterrows() for mag in get_mw_range(row.M_min, row.M_cutoff, row.n_mags) ]

    station_df = formats.load_station_ll_vs30(ll_file, vs30_file)

    init_data = np.full((len(station_df), n_faults), np.nan, dtype=np.float32)
    rrup_df = pd.DataFrame(init_data, index=station_df.index.values, columns=fault_list)

    for index, station in station_df.iterrows():
        rrup_df, im_df = calculate_ds_site(background_data, station.lat, station.lon, station.vs30)


def calculate_ds_site(background_data, lat, lon, vs30):
    rrup_dict = {}
    im_df = pd.DataFrame()


    site = Site()
    site.vs30 = vs30
    site.rtvz = 0
    site.rx = 0

    model_dict = empirical_factory.read_model_dict()

    im = 'pSA'
    period = [5.0]

    for index, row in background_data.iterrows():
        if index % PRINT_FREQUENCY == 0:
            print("completed {} / ~20000".format(index))
        fault = Fault()
        fault.hdepth = fault.ztor = row.source_depth
        fault.dip = row.dip
        fault.rake = row.rake
        fault.tect_type = classdef.TectType[row.tect_type]
        rjb = distance.distance((lat, lon), (row.source_lat, row.source_lon)).km
        if rjb < MAX_RJB:
            site.Rjb = rjb
            site.Rrup = np.sqrt(row.source_depth ** 2 + rjb ** 2)
            GMM = empirical_factory.determine_gmm(fault, im, model_dict)
            for mag in get_mw_range(row.M_min, row.M_cutoff, row.n_mags):
                fault.Mw = mag
                name = create_rupture_name(row.source_lon, row.source_lat, mag, row.source_depth, row.tect_type)
                value = empirical_factory.compute_gmm(fault, site, GMM, im, period)
                mean = np.log(value[0][0])
                stdev = value[0][1][0]
                print(name, mean, stdev)


def create_rupture_name(lon, lat, mag, depth, tect_type):
    return "{}_{}_{}_{}_{}".format(lon, lat, mag, depth, tect_type)


def calculate_erf(background_file):
    background_data = read_background_txt(background_file)
    for index, row in background_data.iterrows():
        for mag in get_mw_range(row.M_min, row.M_cutoff, row.n_mags):
            print(create_rupture_name(row.source_lon, row.source_lat, mag, row.source_depth, row.tect_type))
        exit()


def get_mw_range(m_min, m_max, n_mags):
    step = (m_max - m_min) / (n_mags - 1)
    return np.around(np.arange(m_min, m_max + step, step), 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("background_txt", help="background txt file")
    parser.add_argument("ll_file")
    parser.add_argument("vs30_file")
    parser.add_argument("--store_erf", action="store_true", help="writes the erf to a file")
    parser.add_argument("--skip_calculation", action="store_true", help="skip calculations")

    return parser.parse_args()


def calculate_emp_ds():
    args = parse_args()
    if args.store_erf:
        #calculate_erf(args.background_txt)
        pass
    if not args.skip_calculation:
        calculate_ds(args.background_txt, args.ll_file, args.vs30_file)


if __name__ == '__main__':
    calculate_emp_ds()
