import argparse
import numpy as np

import os
import pandas as pd
from datetime import datetime


def aggregate_data():

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir',
                        help='path to output folder that stores the aggregated measures')
    parser.add_argument('-i', '--identifier', help="run-name for run")
    parser.add_argument('-r', '--rupture', default='unknown',
                        help='Please specify the rupture name of the simulation. eg.Albury')
    parser.add_argument('-v', '--version', default='XXpY', help='The version of the simulation. eg.18p4')
    parser.add_argument('im_files', nargs='+')
    args = parser.parse_args()

    csv_np = []
    header = ['station', 'component']

    n_cols = 0

    for im_file in args.im_files:
        d = pd.read_csv(im_file)

        csv_np.append(d)
        header = np.append(header, d.keys().values[2:])
        n_cols += len(d.keys()) - 2

    n_csv = len(csv_np)
    n_stat = csv_np[0].shape[0]

    for i in xrange(len(csv_np)-1):
        # Check if the stations are in the same order for each output file
        assert(np.array_equiv(csv_np[i].station.values, csv_np[i+1].station.values))

    out_dtype = np.dtype(','.join(['|S7'] + ['|S4'] + ['f'] * n_cols))
    out_fmt = ','.join(['%s'] + ['%s'] + ['%f'] * n_cols)

    out_fname = '{}.csv'.format(args.identifier)
    out_file = os.path.join(args.output_dir, out_fname)

    out_data = np.zeros(n_stat, dtype=out_dtype)
    out_data['f0'] = csv_np[0].station.values
    out_data['f1'] = csv_np[0].component.values
    col_i = 2
    for i in xrange(n_csv):
        for col in csv_np[i].columns[2:]:
            out_data['f{}'.format(col_i)] = csv_np[i][col].values
            col_i += 1

    header_str = ','.join(header)
    np.savetxt(out_file, out_data, delimiter=',', fmt=out_fmt, header=header_str, comments='')

    metadata_fname = "{}_empirical.info".format(args.identifier)
    metadata_path = os.path.join(args.output_dir, metadata_fname)
    with open(metadata_path, 'w') as f:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')

        f.write("identifier,rupture,type,date,version\n")
        f.write("{},{},emp,{},{}".format(args.identifier, args.rupture, date, args.version))


if __name__ == '__main__':
    aggregate_data()
