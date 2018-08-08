import argparse
import numpy as np

import os


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

    dtype = [('station', '|S7',), ('component', '|S4'), ('im_value', 'f'), ('im_sigma', 'f')]
    csv_np = []
    header = ['station', 'component']

    for im_file in args.im_files:
        filename = os.path.basename(im_file)
        im_name = filename.split('_')[-1].split('.')[0]

        if im_name == 'pSA':
            pass

        d = np.loadtxt(im_file, dtype=dtype, delimiter=',', skiprows=1)
        csv_np.append(d)
        header += [im_name, im_name + '_sigma']

    n_csv = len(csv_np)
    n_stat = csv_np[0].size

    for i in xrange(len(csv_np)-1):
        assert(np.array_equiv(csv_np[i]['station'], csv_np[i+1]['station']))

    out_dtype = np.dtype(','.join(['|S7'] + ['|S4'] + ['f'] * n_csv * 2))
    out_fmt = ','.join(['%s'] + ['%s'] + ['%f'] * n_csv * 2)

    out_fname = '{}_emp_im.csv'.format(args.identifier)
    out_file = os.path.join(args.output_dir, out_fname)

    out_data = np.zeros(n_stat, dtype=out_dtype)
    out_data['f0'] = csv_np[0]['station']
    out_data['f1'] = csv_np[1]['component']
    for i in xrange(n_csv):
        out_data['f{}'.format(2 * i + 2)] = csv_np[i]['im_value']
        out_data['f{}'.format(2 * i + 3)] = csv_np[i]['im_sigma']

    header_str = ','.join(header)
    np.savetxt(out_file, out_data, delimiter=',', fmt=out_fmt, header=header_str, comments='')


if __name__ == '__main__':
    aggregate_data()
