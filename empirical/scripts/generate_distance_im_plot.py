import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from empirical.util import classdef
from empirical.util import empirical_factory
from qcore.im import IM


def main(mw, models, im_list, vs30, rrups, tect_type, output_dir):
    site = classdef.Site()
    site.vs30 = vs30

    fault = classdef.Fault()
    fault.Mw = mw
    fault.tect_type = tect_type

    # Default parameters
    fault.rake = 180
    fault.width = 0.1
    fault.dip = 45
    fault.ztor = 0
    fault.hdepth = 0

    median_im = np.zeros(len(rrups))
    std_dev = np.zeros(len(rrups))

    for im in im_list:
        fig_im, ax_im = plt.subplots()
        for model in models:
            model = classdef.GMM[model]
            for i, rrup in enumerate(rrups):
                site.Rrup = rrup
                site.Rjb = rrup
                result = empirical_factory.compute_gmm(fault, site, model, im.name, im.period)
                median_im[i] = result[0]
                std_dev[i] = result[1][0]

            fig, ax = plt.subplots()
            ax.loglog(rrups, median_im,
                    color="black",
                    label=model.name,)
            ax.loglog(rrups, median_im * np.exp(-std_dev), color="black",                linestyle="dashed",)
            ax.loglog(rrups, median_im * np.exp(std_dev), color="black",                linestyle="dashed",)
            ax.set_xlabel("Rrup (km)")
            ax.set_ylabel(im.pretty_im_name())
            ax.set_title(f"Mw: {mw} dip: {fault.dip} rake: {fault.rake} hdepth: {fault.hdepth} - {tect_type}")
            ax.legend()

            ax_im.loglog(rrups, median_im, label=model.name,)
            ax_im.set_title(f"Mw: {mw} dip: {fault.dip} rake: {fault.rake} hdepth: {fault.hdepth} - {tect_type}")
            fig.savefig(output_dir / f"{im.get_im_name()}_vs_rrup_{model}.png")
            plt.close(fig)

        ax_im.set_xlabel("Rrup (km)")
        ax_im.set_ylabel(im.pretty_im_name())
        ax_im.legend()
        fig_im.savefig(output_dir / f"{im.get_im_name()}_vs_rrup.png")
        plt.close(fig_im)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mw",
                    "-mw", help="magnitude to calculate empiricals", type=float)
    parser.add_argument("--model", "-m", help="which model to calculate", choices=classdef.GMM.get_names(), required=True, nargs="+")
    parser.add_argument("--im", "-i", help="IM to calculate", choices=["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "Ds2080", "pSA"], required=True, nargs="+")
    parser.add_argument("--psa-period", "-p", help="pSA periods to calculate for", nargs="+", type=float)
    parser.add_argument("--vs30", "-v", help="sets a Vs30 value for calculation", type=float, default=250.0)
    parser.add_argument("--tect-type", "-t", help="which tect-type to use", choices=classdef.TectType.get_names(), default=classdef.TectType.ACTIVE_SHALLOW.name)
    parser.add_argument("--rrup-min", help="minimum Rrup range to use", type=float, default=50.0)
    parser.add_argument("--rrup-max", help="maximum Rrup range to use", type=float, default=300.0)
    parser.add_argument("--output-dir", "-o", help="path to output directory - will write plots here", default=".", type=Path)
    args = parser.parse_args()

    rrups = np.linspace(args.rrup_min, args.rrup_max, num=100, endpoint=True)

    im_list = []
    for im in args.im:
        if im == "pSA":
            for period in args.psa_period:
                im_list.append(IM(name=im, period=period))
            continue
        im_list.append(IM(name=im))
    main(args.mw, args.model, im_list, args.vs30, rrups, classdef.TectType[args.tect_type], args.output_dir)