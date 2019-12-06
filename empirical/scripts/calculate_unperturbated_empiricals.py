import argparse
import pathlib
from multiprocessing.pool import Pool

from qcore.formats import load_fault_selection_file
from qcore.simulation_structure import (
    get_sources_dir,
    get_realisation_name,
    get_fault_from_realisation,
    get_srf_info_location,
    get_realisation_verification_dir,
)
from qcore.utils import load_yaml

from empirical.scripts.calculate_empirical import IM_LIST, calculate_empirical
from empirical.util import classdef


def load_args():
    parser = argparse.ArgumentParser("""Script to generate """)

    def absolute_path(path):
        """Takes a path string and returns the absolute path object to it"""
        return pathlib.Path(path).resolve()

    parser.add_argument(
        "fault_selection_file",
        type=absolute_path,
        help="The path to the file listing the faults or events in a simulation",
    )
    parser.add_argument(
        "simulation_root",
        type=absolute_path,
        help="The directory containing a simulations Data and Runs folders",
        default=pathlib.Path(".").resolve(),
        nargs="?",
    )
    parser.add_argument(
        "--vs30_default",
        default=classdef.VS30_DEFAULT,
        help="Sets the default value for the vs30",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="configuration file to select which model is being used",
        type=absolute_path,
    )
    parser.add_argument(
        "--extended_period",
        "-e",
        action="store_true",
        help="Indicate the use of extended(100) pSA periods",
    )
    parser.add_argument(
        "--n_processes", "-n", help="number of processes", type=int, default=1
    )

    args = parser.parse_args()

    return args


def create_event_tasks(events, sim_root, config_file, vs30_default, extended_period):
    tasks = []
    sources = pathlib.Path(get_sources_dir(sim_root))
    vs30_file = load_yaml(pathlib.Path(sim_root) / "Runs" / "root_params.yaml")[
        "stat_vs_est"
    ]
    for realisation_name in events:
        event_name = get_fault_from_realisation(realisation_name)
        fault_info = sources / get_srf_info_location(event_name)
        output_dir = pathlib.Path(
            get_realisation_verification_dir(sim_root, realisation_name)
        )
        rupture_distance = output_dir / f"rrup_{event_name}.csv"
        tasks.append(
            [
                event_name,
                fault_info,
                output_dir,
                config_file,
                None,
                vs30_file,
                vs30_default,
                IM_LIST,
                rupture_distance,
                200,
                extended_period,
            ]
        )
    return tasks


def main():
    args = load_args()

    events = load_fault_selection_file(args.fault_selection_file)
    events = [
        name if count == 1 else get_realisation_name(name, 1)
        for name, count in events.items()
    ]
    tasks = create_event_tasks(
        events,
        args.simulation_root,
        args.config,
        args.vs30_default,
        args.extended_period,
    )

    pool = Pool(min(args.n_processes, len(tasks)))
    pool.starmap(calculate_empirical, tasks)


if __name__ == "__main__":
    main()
