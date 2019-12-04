import argparse
import pathlib
from multiprocessing.pool import Pool

import yaml
from qcore.formats import load_fault_selection_file
from qcore.simulation_structure import (
    get_sources_dir,
    get_fault_dir,
    get_root_yaml_path,
    get_sim_dir,
    get_realisation_name,
)

from empirical.scripts.calculate_empirical import IM_LIST, calculate_empirical
from empirical.util import classdef


def load_args():
    parser = argparse.ArgumentParser()

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
    for event_name in events:
        fault_info = (
            pathlib.Path(get_sources_dir(sim_root)).resolve()
            / event_name
            / f"{event_name}.info"
        )
        output_dir = pathlib.Path(get_fault_dir(sim_root, event_name)) / event_name
        output_dir.mkdir()
        vs30_file = yaml.safe_load(get_root_yaml_path(sim_root))["stat_vs_est"]
        first_rel = get_realisation_name(event_name, 1)
        rupture_distance = (
            pathlib.Path(get_sim_dir(sim_root, first_rel))
            / "verification"
            / f"rrup_{first_rel}.csv"
        )
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

    events = load_fault_selection_file(args.fault_selection_file).keys()
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
