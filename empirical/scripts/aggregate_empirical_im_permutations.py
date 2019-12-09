import argparse
import pathlib
from logging import Logger
from typing import List, Union

from qcore.simulation_structure import get_fault_from_realisation, get_verification_dir
from qcore.qclogging import get_logger, get_basic_logger, add_general_file_handler

from empirical.scripts.calculate_empirical import IM_LIST
from empirical.scripts.emp_aggregation import aggregate_data


def load_args():
    parser = argparse.ArgumentParser(
        """Script to generate aggregate empirical im files from all possible permutations of input im files"""
    )

    def absolute_path(path):
        """Takes a path string and returns the absolute path object to it"""
        return pathlib.Path(path).resolve()

    parser.add_argument(
        "simulation_directory",
        type=absolute_path,
        help="The directory of the event or fault realisation to generate aggregated empirical intensity measure files for",
    )
    parser.add_argument(
        "--version", "-v", help="The version of the simulation", default="unversioned"
    )

    args = parser.parse_args()

    return args


def calculate_aggregation_groups(ims):
    """
    Generates groups of im files to be aggregated together
    :param ims: A dictionary of im to im file list mappings
    :return: A list of lists where each list contains a group of files to be aggregated together
    """
    groups = []
    _make_group([], ims, groups)
    return groups


def _make_group(im_list, im_dict, groups):
    """
    Recursively makes groups of im files to be aggregated together.
    The relatively low number of IMs means that stack overflow issues should not occur
    :param im_list: The partial list of im files to be grouped
    :param im_dict: The remaining dictionary of im-im file lists to be added to the im_list
    :param groups: The list of output im file groups, modified in place
    :return: Nothing. Results are returned via the groups argument
    """
    im = list(im_dict.keys())[0]
    im_files = im_dict.pop(im)
    for f in im_files:
        im_list.append(f)
        if im_dict:
            _make_group(im_list, im_dict, groups)
        else:
            groups.append(im_list.copy())
        im_list.pop()
    im_dict[im] = im_files


def get_agg_identifier(realisation: str, im_files: List[pathlib.Path]) -> str:
    """
    Takes a list of files to generate an identifier for and groups the ims together by model
    If only one model is used then don't specify the IMs used
    :param realisation: The name of the realisation being aggregated
    :param im_files: The list of im files to generate the identifier for
    :return: A string identifier for the given group of ims. Should be unique within a group of ims.
    """

    models = {}

    for f in im_files:
        event, *model, im = f.stem.split("_")
        model = "_".join(model)
        if model not in models.keys():
            models[model] = []
        models[model].append(im)

    name = [realisation]

    if len(models.keys()) == 1:
        name.extend(models.keys())
    else:
        for model in models.keys():
            models[model] = sorted(models[model], key=lambda x: IM_LIST.index(x))
        for model in sorted(models.keys(), key=lambda x: IM_LIST.index(models[x][0])):
            name.append(model)
            name.extend(sorted(models[model]))

    return "_".join(name)


def agg_emp_perms(
    simulation_directory: pathlib.Path,
    version: str,
    aggregation_logger: Union[Logger, str] = get_basic_logger(),
):
    """
    Generates aggregated empirical permutations for a given realisation within a simulation run
    :param simulation_directory: The directory of the event or fault realisation to generate aggregation files for
    :param version: The version of the simulation. e.g. the perturbation version
    :param aggregation_logger: The logger object or name of required logger object to be used for logging
    """
    if isinstance(aggregation_logger, str):
        aggregation_logger = get_logger(aggregation_logger)

    verification_dir = pathlib.Path(get_verification_dir(simulation_directory))
    aggregation_logger.debug(f"Using verification directory {verification_dir}")

    event = get_fault_from_realisation(simulation_directory.stem)
    empirical_files = verification_dir.glob(f"{event}_*.csv")

    ims = {}
    for f in empirical_files:
        im = f.stem.split("_")[-1]
        if im not in ims.keys():
            ims[im] = []
        ims[im].append(f)

    if not ims:
        aggregation_logger.error("No empirical IM files found, exiting")
        return

    aggregation_logger.debug(
        f"Found {sum([len(fs) for fs in ims.values()])} empirical IM files"
    )

    groups = calculate_aggregation_groups(ims)

    aggregation_logger.debug(f"Created {len(groups)} aggregated IM groups")
    for group in groups:
        identifier = get_agg_identifier(event, group)
        aggregation_logger.debug(
            f"The identifier {identifier} is being used for the IM group {group}"
        )
        aggregate_data(group, verification_dir, identifier, event, version)
        aggregation_logger.debug(
            f"""Saved empirical IMs to {verification_dir / "{}.csv".format(identifier)}"""
        )


def main():
    main_aggregation_logger = get_logger("Aggregation_logger")
    args = load_args()

    add_general_file_handler(
        main_aggregation_logger,
        args.simulation_directory / "empirical_aggregation_log.txt",
    )
    main_aggregation_logger.debug(
        f"Loaded arguments, creating aggregated empiricals for input: {args}"
    )
    agg_emp_perms(args.simulation_directory, args.version, main_aggregation_logger)


if __name__ == "__main__":
    main()
