import numpy as np


def load_weights(file_path, im, tect_type):
    """
    Load the weights for a specific im and fault from a given weights file
    :param file_path: The path to the weights file to use
    :param im: The string representing the im to use (PGV, pSA, AI etc.)
    :param tect_type: The tect type of the fault being calculated for
    :return: A dictionary of GMM names to weights
    """
    from empirical.util.empirical_factory import read_gmm_weights

    return read_gmm_weights(file_path)[im][tect_type.name]


def meta_model(fault, site, im, weights_path=None, period=None, config=None, **kwargs):
    """
    Computes the meta model gmm from a list of gmms and associated weights
    :param fault: Fault object representing the fault the empirical is to be calculated for
    :param site: Site object representing the location the empirical value is to be calculated for
    :param weights_path: Path to the weights file
    :param im: The intensity measure to be calculated
    :param period: If the im takes a period, then this should be a list of periods to calculate the values for
    :param config: A dictionary of any settings to be passed to subsequent gmpes
    :param kwargs: Any additional settings to be passed to all gmpes
    :return: a list of (median, (total sigma, intramodel sigma, intermodel sigma)) nested tuples.
    Of length one or equal to the length of period
    """

    # Local import to prevent cyclic dependencies
    from empirical.util.classdef import GMM
    from empirical.util.empirical_factory import compute_gmm

    # Load the models and weights, then make sure they correspond
    gmms = load_weights(weights_path, im, fault.tect_type)
    models, weights = zip(*gmms.items())
    assert np.isclose(np.sum(weights), 1, atol=0.01), (
        f"Weights don't add to 1 ({np.sum(weights)} is more than 0.01 "
        f"away from 1). Check your gmpe weights file. "
    )

    medians = []
    sigmas = []

    for gmm in models:
        if config is not None and gmm in config.keys():
            tmp_params_dict = config[gmm]
        else:
            tmp_params_dict = {}
        res = compute_gmm(
            fault, site, GMM[gmm], im, period, **tmp_params_dict, **kwargs
        )
        if isinstance(res, tuple):
            median, (sigma, _, _) = res
        else:
            median = [x[0] for x in res]
            sigma = [x[1][0] for x in res]
        medians.append(median)
        sigmas.append(sigma)

    # Get values as arrays in log space
    logmedians = np.log(np.asarray(medians))
    logsigmas = np.asarray(sigmas)

    # Get weighted median
    weighted_average_median = np.dot(weights, logmedians)

    # Get weighted sigmas
    sigma_average = np.dot(weights, logsigmas)
    sigma_intermodel = np.sqrt(
        np.square(logmedians - weighted_average_median).sum(axis=0) / len(weights)
    )
    average_sigma_total = np.sqrt(
        np.square(sigma_average) + np.square(sigma_intermodel)
    )

    # Get output matrices
    e_medians = np.exp(weighted_average_median).squeeze()
    e_sigmas = average_sigma_total.squeeze()

    # Convert output into (median, (total sigma, intramodel sigma, intermodel sigma)) format
    if isinstance(e_medians, np.ndarray):
        sigmas = list(zip(e_sigmas, sigma_average, sigma_intermodel))
        res = list(zip(e_medians, sigmas))
    else:
        res = [(e_medians, (e_sigmas, sigma_average, sigma_intermodel))]

    return res
