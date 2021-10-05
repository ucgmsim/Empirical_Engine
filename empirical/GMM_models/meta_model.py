import numpy as np


def meta_model(fault, site, gmms, im, period=None, config=None, **kwargs):
    """
    Computes the meta model gmm from a list of gmms and associated weights
    :param fault: Fault object representing the fault the empirical is to be calculated for
    :param site: Site object representing the location the empirical value is to be calculated for
    :param gmms: A list of strings of valid GMM names. Should align with the names of the elements of the GMM enum
    :param weights: An ordered list of weights, corresponding to the weight associated with each entry in gmms
    :param im: The intensity measure to be calculated
    :param period: If the im takes a period, then this should be a list of periods to calculate the values for
    :param config: A dictionary of any settings to be passed to the gmpe. Only used for openQuake models
    :param kwargs: Any additional settings to be passed to the gmpe
    :return: a list of (median, (total sigma, intramodel sigma, intermodel sigma)) nested tuples.
    Of length one or equal to the length of period
    """

    # Local import to prevent cyclic dependencies
    from empirical.util.classdef import GMM
    from empirical.util.empirical_factory import compute_gmm

    # Make sure they correspond
    models = gmms.keys()
    weights = [gmms[x] for x in models]
    assert np.sum(weights) == 1, "weights don't add to 1. Check your gmpe_param_config file."

    medians = []
    sigmas = []

    for gmm in models:
        if config is not None and gmm in config.keys():
            tmp_params_dict = config[gmm]
        else:
            tmp_params_dict = {}
        res = compute_gmm(fault, site, GMM[gmm], im, period, **tmp_params_dict, **kwargs)
        if isinstance(res, tuple):
            m = res[0]
            s = res[1][0]
        else:
            m = [x[0] for x in res]
            s = [x[1][0] for x in res]
        medians.append(m)
        sigmas.append(s)

    logmedians = np.log(np.asarray(medians))
    logsigmas = np.asarray(sigmas)

    weighted_average_median = np.dot(weights, logmedians)

    sigma_average = np.dot(weights, logsigmas)
    sigma_intermodel = np.sqrt(
        np.square(logmedians - weighted_average_median).sum(axis=0) / len(weights)
    )
    average_sigma_total = np.sqrt(
        np.square(sigma_average) + np.square(sigma_intermodel)
    )

    e_medians = np.exp(weighted_average_median).squeeze()
    e_sigmas = average_sigma_total.squeeze()
    if isinstance(e_medians, np.ndarray):
        sigmas = list(zip(e_sigmas, sigma_average, sigma_intermodel))
        res = list(zip(e_medians, sigmas))
    else:
        res = [(e_medians, (e_sigmas, sigma_average, sigma_intermodel))]

    return res
