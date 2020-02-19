import numpy as np
import pytest

# Taken from running MatLab code on Octave
from empirical.GMM_models.Burks_Baker_2013_iesdr import fn_sdi_atten

means = np.asarray(
    [0.887273, 0.622812, 0.44833, 0.325181, 0.235775, 0.170041, 0.121539, 0.0858498, 0.0597837, 0.0409586]
)
sigmas = np.asarray([0.1835, 0.0965, 0.0095, -0.0775, -0.1645, -0.2515, -0.3385, -0.4255, -0.3855, -0.3455])

period = 10
R_hat = np.linspace(0.5, 9.5, 10)
mw = 5.49526674438781


@pytest.mark.parametrize(["period", "R_hat", "mw", "bench_means", "bench_sigmas"], [[period, R_hat, mw, means, sigmas]])
def test_Burks_Barker_2013_Sa(period, R_hat, mw, bench_means, bench_sigmas):
    test_means, test_sigmas = list(zip(*fn_sdi_atten(period, R_hat, mw)))
    assert np.allclose(test_means, bench_means)
    assert np.allclose(test_sigmas, bench_sigmas)