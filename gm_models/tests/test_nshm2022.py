"""NSHM 2022 logic tree weights (combination is checked by the oq_wrapper parity test)."""

import pytest

import gm_models as gmm


def test_weights_sum_to_one() -> None:
    assert sum(gmm.nshm2022.BRANCH_WEIGHTS.values()) == pytest.approx(1)
    for models in gmm.nshm2022.MODELS.values():
        assert sum(weight for _, weight in models.values()) == pytest.approx(1)
