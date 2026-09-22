[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Empirical_Engine

A Python wrapper (package name `oq-wrapper`, importable as `oq_wrapper`) around the
[OpenQuake engine](https://github.com/gem/oq-engine)'s ground motion models (GMMs), for
computing empirical intensity measures (IMs) either from a single GMM or from a
weighted GMM logic tree.

## Installation

```bash
uv sync
# or
pip install .
```

Requires Python >= 3.11.

### Running a single GMM

```python
import oq_wrapper as oqw

# rupture_df must contain the columns required by the chosen model, e.g.
# vs30, z1pt0 (in km), z2pt5, rrup, rjb, mag, rake, dip, ztor, hypo_depth, backarc, vs30measured, ...
# See the model's implementation in openquake.hazardlib.gsim for its
# REQUIRES_SITES_PARAMETERS / REQUIRES_RUPTURE_PARAMETERS / REQUIRES_DISTANCES.

pga_result = oqw.run_gmm(
    oqw.constants.GMM.Br_13,
    oqw.constants.TectType.ACTIVE_SHALLOW,
    rupture_df,
    "PGA",
)

psa_result = oqw.run_gmm(
    oqw.constants.GMM.Br_13,
    oqw.constants.TectType.ACTIVE_SHALLOW,
    rupture_df,
    "pSA",
    periods=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
)
```

`run_gmm` returns a `pd.DataFrame` with one row per input row, and per IM/pSA period
columns:
- `{IM}_mean` — mean lnIM value
- `{IM}_std_Total` — total standard deviation
- `{IM}_std_Intra` — within-event (intra-event) standard deviation
- `{IM}_std_Inter` — between-event (inter-event) standard deviation

Tectonic types available: `ACTIVE_SHALLOW`, `SUBDUCTION_SLAB`, `SUBDUCTION_INTERFACE`.

### Running a GMM logic tree

```python
import oq_wrapper as oqw

tect_type = oqw.constants.TectType.ACTIVE_SHALLOW
gmm_lt = oqw.constants.GMMLogicTree.NSHM2022

psa_result, ind_results = oqw.run_gmm_logic_tree(
    gmm_lt,
    tect_type,
    rupture_df,
    "pSA",
    periods=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    return_ind_results=True,
)
```

`run_gmm_logic_tree` computes the weighted mean and standard deviation (within- +
between-model variance) across all GMMs configured for the given logic tree,
tectonic type and IM. Set `return_ind_results=True` to also get back a dict of each
individual model's result and weight. Available logic trees: `NHM2010_BB`, `NSHM2022`,
configured in `oq_wrapper/gmm_logic_tree_configs/*.yaml` (see
`constants.GMM_LT_CONFIG_MAPPING`).

More complete runnable examples (loading a rupture dataframe, renaming columns to
what OpenQuake expects, handling NaNs) are in `oq_wrapper/examples/run_emp_gmm.py`
and `oq_wrapper/examples/run_emp_gmm_logic_tree.py`.

## Testing

```bash
pytest
```

`tests/test_wrapper.py` covers the wrapper functions directly, and
`tests/test_gmm_benchmark.py` runs regression benchmarks against reference outputs
in `tests/benchmark_data/` (generated via `tests/benchmark_data/generate_benchmark_outputs.py`).
