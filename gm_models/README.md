# gm_models

Native (Rust) reimplementations of empirical ground motion models, for use
alongside [`oq_wrapper`](../oq_wrapper). Model math lives in Rust
(`src-rust/`); input validation, epistemic-branch handling, and result
assembly live in Python (`gm_models/`).

## Usage

```python
import gm_models as gmm

ds = gmm.run_p21(
    gmm.constants.TectType.SUBDUCTION_INTERFACE, rupture_df, "pSA", periods=[0.1, 1.0]
)
```

Every model is a `gmm.run_*` function taking the tectonic type, a rupture/site dataframe (OpenQuake column names, Z1.0 in km, as for `oq_wrapper.run_gmm`) and the IM, and returning an `xarray.Dataset` (see below). Its docstring lists the columns it needs.

## Models

- **K_20** (`run_k20`): NZ NSHM2022 modification of Kuehn et al. (2020), global (GLO) region, subduction interface/slab (backarc term, modified sigma, epistemic branches).
- **K_20_NZ** (`run_k20_nz`): Kuehn et al. (2020) NGA-Subduction model with the New Zealand regional coefficients and basin term.
- **P_20** (`run_p20`): Parker et al. (2020) NGA-Subduction global model, subduction interface/slab, no regional adjustments.
- **P_21** (`run_p21`): NZ NSHM2022 modification of Parker et al. (2020), global (GLO) region, the subduction interface/slab model used in NZ's NSHM2022 logic tree.
- **AG_20** (`run_ag20`): NZ NSHM2022 modification of Abrahamson and Gulerce (2020), global (GLO) region with the slab backarc term and epistemic branches.
- **AG_20_NZ** (`run_ag20_nz`): Abrahamson and Gulerce (2020) with the New Zealand (NZL) regional adjustment, no backarc term.
- **A_22** (`run_a22`): Atkinson (2022) NZ NSHM backbone model for active shallow crust (and volcanic), subduction interface and slab, with the NSHM2022 modified sigma.
- **S_22** (`run_s22`): Stafford (2022) NZ NSHM backbone model (modified Chiou and Youngs 2014) for active shallow crust (and volcanic).
- **ASK_14** (`run_ask14`): Abrahamson, Silva and Kamai (2014) NGA-West2, default California model for mainshocks (no aftershock or regional terms), active shallow and volcanic.
- **BSSA_14** (`run_bssa14`): Boore, Stewart, Seyhan and Atkinson (2014) NGA-West2, global model without basin term, active shallow and volcanic.
- **CY_14** (`run_cy14`): Chiou and Youngs (2014) NGA-West2, default OpenQuake configuration (California region, hanging-wall term on), for active shallow and volcanic.
- **Br_13** (`run_br13`): Bradley (2013) New Zealand active shallow crust model (`Bradley2013`, and `Bradley2013Volc` for volcanic), PGA and SA.
- **CB_14** (`run_cb14`): Campbell and Bozorgnia (2014) NGA-West2, with estimated rupture width and OpenQuake's `sigma_mu_epsilon` epistemic branches (active shallow and volcanic).
- **CB_10** (`run_cb10`): Campbell and Bozorgnia (2008) NGA with the 2010 CAV model (active shallow; PGA, PGV, CAV, SA).
- **ZA_06** (`run_za06`): Zhao et al. (2006), active shallow crust, subduction interface and slab.
- **A_18** (`run_a18`): Abrahamson et al. (2018) update of the BC Hydro subduction model (Cascadia adjustment, central branch), interface and slab.
- **AS_16** (`run_as16`): Afshari and Stewart (2016) significant durations (`Ds575`, `Ds595`, `Ds2080`) for active shallow crust, California basin model.
- **BA_18** (`run_ba18`): Bayless and Abrahamson (2018) effective amplitude spectrum (`EAS`, `frequencies=`) for active shallow crust. Above the 24 Hz regression limit it follows the paper rather than OpenQuake, whose total sigma there reads placeholder coefficients (see `src-rust/ba18.rs`).
- **BCH_16** (`run_bch16`): BC Hydro (Abrahamson et al. 2016) with the ESHM20 theta6 calibration and the "Step" forearc/backarc taper, interface and slab.

## Logic tree

`gmm.run_nshm2022` runs the NZ NSHM 2022 GMCM logic tree (Bradley et al. 2024, pSA only), with the same models, weights and mixture combination as `oq_wrapper.run_gmm_logic_tree`. It returns the combined `ln_mean` and `std_total`, plus every branch's values along a `branch` dimension (coordinates `model`, `epistemic_branch`, `weight`).

## Examples

`examples/run_model.py` runs a single model (Br_13) and `examples/run_logic_tree.py` the NSHM 2022 logic tree per tectonic type, both on the NZGMDB records in `oq_wrapper`'s benchmark data.

## Licence

Unlike the rest of this repository (MIT), `gm_models` is licensed under the GNU Affero General Public License v3.0 or later (see [`LICENSE`](LICENSE)). The models are transcribed from OpenQuake's `openquake.hazardlib` (Copyright (C) GEM Foundation, AGPL-3.0-or-later) and modified, so this package is a derivative work of OpenQuake.

## Verification

Two layers:

1. **OpenQuake verification** (`cargo test -- --ignored`): compares directly against OpenQuake's (`openquake.hazardlib`) own independently-sourced reference tables, read from the `external/oq-engine` git submodule (`git submodule update --init external/oq-engine`), which pins the oq-engine commit. The tests are `#[ignore]`d so a plain `cargo test` works without the submodule. CI runs them in `openquake-verification.yml`; `openquake-data-drift.yml` checks weekly whether the reference data changed on oq-engine master and opens an issue if so.
2. **oq_wrapper benchmark parity** (`pytest tests/`): compares the values of the full Python-facing output against `oq_wrapper`'s existing benchmark parquet fixtures. The output format differs on purpose: models return an `xarray.Dataset` (variables `ln_mean`, `std_total`, `std_inter`, `std_intra`, with a numeric `period` dimension for pSA and `frequency` for EAS) rather than `oq_wrapper`'s flat DataFrame with `pSA_{period}_std_Total`-style columns.

Where OpenQuake's own tests only match a reference table loosely (the tables come from the model authors' original code, rounded), the Rust test uses OpenQuake's tolerance instead of the default 0.1%: BSSA_14 (2% mean, 1% sigma), ZA_06 (0.4% mean), A_22 (0.5% mean), AS_16 (3% mean) and BA_18 (0.8% mean). The exact comparison for every model is the oq_wrapper benchmark parity (layer 2).

SA periods outside a model's coefficient table are an error; only ZA_06 has an opt-in `extrapolate` option (`oq_wrapper`'s constant-displacement extrapolation above 5 s, off by default).

A plain `cargo test` also runs property tests (`proptest`) over the supported input domain, which need no external data.

Coefficient tables in `data/` are exported from OpenQuake with `scripts/export_coeffs.py` and compiled in by `build.rs`.

## Benchmarks

`cargo bench` runs the criterion benchmarks in `benches/` and writes an HTML report to `target/criterion/report/index.html`. `cargo bench -- --profile-time 5` also writes flamegraphs (via `pprof`).

## Building

```sh
uv sync --extra test   # builds the Rust extension via setuptools-rust
git submodule update --init external/oq-engine
cargo test -- --ignored   # Layer 1
uv run --extra test pytest tests -v   # Layer 2
```
