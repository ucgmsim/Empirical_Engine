# gm_models

Native (Rust) reimplementations of empirical ground motion models, for use
alongside [`oq_wrapper`](../oq_wrapper). Model math lives in Rust
(`src-rust/`); input validation, epistemic-branch handling, and result
assembly live in Python (`gm_models/`).

## Models

- **P_21** (`gm_models.p21`): NZ NSHM2022 modification of Parker et al.
  (2020), global (GLO) region — the subduction interface/slab model used in
  NZ's NSHM2022 logic tree.

## Licence

Unlike the rest of this repository (MIT), `gm_models` is licensed under the GNU Affero General Public License v3.0 or later (see [`LICENSE`](LICENSE)). The models are transcribed from OpenQuake's `openquake.hazardlib` (Copyright (C) GEM Foundation, AGPL-3.0-or-later) and modified, so this package is a derivative work of OpenQuake.

## Verification

Two layers:

1. **OpenQuake verification** (`cargo test -- --ignored`): compares directly against OpenQuake's (`openquake.hazardlib`) own independently-sourced reference tables, read from the `external/oq-engine` git submodule (`git submodule update --init external/oq-engine`), which pins the oq-engine commit. The tests are `#[ignore]`d so a plain `cargo test` works without the submodule. CI runs them in `openquake-verification.yml`; `openquake-data-drift.yml` checks weekly whether the reference data changed on oq-engine master and opens an issue if so.
2. **oq_wrapper benchmark parity** (`pytest tests/`): compares the values of the full Python-facing output against `oq_wrapper`'s existing benchmark parquet fixtures. The output format differs on purpose: models return an `xarray.Dataset` (variables `ln_mean`, `std_total`, `std_inter`, `std_intra`, with a numeric `period` dimension for pSA) rather than `oq_wrapper`'s flat DataFrame with `pSA_{period}_std_Total`-style columns.

A plain `cargo test` also runs property tests (`proptest`) over the supported input domain, which need no external data.

## Benchmarks

`cargo bench` runs the criterion benchmarks in `benches/` and writes an HTML report to `target/criterion/report/index.html`. `cargo bench -- --profile-time 5` also writes flamegraphs (via `pprof`).

## Building

```sh
uv sync --extra test   # builds the Rust extension via setuptools-rust
git submodule update --init external/oq-engine
cargo test -- --ignored   # Layer 1
uv run --extra test pytest tests -v   # Layer 2
```
