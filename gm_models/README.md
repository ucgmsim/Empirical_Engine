# gm_models

Native (Rust) reimplementations of empirical ground motion models, for use
alongside [`oq_wrapper`](../oq_wrapper). Model math lives in Rust
(`src-rust/`); input validation, epistemic-branch handling, and result
assembly live in Python (`gm_models/`).

## Models

- **P_21** (`gm_models.p21`): NZ NSHM2022 modification of Parker et al.
  (2020), global (GLO) region — the subduction interface/slab model used in
  NZ's NSHM2022 logic tree.

## Verification

Two layers, see the top-level plan for detail:

1. **OpenQuake verification** (`cargo test -- --ignored`): compares directly against
   OpenQuake's (`openquake.hazardlib`) own independently-sourced reference tables. Requires
   `OQ_ENGINE_PATH` to point at a local `oq-engine` checkout; the tests are
   `#[ignore]`d so a plain `cargo test` lists them as ignored. CI runs them
   against oq-engine master in `openquake-verification.yml`, so a failure
   there can come from an upstream change to the reference data.
2. **oq_wrapper benchmark parity** (`pytest tests/`): compares the full
   Python-facing output against `oq_wrapper`'s existing benchmark parquet
   fixtures, to check this is a safe drop-in replacement.

A plain `cargo test` also runs property tests (`proptest`) over the supported input domain, which need no external data.

## Benchmarks

`cargo bench` runs the criterion benchmarks in `benches/` and writes an HTML report to `target/criterion/report/index.html`. `cargo bench -- --profile-time 5` also writes flamegraphs (via `pprof`).

## Building

```sh
uv sync --extra test   # builds the Rust extension via setuptools-rust
OQ_ENGINE_PATH=<oq-engine checkout> cargo test -- --ignored   # Layer 1
uv run --extra test pytest tests -v   # Layer 2
```
