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

1. **hazardlib verification** (`cargo test`): compares directly against
   hazardlib's own independently-sourced reference tables. Requires
   `OQ_ENGINE_PATH` to point at a local `oq-engine` checkout; tests are
   skipped (not silently passed) if it's unset.
2. **oq_wrapper benchmark parity** (`pytest tests/`): compares the full
   Python-facing output against `oq_wrapper`'s existing benchmark parquet
   fixtures, to check this is a safe drop-in replacement.

## Building

```sh
uv sync --extra test   # builds the Rust extension via setuptools-rust
cargo test              # Layer 1 (set OQ_ENGINE_PATH first)
uv run --extra test pytest tests -v   # Layer 2
```
