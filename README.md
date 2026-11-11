# Empirical_Engine

Empirical ground motion models (GMMs), as two separately versioned Python packages:

| Package | PyPI | What it is |
|---|---|---|
| [`oq_wrapper`](oq_wrapper) | [`oq-wrapper`](https://pypi.org/p/oq-wrapper) | A wrapper around the [OpenQuake engine](https://github.com/gem/oq-engine)'s GMMs, for computing intensity measures from a single GMM or a weighted GMM logic tree. |
| [`gm_models`](gm_models) | [`ucgmsim-gmm`](https://pypi.org/p/ucgmsim-gmm) | Native (Rust) reimplementations of GMMs, including the NZ NSHM2022 ground motion logic tree, verified against OpenQuake's reference data. |

See each package's README for installation and usage.

## Development

Each package has its own `pyproject.toml` and virtual environment, so work from inside the package directory, e.g. `cd gm_models && uv sync --all-extras --dev`. `external/oq-engine` is a submodule pinning the OpenQuake commit whose reference data `gm_models` is verified against (`git submodule update --init` to fetch it).

Git hooks running the same checks as CI are set up once per clone with `uvx lefthook install` (see `lefthook.yml`).

## Releasing

Each package is versioned and released separately, from its own git tags. To release one, create a GitHub release targeting `master` with a new tag of the form `<package>-v<version>`:

- `oq_wrapper-v2026.11.1` publishes `oq-wrapper` 2026.11.1
- `gm_models-v2026.11.1` publishes `ucgmsim-gmm` 2026.11.1

Publishing the release runs `.github/workflows/publish-PyPI.yml`, which builds that package (one pure wheel for `oq_wrapper`; Linux, macOS and Windows wheels for `gm_models`) and uploads it to PyPI. It can also be run by hand from the Actions tab with an existing tag.

The version on PyPI is the tag without its prefix, so install with e.g. `pip install ucgmsim-gmm==2026.11.1`. A tag without one of the two prefixes fails the workflow and publishes nothing, but a typo in the version part is not caught, and PyPI never allows a version to be reused, so double-check the tag. Untagged builds get a development version counted from the package's latest tag (e.g. `2026.11.2.dev3+g1a2b3c4`).
