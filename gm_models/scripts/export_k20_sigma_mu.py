"""Export the global Kuehn et al. (2020) sigma_mu tables to the CSVs `src-rust/k20.rs` embeds.

Run with an environment that has openquake installed, from `gm_models/`:

    <python> scripts/export_k20_sigma_mu.py

One CSV per tectonic type (`data/k20_sigma_mu_{if,slab}.csv`), one row per (M, R) pair
with columns `M,R,PGA,PGV,<SA periods>`, sorted by M then R exactly as
OpenQuake's `_retrieve_sigma_mu_data` sorts the HDF5 tables.
"""

import csv
import importlib

TRT = importlib.import_module("openquake.hazardlib.const").TRT
_retrieve_sigma_mu_data = importlib.import_module(
    "openquake.hazardlib.gsim.kuehn_2020"
)._retrieve_sigma_mu_data

for name, trt in [("if", TRT.SUBDUCTION_INTERFACE), ("slab", TRT.SUBDUCTION_INTRASLAB)]:
    t = _retrieve_sigma_mu_data(trt, "GLO")
    with open(f"data/k20_sigma_mu_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "R", "PGA", "PGV", *(repr(float(p)) for p in t["periods"])])
        for i, m in enumerate(t["M"]):
            for j, r in enumerate(t["R"]):
                w.writerow(
                    [
                        repr(float(m)),
                        repr(float(r)),
                        repr(float(t["PGA"][i, j])),
                        repr(float(t["PGV"][i, j])),
                        *(repr(float(v)) for v in t["SA"][i, j]),
                    ]
                )
