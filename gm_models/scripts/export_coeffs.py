"""Export an OpenQuake `CoeffsTable` to the CSV format `build.rs` reads.

Run with an environment that has openquake installed (e.g. the repository
root `.venv`), from `gm_models/`:

    ../.venv/bin/python scripts/export_coeffs.py \
        openquake.hazardlib.gsim.zhao_2006:ZhaoEtAl2006Asc.COEFFS_ASC data/za06_asc_coefficients.csv

The first column is `IMT`: the SA period (for EAS rows the period 1/frequency, as OpenQuake interpolates EAS in period), or the lowercased IM name (`pga`,
`pgv`, `cav`, `rsd575`, ...). Values are written with Python's shortest
round-trip `repr`, so they parse back to exactly OpenQuake's float64 values.
"""

import csv
import importlib
import sys

table_path, out_path = sys.argv[1:]
module_name, attr_path = table_path.split(":")
table = importlib.import_module(module_name)
for attr in attr_path.split("."):
    table = getattr(table, attr)

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["IMT", *table.rb.names])
    for imt, record in table._coeffs.items():
        name = (
            repr(imt.period)
            if imt.string.startswith(("SA(", "EAS("))
            else imt.string.lower()
        )
        writer.writerow([name, *(repr(float(v)) for v in record)])
