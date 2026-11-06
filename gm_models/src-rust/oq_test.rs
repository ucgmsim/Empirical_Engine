//! Shared test helper: compares a model against OpenQuake's reference CSVs
//! (`openquake/hazardlib/tests/gsim/data`), read from the `external/oq-engine`
//! git submodule. The submodule pins the oq-engine commit, so these tests are
//! deterministic; bump it to pick up upstream data changes.

use crate::im::IM;
use approx::relative_eq;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

const GSIM_DATA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../external/oq-engine/openquake/hazardlib/tests/gsim/data"
);
const MAX_RELATIVE: f64 = 1e-3;

/// Checks `predict(im, &row)` against every IM column of the reference CSV
/// `file` (relative to OpenQuake's gsim test data directory). IM columns are
/// those after `damping` (`pga`, `pgv`, or an SA period). `row` builds the
/// model's input from one CSV record, keyed by column name.
pub fn check_csv<R: Debug>(
    file: &str,
    row: impl Fn(&HashMap<&str, &str>) -> R,
    predict: impl Fn(IM, &R) -> f64,
) {
    let data = Path::new(GSIM_DATA);
    assert!(
        data.is_dir(),
        "{} not found, run `git submodule update --init external/oq-engine`",
        data.display()
    );
    let path = data.join(file);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let headers = rdr.headers().unwrap().clone();
    let im_start = 1 + headers
        .iter()
        .position(|h| h == "damping")
        .unwrap_or_else(|| panic!("no 'damping' column in {headers:?}"));
    let ims: Vec<IM> = headers
        .iter()
        .skip(im_start)
        .map(|h| match h {
            "pga" => IM::PGA {},
            "pgv" => IM::PGV {},
            _ => IM::SA {
                period: h
                    .parse()
                    .unwrap_or_else(|_| panic!("column '{h}' is not an IM")),
            },
        })
        .collect();

    let mut n_checked = 0usize;
    for record in rdr.records() {
        let record = record.unwrap();
        let row = row(&headers.iter().zip(record.iter()).collect());
        for (&im, value) in ims.iter().zip(record.iter().skip(im_start)) {
            let (predicted, expected) = (predict(im, &row), value.parse::<f64>().unwrap());
            assert!(
                relative_eq!(predicted, expected, max_relative = MAX_RELATIVE),
                "{file}: mismatch for {im:?} ({row:?}): predicted={predicted}, expected={expected}",
            );
            n_checked += 1;
        }
    }
    assert!(n_checked > 0, "no values checked in {file}");
    eprintln!("{file}: checked {n_checked} values");
}
