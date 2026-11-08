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
/// those after `damping` (`pga`, `pgv`, `rsd*`, `EAS(f)` or an SA period; `pgd`
/// is skipped). `row` builds the
/// model's input from one CSV record, keyed by column name.
pub fn check_csv<R: Debug>(
    file: &str,
    row: impl Fn(&HashMap<&str, &str>) -> R,
    predict: impl Fn(IM, &R) -> f64,
) {
    check_csv_tol(file, row, predict, MAX_RELATIVE);
}

/// [`check_csv`] with a custom relative tolerance, for reference tables that
/// OpenQuake itself only matches loosely (e.g. BSSA14's Fortran output).
pub fn check_csv_tol<R: Debug>(
    file: &str,
    row: impl Fn(&HashMap<&str, &str>) -> R,
    predict: impl Fn(IM, &R) -> f64,
    max_relative: f64,
) {
    let path = Path::new(GSIM_DATA).join(file);
    assert!(
        path.is_file(),
        "{} not found, run `git submodule update --init external/oq-engine`",
        path.display()
    );
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let headers = rdr.headers().unwrap().clone();
    let im_start = 1 + headers
        .iter()
        .position(|h| h == "damping")
        .unwrap_or_else(|| panic!("no 'damping' column in {headers:?}"));
    let ims: Vec<Option<IM>> = headers
        .iter()
        .skip(im_start)
        // Some CSVs carry a trailing, empty "Unnamed: N" column (pandas artifact).
        .take_while(|h| !h.starts_with("Unnamed"))
        .map(|h| match h {
            "pga" => Some(IM::PGA {}),
            "pgv" => Some(IM::PGV {}),
            // No PGD in `IM`: columns of models that predict it are skipped.
            "pgd" => None,
            "rsd575" => Some(IM::Ds575 {}),
            "rsd595" => Some(IM::Ds595 {}),
            "rsd2080" => Some(IM::Ds2080 {}),
            _ if h.starts_with("EAS(") => Some(IM::EAS {
                frequency: h[4..h.len() - 1].parse().unwrap(),
            }),
            _ => Some(IM::SA {
                period: h
                    .parse()
                    .unwrap_or_else(|_| panic!("column '{h}' is not an IM")),
            }),
        })
        .collect();

    let mut n_checked = 0usize;
    for record in rdr.records() {
        let record = record.unwrap();
        let row = row(&headers.iter().zip(record.iter()).collect());
        for (&im, value) in ims.iter().zip(record.iter().skip(im_start)) {
            let Some(im) = im else { continue };
            let (predicted, expected) = (predict(im, &row), value.trim().parse::<f64>().unwrap());
            assert!(
                relative_eq!(predicted, expected, max_relative = max_relative),
                "{file}: mismatch for {im:?} ({row:?}): predicted={predicted}, expected={expected}",
            );
            n_checked += 1;
        }
    }
    assert!(n_checked > 0, "no values checked in {file}");
    eprintln!("{file}: checked {n_checked} values");
}
