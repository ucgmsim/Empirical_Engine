use csv::Reader;
use std::fs;
use std::path::Path;

/// Column names in `data/p21_coefficients.csv`, in the order the CSV was
/// generated (see the script used to produce it), mapped to the Rust struct
/// field name used in the generated code. Lowercased/renamed only where the
/// hazardlib column name isn't a valid (or idiomatic) Rust identifier
/// (`V2` -> `v2`, `Tau` -> `tau`, `phi2V` -> `phi2v`).
const FIELDS: &[(&str, &str)] = &[
    ("c0", "c0"),
    ("c0slab", "c0slab"),
    ("c1", "c1"),
    ("c1slab", "c1slab"),
    ("a0", "a0"),
    ("a0slab", "a0slab"),
    ("c4", "c4"),
    ("c5", "c5"),
    ("c6", "c6"),
    ("c4slab", "c4slab"),
    ("c5slab", "c5slab"),
    ("c6slab", "c6slab"),
    ("d", "d"),
    ("m", "m"),
    ("db", "db"),
    ("V2", "v2"),
    ("s2", "s2"),
    ("f4", "f4"),
    ("f5", "f5"),
    ("Tau", "tau"),
    ("phi21", "phi21"),
    ("phi22", "phi22"),
    ("phi2V", "phi2v"),
];

fn p21_coefficients() {
    let csv_path = "data/p21_coefficients.csv";
    println!("cargo:rerun-if-changed={}", csv_path);

    let mut rdr = Reader::from_path(csv_path).expect("Could not open p21_coefficients.csv");
    let headers = rdr.headers().expect("Could not read CSV header").clone();
    let col_index: Vec<usize> = FIELDS
        .iter()
        .map(|(csv_name, _)| {
            headers
                .iter()
                .position(|h| h == *csv_name)
                .unwrap_or_else(|| panic!("Column {csv_name} not found in p21_coefficients.csv"))
        })
        .collect();

    let mut pga_row: Option<Vec<f64>> = None;
    let mut pgv_row: Option<Vec<f64>> = None;
    let mut sa_periods: Vec<f64> = Vec::new();
    let mut sa_rows: Vec<Vec<f64>> = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Could not read a row of p21_coefficients.csv");
        let imt = record.get(0).expect("Missing IMT column").to_string();
        let values: Vec<f64> = col_index
            .iter()
            .map(|&idx| {
                record
                    .get(idx)
                    .expect("Missing coefficient value")
                    .parse()
                    .expect("Could not parse coefficient value as f64")
            })
            .collect();

        match imt.as_str() {
            "pga" => pga_row = Some(values),
            "pgv" => pgv_row = Some(values),
            period_str => {
                let period: f64 = period_str
                    .parse()
                    .unwrap_or_else(|_| panic!("Could not parse IMT '{period_str}' as a period"));
                sa_periods.push(period);
                sa_rows.push(values);
            }
        }
    }

    let pga_row = pga_row.expect("p21_coefficients.csv is missing the pga row");
    let pgv_row = pgv_row.expect("p21_coefficients.csv is missing the pgv row");
    let n_periods = sa_periods.len();

    let struct_fields = FIELDS
        .iter()
        .map(|(_, rust_name)| format!("    pub {rust_name}: f64,"))
        .collect::<Vec<_>>()
        .join("\n");

    let coeffs_literal = |values: &[f64]| -> String {
        FIELDS
            .iter()
            .zip(values.iter())
            .map(|((_, rust_name), value)| format!("{rust_name}: {value:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let sa_coeffs_literals = sa_rows
        .iter()
        .map(|row| format!("    Coeffs {{ {} }},", coeffs_literal(row)))
        .collect::<Vec<_>>()
        .join("\n");

    let generated_code = format!(
        r#"
/// P21 (NZ NSHM2022 Parker et al. 2020) coefficients, transcribed from
/// `nz_nshm2022_parker.py`'s `COEFFS` table (global/GLO region only —
/// region-, saturation-region-, and basin-specific columns are omitted
/// since the NZ NSHM2022 logic tree always instantiates this model with
/// `region=None, saturation_region=None, basin=None`).
#[derive(Debug, Clone, Copy)]
pub struct Coeffs {{
{struct_fields}
}}

pub static PGA_COEFFS: Coeffs = Coeffs {{ {pga_literal} }};
pub static PGV_COEFFS: Coeffs = Coeffs {{ {pgv_literal} }};

/// Native SA periods (seconds), ascending, matching `SA_COEFFS` row-for-row.
pub static SA_PERIODS: [f64; {n_periods}] = {sa_periods:?};

/// Coefficient rows for each of `SA_PERIODS`, in the same order.
pub static SA_COEFFS: [Coeffs; {n_periods}] = [
{sa_coeffs_literals}
];
"#,
        struct_fields = struct_fields,
        pga_literal = coeffs_literal(&pga_row),
        pgv_literal = coeffs_literal(&pgv_row),
        n_periods = n_periods,
        sa_periods = sa_periods,
        sa_coeffs_literals = sa_coeffs_literals,
    );

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest_path = Path::new(&out_dir).join("p21_coefficients.rs");
    fs::write(dest_path, generated_code).unwrap();
}

fn main() {
    p21_coefficients();
}
