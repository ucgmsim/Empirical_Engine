use csv::Reader;
use std::fs;
use std::path::Path;

/// Columns of `data/p21_coefficients.csv` to generate fields for. The Rust
/// field name is the lowercased column name (`V2` -> `v2`, `Tau` -> `tau`).
const P21_FIELDS: &[&str] = &[
    "c0", "c0slab", "c1", "c1slab", "a0", "a0slab", "c4", "c5", "c6", "c4slab", "c5slab", "c6slab",
    "d", "m", "db", "V2", "s2", "f4", "f5", "Tau", "phi21", "phi22", "phi2V",
];

/// Columns of `data/k20_coefficients.csv` to generate fields for: the global
/// and New Zealand columns only.
const K20_FIELDS: &[&str] = &[
    "mu_c_1_if",
    "mu_c_1_slab",
    "c_2_if",
    "c_2_slab",
    "c_3",
    "c_4_if",
    "c_4_slab",
    "c_5",
    "mu_c_6",
    "mu_c_7",
    "c_9_if",
    "c_9_slab",
    "c_1_if_reg_NZ",
    "c_1_slab_reg_NZ",
    "c_7_reg_NZ",
    "c_6_2_reg_NZ",
    "dz_b_if",
    "dz_b_slab",
    "c_nft_1",
    "c_nft_2",
    "c_11_NZ",
    "c_12_NZ",
    "k1",
    "k2",
    "phi",
    "tau",
];

/// Columns of `data/ag20_coefficients.csv` to generate fields for (global and
/// NZL region terms only).
const AG20_FIELDS: &[&str] = &[
    "c1i", "vlin", "b", "a1", "a2", "a6", "a8", "a10", "a11", "a12", "a13", "a14", "a21", "a28",
    "a35", "d1", "d2", "rhoW", "rhoB", "e1", "e2", "e3",
];

/// Columns of `data/a22_coefficients.csv` (Atkinson 2022) to generate fields for.
const A22_FIELDS: &[&str] = &[
    "f",
    "c",
    "Vc",
    "f4",
    "f5",
    "b3",
    "b4",
    "c0_crust",
    "c1_crust",
    "c2_crust",
    "c0_slab",
    "c1_slab",
    "c2_slab",
    "sigma_crust",
    "sigma_inter",
    "sigma_slab",
    "We_crust",
    "We_inter",
    "We_slab",
    "Be_crust",
    "Be_inter",
    "Be_slab",
    "phiS2S",
];

/// Columns of `data/s22_coefficients.csv` (Stafford 2022) to generate fields for.
const S22_FIELDS: &[&str] = &[
    "c1", "c1a", "c1b", "c1c", "c1d", "cn", "cm", "c3", "c5", "chm", "c6", "c7", "c7b", "c9",
    "c9a", "c9b", "c11b", "cg1", "cg2", "cg3", "phi1", "phi2", "phi3", "phi4", "phi5",
];

/// Columns of `data/ask14_coefficients.csv` used by the mainshock, California
/// (no regional term) configuration.
const ASK14_FIELDS: &[&str] = &[
    "m1", "vlin", "b", "c", "c4", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a10", "a11",
    "a12", "a13", "a15", "a17", "a43", "a44", "a45", "a46", "s1e", "s2e", "s3", "s4", "s1m", "s2m",
];

/// Columns of `data/bssa14_coefficients.csv` used by the `region="nobasin"`
/// configuration (no `f6`/`f7` basin coefficients).
const BSSA14_FIELDS: &[&str] = &[
    "e0", "e1", "e2", "e3", "e4", "e5", "e6", "Mh", "c1", "c2", "c3", "h", "Dc3", "c", "Vc", "f4",
    "f5", "R1", "R2", "DfR", "DfV", "f1", "f2", "tau1", "tau2",
];

/// Columns of `data/cy14_coefficients.csv` used by CY14 (the table's region,
/// Japan and constant columns are not needed).
const CY14_FIELDS: &[&str] = &[
    "c1", "c1a", "c1b", "c1c", "c1d", "cn", "cm", "c3", "c5", "chm", "c6", "c7", "c7b", "c9",
    "c9a", "c9b", "c11b", "cg1", "cg2", "cg3", "phi1", "phi2", "phi3", "phi4", "phi5", "tau1",
    "tau2", "sig1", "sig2", "sig3",
];

/// Columns of `data/br13_coefficients.csv`.
const BR13_FIELDS: &[&str] = &[
    "c2", "c3", "c4", "c4a", "crb", "chm", "cg3", "c1", "c1a", "c1b", "cn", "cm", "c5", "c6", "c7",
    "c7a", "c8", "c9", "c9a", "c10", "cg1", "cg2", "ctvz", "phi1", "phi2", "phi3", "phi4", "phi5",
    "phi6", "phi7", "phi8", "tau1", "tau2", "sig1", "sig2", "sig3", "sig4",
];

const CB14_FIELDS: &[&str] = &[
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14",
    "c15", "c16", "c17", "c18", "c19", "c20", "Dc20", "a2", "h1", "h2", "h3", "h5", "h6", "k1",
    "k2", "k3", "phi1", "phi2", "tau1", "tau2", "philnAF", "rholny",
];

const CB10_FIELDS: &[&str] = &[
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "k1", "k2",
    "k3", "c", "n", "s_lny", "t_lny", "s_lnAF", "rho",
];

const ZA06_ASC_FIELDS: &[&str] = &[
    "a", "b", "c", "d", "e", "FR", "CH", "C1", "C2", "C3", "C4", "sigma", "QC", "WC", "tauC",
];

const ZA06_SINTER_FIELDS: &[&str] = &["SI", "QI", "WI", "tauI"];

const ZA06_SSLAB_FIELDS: &[&str] = &["SS", "SSL", "PS", "QS", "WS", "tauS"];

const A18_FIELDS: &[&str] = &[
    "C1inter", "vlin", "b", "a1", "a2", "a4", "a6", "a11", "a12", "a13", "a14", "adj_int",
    "adj_slab", "phi0", "tau0", "rho_w", "rho_b",
];

const BCH16_FIELDS: &[&str] = &[
    "vlin", "b", "theta1", "theta2", "theta6", "theta7", "theta8", "theta10", "theta11", "theta12",
    "theta13", "theta14", "theta15", "theta16", "phi", "tau", "sigma",
];

/// Columns of `data/as16_coefficients.csv` and `data/ba18_coefficients.csv`.
const AS16_FIELDS: &[&str] = &[
    "m1", "m2", "b0N", "b0R", "b0SS", "b1N", "b1R", "b1SS", "b2", "b3", "c1", "c2", "c3", "c4",
    "c5", "vref", "tau1", "tau2", "phi1", "phi2",
];

const BA18_FIELDS: &[&str] = &[
    "c1", "c2", "c3", "cn", "cM", "c4", "c5", "c6", "chm", "c7", "c8", "c9", "c10", "c11a", "c11b",
    "c11c", "c11d", "c1a", "s1", "s2", "s3", "s4", "s5", "s6", "f3", "f4", "f5",
];

/// `IM` variant for each non-SA row name that has one. Other named rows (e.g.
/// `pgd`) still get a `{NAME}_COEFFS` static, just no `Coeffs::for_im` arm.
const NAMED_IMS: &[(&str, &str)] = &[
    ("pga", "PGA {}"),
    ("pgv", "PGV {}"),
    ("cav", "CAV {}"),
    ("rsd575", "Ds575 {}"),
    ("rsd595", "Ds595 {}"),
    ("rsd2080", "Ds2080 {}"),
];

/// Generates a coefficient-table module from an OpenQuake `CoeffsTable`
/// exported to CSV (first column `IMT`: SA periods, or IM names such as `pga`,
/// `pgv`, `cav`, `rsd575`). Writes `$OUT_DIR/{out_file}` containing:
///
/// - a `Coeffs` struct, one `f64` per entry in `fields` (lowercased);
/// - `{NAME}_COEFFS` for each named row (e.g. `PGA_COEFFS`);
/// - if there are SA rows, `SA_PERIODS`/`SA_COEFFS` (ascending) and
///   `Coeffs::for_sa_period`;
/// - `Coeffs::for_im`, for the IMs in the table.
///
/// Each model `include!`s its own file into its own module (e.g.
/// `p21::coeffs`), so the names don't clash.
fn generate_coeffs(csv_path: &str, out_file: &str, doc: &str, fields: &[&str]) {
    println!("cargo:rerun-if-changed={csv_path}");

    let mut rdr =
        Reader::from_path(csv_path).unwrap_or_else(|e| panic!("Could not open {csv_path}: {e}"));
    let headers = rdr.headers().expect("Could not read CSV header").clone();
    let col_index: Vec<usize> = fields
        .iter()
        .map(|csv_name| {
            headers
                .iter()
                .position(|h| h == *csv_name)
                .unwrap_or_else(|| panic!("Column {csv_name} not found in {csv_path}"))
        })
        .collect();

    let mut named_rows: Vec<(String, Vec<f64>)> = Vec::new();
    let mut sa_rows: Vec<(f64, Vec<f64>)> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap_or_else(|e| panic!("Could not read a row of {csv_path}: {e}"));
        let imt = record
            .get(0)
            .expect("Missing IMT column")
            .trim()
            .to_lowercase();
        let values: Vec<f64> = col_index
            .iter()
            .map(|&idx| {
                record
                    .get(idx)
                    .expect("Missing coefficient value")
                    .trim()
                    .parse()
                    .expect("Could not parse coefficient value as f64")
            })
            .collect();
        match imt.parse::<f64>() {
            Ok(period) => sa_rows.push((period, values)),
            Err(_) => named_rows.push((imt, values)),
        }
    }
    sa_rows.sort_by(|a, b| a.0.total_cmp(&b.0));

    let rust_names: Vec<String> = fields.iter().map(|f| f.to_lowercase()).collect();
    let struct_fields = rust_names
        .iter()
        .map(|rust_name| format!("    pub {rust_name}: f64,"))
        .collect::<Vec<_>>()
        .join("\n");
    let interp_fields = rust_names
        .iter()
        .map(|f| format!("            {f}: (above.{f} - self.{f}) * ratio + self.{f},"))
        .collect::<Vec<_>>()
        .join("\n");
    let coeffs_literal = |values: &[f64]| -> String {
        let fields = rust_names
            .iter()
            .zip(values.iter())
            .map(|(rust_name, value)| format!("{rust_name}: {value:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Coeffs {{ {fields} }}")
    };

    let named_statics = named_rows
        .iter()
        .map(|(name, row)| {
            format!(
                "pub static {}_COEFFS: Coeffs = {};",
                name.to_uppercase(),
                coeffs_literal(row)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let has_pga = named_rows.iter().any(|(name, _)| name == "pga");
    let mut for_im_arms: Vec<String> = NAMED_IMS
        .iter()
        .filter(|(name, _)| named_rows.iter().any(|(n, _)| n == name))
        .map(|(name, variant)| {
            format!(
                "            IM::{variant} => Ok({}_COEFFS),",
                name.to_uppercase()
            )
        })
        .collect();

    let sa_code = if sa_rows.is_empty() {
        String::new()
    } else {
        for_im_arms.push("            IM::SA { period } => Self::for_sa_period(period),".into());
        let periods: Vec<f64> = sa_rows.iter().map(|(p, _)| *p).collect();
        let rows = sa_rows
            .iter()
            .map(|(_, row)| format!("    {},", coeffs_literal(row)))
            .collect::<Vec<_>>()
            .join("\n");
        // OpenQuake's PGA-anchored fallback below the smallest SA period
        // (`CoeffsTable._pga_interp_fallback`), only possible with a PGA row.
        let below_min = if has_pga {
            "if i == 0 && period >= PGA_ANCHOR && SA_PERIODS[0] <= MIN_SA_GATE {
            return Ok(PGA_COEFFS.interp(&SA_COEFFS[0], log_ratio(period, PGA_ANCHOR, SA_PERIODS[0])));
        }"
        } else {
            ""
        };
        format!(
            r#"
/// Native SA periods (seconds), ascending, matching `SA_COEFFS` row-for-row.
pub static SA_PERIODS: [f64; {n}] = {periods:?};

/// Coefficient rows for each of `SA_PERIODS`, in the same order.
pub static SA_COEFFS: [Coeffs; {n}] = [
{rows}
];

/// PGA stands in for SA at this period below the smallest tabulated period,
/// if that period is at most `MIN_SA_GATE` (OpenQuake's `_pga_interp_fallback`).
#[allow(dead_code)]
const PGA_ANCHOR: f64 = 0.01;
#[allow(dead_code)]
const MIN_SA_GATE: f64 = 0.05;

fn log_ratio(period: f64, below: f64, above: f64) -> f64 {{
    (period.ln() - below.ln()) / (above.ln() - below.ln())
}}

impl Coeffs {{
    /// SA coefficients at `period`, reproducing OpenQuake's
    /// `CoeffsTable.__getitem__` (`opt=0`, `logratio=True`): the tabulated row
    /// if there is one, else linear interpolation in ln(period) between the
    /// neighbouring rows, else (below the smallest SA period, if the table
    /// has PGA) interpolation from PGA anchored at [`PGA_ANCHOR`]. Other
    /// periods are errors.
    pub fn for_sa_period(period: f64) -> Result<Self, GMMError> {{
        let (min, max) = (SA_PERIODS[0], SA_PERIODS[SA_PERIODS.len() - 1]);
        if period.is_nan() || period > max {{
            return Err(GMMError::UnsupportedPeriod {{ period, min, max }});
        }}
        // First index with SA_PERIODS[i] >= period.
        let i = SA_PERIODS.partition_point(|&p| p < period);
        if SA_PERIODS[i] == period {{
            return Ok(SA_COEFFS[i]);
        }}
        {below_min}
        if i == 0 {{
            return Err(GMMError::UnsupportedPeriod {{ period, min, max }});
        }}
        let ratio = log_ratio(period, SA_PERIODS[i - 1], SA_PERIODS[i]);
        Ok(SA_COEFFS[i - 1].interp(&SA_COEFFS[i], ratio))
    }}
}}
"#,
            n = periods.len(),
        )
    };

    let doc = doc
        .lines()
        .map(|line| format!("/// {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let for_im_arms = for_im_arms.join("\n");

    let generated_code = format!(
        r#"
#[allow(unused_imports)]
use crate::error::GMMError;
#[allow(unused_imports)]
use crate::im::IM;

{doc}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coeffs {{
{struct_fields}
}}

impl Coeffs {{
    /// Field-wise linear interpolation from `self` (ratio 0) to `above` (ratio 1).
    #[allow(dead_code)]
    pub(crate) fn interp(&self, above: &Self, ratio: f64) -> Self {{
        Self {{
{interp_fields}
        }}
    }}

    /// Coefficients for `im`. Fails for IMs not in the table and for SA
    /// periods `Coeffs::for_sa_period` cannot look up.
    #[allow(unreachable_patterns)]
    pub fn for_im(im: IM) -> Result<Self, GMMError> {{
        match im {{
{for_im_arms}
            _ => Err(GMMError::UnsupportedIM(im)),
        }}
    }}
}}

{named_statics}
{sa_code}
"#
    );

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    fs::write(Path::new(&out_dir).join(out_file), generated_code).unwrap();
}

fn main() {
    generate_coeffs(
        "data/p21_coefficients.csv",
        "p21_coefficients.rs",
        "P21 (NZ NSHM2022 Parker et al. 2020) coefficients, transcribed from\n\
         `nz_nshm2022_parker.py`'s `COEFFS` table (global/GLO region only:\n\
         region-, saturation-region-, and basin-specific columns are omitted\n\
         since the NZ NSHM2022 logic tree always instantiates this model with\n\
         `region=None, saturation_region=None, basin=None`).",
        P21_FIELDS,
    );

    generate_coeffs(
        "data/p20_coefficients.csv",
        "p20_coefficients.rs",
        "P20 (Parker et al. 2020) coefficients, transcribed from\n\
         `parker_2020.py`'s `COEFFS` table (global columns only: region-specific\n\
         columns are omitted since `oq_wrapper` instantiates this model with\n\
         `region=None, saturation_region=None, basin=None`).",
        P21_FIELDS,
    );

    generate_coeffs(
        "data/k20_coefficients.csv",
        "k20_coefficients.rs",
        "K20 (Kuehn et al. 2020) coefficients, transcribed from\n\
         `kuehn_2020_coeffs.csv` (global and New Zealand columns only: other\n\
         regional columns are omitted since `oq_wrapper` only instantiates this\n\
         model with `region=\"GLO\"` or `region=\"NZL\"`).",
        K20_FIELDS,
    );
    generate_coeffs(
        "data/ag20_coefficients.csv",
        "ag20_coefficients.rs",
        "AG20 (Abrahamson and Gulerce 2020) coefficients, transcribed from\n\
         `abrahamson_gulerce_2020.py`'s `COEFFS` table (global and NZL region\n\
         terms only: other regional, basin and bias-adjustment columns are omitted).",
        AG20_FIELDS,
    );
    generate_coeffs(
        "data/a22_coefficients.csv",
        "a22_coefficients.rs",
        "A22 (Atkinson 2022 NZ NSHM backbone) coefficients, transcribed from\n\
         `Atkinson22_coeffs_mod_v8b_sanjay_v2.csv` (the columns the model reads).",
        A22_FIELDS,
    );
    generate_coeffs(
        "data/s22_coefficients.csv",
        "s22_coefficients.rs",
        "S22 (Stafford 2022 NZ NSHM backbone) coefficients, transcribed from\n\
         `stafford_2022.py`'s `COEFFS` table (the columns the model reads; the\n\
         Japan, Italy and Wenchuan columns are omitted).",
        S22_FIELDS,
    );
    generate_coeffs(
        "data/ask14_coefficients.csv",
        "ask14_coefficients.rs",
        "ASK14 (Abrahamson et al. 2014) coefficients, transcribed from\n\
         `abrahamson_2014.py`'s `COEFFS` table (regional, aftershock and CyberShake\n\
         columns are omitted since `oq_wrapper` uses the default California model\n\
         for mainshocks).",
        ASK14_FIELDS,
    );
    generate_coeffs(
        "data/bssa14_coefficients.csv",
        "bssa14_coefficients.rs",
        "BSSA14 (Boore et al. 2014) coefficients, transcribed from\n\
         `boore_2014.py`'s `COEFFS` table (basin columns `f6`/`f7` are omitted\n\
         since `oq_wrapper` uses the default `region=\"nobasin\"`).",
        BSSA14_FIELDS,
    );
    generate_coeffs(
        "data/cy14_coefficients.csv",
        "cy14_coefficients.rs",
        "CY14 (Chiou and Youngs, 2014) coefficients, transcribed from\n\
         `chiou_youngs_2014.py`'s `COEFFS` table (only the columns used by the\n\
         default California-region configuration).",
        CY14_FIELDS,
    );
    generate_coeffs(
        "data/br13_coefficients.csv",
        "br13_coefficients.rs",
        "Br13 (Bradley, 2013) coefficients, transcribed from `bradley_2013.py`'s\n\
         `COEFFS` table.",
        BR13_FIELDS,
    );
    generate_coeffs(
        "data/cb14_coefficients.csv",
        "cb14_coefficients.rs",
        "CB_14 (Campbell and Bozorgnia, 2014) coefficients, transcribed from\n\
         `campbell_bozorgnia_2014.py`'s `COEFFS` table (PGA, PGV and SA rows; the\n\
         CAV and IA rows are omitted, and the unused rho1pga, rho2pga and phiC columns).",
        CB14_FIELDS,
    );
    generate_coeffs(
        "data/cb10_coefficients.csv",
        "cb10_coefficients.rs",
        "CB_10 (Campbell and Bozorgnia, 2008, with the 2010 CAV model) coefficients,\n\
         transcribed from `campbell_bozorgnia_2008.py`'s `COEFFS` table (the PGD row\n\
         is kept as `PGD_COEFFS` but unused, and the arbitrary-component c_lny column\n\
         is omitted).",
        CB10_FIELDS,
    );
    generate_coeffs(
        "data/za06_asc_coefficients.csv",
        "za06_asc_coefficients.rs",
        "ZA_06 (Zhao et al. 2006) coefficients shared by all tectonic types\n\
         (`ZhaoEtAl2006Asc.COEFFS_ASC`).",
        ZA06_ASC_FIELDS,
    );
    generate_coeffs(
        "data/za06_sinter_coefficients.csv",
        "za06_sinter_coefficients.rs",
        "ZA_06 subduction interface coefficients (`ZhaoEtAl2006SInter.COEFFS_SINTER`).",
        ZA06_SINTER_FIELDS,
    );
    generate_coeffs(
        "data/za06_sslab_coefficients.csv",
        "za06_sslab_coefficients.rs",
        "ZA_06 subduction slab coefficients (`ZhaoEtAl2006SSlab.COEFFS_SSLAB`).",
        ZA06_SSLAB_FIELDS,
    );
    generate_coeffs(
        "data/a18_coefficients.csv",
        "a18_coefficients.rs",
        "A_18 (Abrahamson et al. 2018 BC Hydro update) coefficients\n\
         (`AbrahamsonEtAl2018SInter.COEFFS`, without the unused epistemic columns).",
        A18_FIELDS,
    );
    generate_coeffs(
        "data/bch16_sinter_coefficients.csv",
        "bch16_sinter_coefficients.rs",
        "BCH_16 (BC Hydro, ESHM20 theta6) interface coefficients\n\
         (`BCHydroESHM20SInter.COEFFS`).",
        BCH16_FIELDS,
    );
    generate_coeffs(
        "data/bch16_sslab_coefficients.csv",
        "bch16_sslab_coefficients.rs",
        "BCH_16 slab theta6 (`BCHydroESHM20SSlab.COEFFS`). All other columns of that\n\
         table equal the interface table's, so only theta6 is read.",
        &["theta6"],
    );
    generate_coeffs(
        "data/bch16_sinter_mag_scale.csv",
        "bch16_sinter_mag_scale.rs",
        "BCH_16 interface magnitude-scaling shift dc1\n\
         (`AbrahamsonEtAl2015SInter.COEFFS_MAG_SCALE`, central branch).",
        &["dc1"],
    );
    generate_coeffs(
        "data/as16_coefficients.csv",
        "as16_coefficients.rs",
        "AS16 (Afshari and Stewart 2016) coefficients, transcribed from\n\
         `afshari_stewart_2016.py`'s `COEFFS` table (`rsd575`, `rsd595`, `rsd2080`).",
        AS16_FIELDS,
    );
    generate_coeffs(
        "data/ba18_coefficients.csv",
        "ba18_coefficients.rs",
        "BA18 (Bayless and Abrahamson 2018) coefficients, transcribed from OpenQuake's\n\
         `bayless_abrahamson_2018.csv`. Rows are EAS frequencies, stored by period\n\
         (1/frequency) as OpenQuake interpolates them; see `ba18.rs`.",
        BA18_FIELDS,
    );
}
