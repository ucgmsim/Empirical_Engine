//! NZ NSHM2022 modification of Parker et al. (2020) subduction GMPE
//! ("P_21" in `oq_wrapper`), global (GLO) region only.
//!
//! Transcribed from `openquake.hazardlib.gsim.nz22.nz_nshm2022_parker`
//! (backarc term, "modified sigma" model, epistemic sigma_mu adjustment)
//! and `openquake.hazardlib.gsim.parker_2020` (magnitude/path/depth/site
//! terms, original sigma model), specialised to
//! `region=None, saturation_region=None, basin=None` — the only
//! configuration `oq_wrapper`'s NSHM2022 logic tree ever instantiates
//! this model with. That means the basin term (which requires `z2pt5`
//! and is 0 whenever a rupture context has no `z2pt5`) is always 0 and
//! is omitted entirely, and the linear site term's region-dependent
//! `s1`/`s2` split collapses to a single `s2` coefficient (see
//! `linear_amplification`).

use crate::coeffs::{Coeffs, PGA_COEFFS, PGV_COEFFS, SA_COEFFS, SA_PERIODS};
use crate::nz22_const::{PERIODS, PERIODS_AG20, RHO_BS, RHO_WS, THETA7S, THETA8S};

const B4: f64 = 0.1;
const F3: f64 = 0.05;
const VB: f64 = 200.0;
const VREF_FNL: f64 = 760.0;
const VREF: f64 = 760.0;

const MB_DEFAULT_INTERFACE: f64 = 7.9;
const MB_DEFAULT_SLAB: f64 = 7.6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TectType {
    Interface,
    Slab,
}

#[derive(Debug, Clone, Copy)]
pub enum Im {
    Pga,
    Pgv,
    Sa(f64),
}

impl Im {
    /// hazardlib's `imt.period` convention: 0.0 for PGA/PGV, the period for SA.
    fn period(self) -> f64 {
        match self {
            Im::Pga | Im::Pgv => 0.0,
            Im::Sa(period) => period,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Inputs {
    pub mag: f64,
    pub rrup: f64,
    pub vs30: f64,
    /// Only read for [`TectType::Slab`].
    pub hypo_depth: f64,
    /// Only read for [`TectType::Slab`].
    pub backarc: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct MeanStd {
    /// Mean, in natural-log space (ln(g) for PGA/SA, ln(cm/s) for PGV).
    pub mean: f64,
    pub total: f64,
    pub inter: f64,
    pub intra: f64,
}

/// Coefficient-table lookup, reproducing hazardlib's `CoeffsTable.__getitem__`
/// (`opt=0` path — see `openquake/hazardlib/gsim/coeffs_table.py:237-307`):
/// exact match if the period is tabulated; otherwise linear interpolation in
/// `ln(period)` between the nearest tabulated period below and above; and,
/// only for periods below the smallest tabulated SA period, linear (not
/// log-linear) interpolation between PGA (treated as period 0) and the
/// smallest tabulated SA period. All periods `oq_wrapper` requests for this
/// model fall within `[SA_PERIODS[0], SA_PERIODS[last]]`, so the
/// below-smallest-period branch is exercised only at exactly the smallest
/// native period (where it degenerates to an exact match).
fn coeffs_for_period(period: f64) -> Coeffs {
    if let Some(idx) = SA_PERIODS.iter().position(|&p| p == period) {
        return SA_COEFFS[idx];
    }

    let mut max_below: Option<usize> = None;
    let mut min_above: Option<usize> = None;
    for (i, &p) in SA_PERIODS.iter().enumerate() {
        if p < period && (max_below.is_none() || p > SA_PERIODS[max_below.unwrap()]) {
            max_below = Some(i);
        } else if p > period && (min_above.is_none() || p < SA_PERIODS[min_above.unwrap()]) {
            min_above = Some(i);
        }
    }

    match (max_below, min_above) {
        (None, Some(above_idx)) => {
            let ratio = period / SA_PERIODS[above_idx];
            interp_coeffs(&PGA_COEFFS, &SA_COEFFS[above_idx], ratio)
        }
        (Some(below_idx), Some(above_idx)) => {
            let below_p = SA_PERIODS[below_idx];
            let above_p = SA_PERIODS[above_idx];
            let ratio = (period.ln() - below_p.ln()) / (above_p.ln() - below_p.ln());
            interp_coeffs(&SA_COEFFS[below_idx], &SA_COEFFS[above_idx], ratio)
        }
        _ => panic!(
            "period {period} is outside the supported range [{}, {}]",
            SA_PERIODS[0],
            SA_PERIODS[SA_PERIODS.len() - 1]
        ),
    }
}

fn interp_coeffs(below: &Coeffs, above: &Coeffs, ratio: f64) -> Coeffs {
    macro_rules! lerp {
        ($f:ident) => {
            (above.$f - below.$f) * ratio + below.$f
        };
    }
    Coeffs {
        c0: lerp!(c0),
        c0slab: lerp!(c0slab),
        c1: lerp!(c1),
        c1slab: lerp!(c1slab),
        a0: lerp!(a0),
        a0slab: lerp!(a0slab),
        c4: lerp!(c4),
        c5: lerp!(c5),
        c6: lerp!(c6),
        c4slab: lerp!(c4slab),
        c5slab: lerp!(c5slab),
        c6slab: lerp!(c6slab),
        d: lerp!(d),
        m: lerp!(m),
        db: lerp!(db),
        v2: lerp!(v2),
        s2: lerp!(s2),
        f4: lerp!(f4),
        f5: lerp!(f5),
        tau: lerp!(tau),
        phi21: lerp!(phi21),
        phi22: lerp!(phi22),
        phi2v: lerp!(phi2v),
    }
}

fn coeffs_for(im: Im) -> Coeffs {
    match im {
        Im::Pga => PGA_COEFFS,
        Im::Pgv => PGV_COEFFS,
        Im::Sa(period) => coeffs_for_period(period),
    }
}

/// (c0, c1, a0, c4, c5, c6) for the given tectonic type — interface reads the
/// unsuffixed columns, slab reads the `*slab` columns (`SUFFIX` in the
/// Python source).
fn select(tect_type: TectType, c: &Coeffs) -> (f64, f64, f64, f64, f64, f64) {
    match tect_type {
        TectType::Interface => (c.c0, c.c1, c.a0, c.c4, c.c5, c.c6),
        TectType::Slab => (c.c0slab, c.c1slab, c.a0slab, c.c4slab, c.c5slab, c.c6slab),
    }
}

fn magnitude_scaling(c6: f64, c4: f64, c5: f64, mag: f64, m_b: f64) -> f64 {
    let m_diff = mag - m_b;
    if m_diff > 0.0 {
        c6 * m_diff
    } else {
        c4 * m_diff + c5 * m_diff.powi(2)
    }
}

fn path_term_h(tect_type: TectType, mag: f64, m_b: f64) -> f64 {
    match tect_type {
        TectType::Interface => 10f64.powf(-0.82 + 0.252 * mag),
        TectType::Slab => {
            let slope = (35f64.log10() - 3.12f64.log10()) / (m_b - 4.0);
            if mag <= m_b {
                10f64.powf(slope * (mag - m_b) + 35f64.log10())
            } else {
                35.0
            }
        }
    }
}

fn path_term(c1: f64, a0: f64, mag: f64, rrup: f64, h: f64) -> f64 {
    let r = (rrup.powi(2) + h.powi(2)).sqrt();
    let r_rref = (r / (1.0 + h.powi(2)).sqrt()).ln();
    c1 * r.ln() + (B4 * mag) * r_rref + a0 * r
}

/// Depth scaling (subduction slab only; always 0 for interface).
fn depth_scaling(hypo_depth: f64, d: f64, m: f64, db: f64) -> f64 {
    if hypo_depth <= 20.0 {
        m * (20.0 - db) + d
    } else if hypo_depth >= db {
        d
    } else {
        m * (hypo_depth - db) + d
    }
}

/// Linear site term. With `region=None`, hazardlib's `s1 == s2`, which
/// collapses the three-branch piecewise form into a single clamp.
fn linear_amplification(vs30: f64, s2: f64, v2: f64) -> f64 {
    s2 * (vs30.min(v2) / VREF).ln()
}

/// Non-linear site term (NZ version — no `period >= 3` cutoff, unlike the
/// base `parker_2020._non_linear_term`; see the NZ source's docstring).
fn non_linear_term(f4: f64, f5: f64, vs30: f64, pgar: f64) -> f64 {
    f4 * ((f5 * (vs30.min(VREF_FNL) - VB)).exp() - (f5 * (VREF_FNL - VB)).exp())
        * ((pgar + F3) / F3).ln()
}

/// Linear interpolation over `ys` at `ln(x)`, matching `scipy.interpolate.interp1d`
/// with default (linear) kind, clamped at the boundaries (never exercised in
/// practice since callers only interpolate within `xs`'s natural range).
fn log_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let lx = x.ln();
    let logs: Vec<f64> = xs.iter().map(|v| v.ln()).collect();
    if lx <= logs[0] {
        return ys[0];
    }
    if lx >= logs[logs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for i in 0..logs.len() - 1 {
        if lx >= logs[i] && lx <= logs[i + 1] {
            let t = (lx - logs[i]) / (logs[i + 1] - logs[i]);
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    ys[ys.len() - 1]
}

/// BC Hydro 2016 / Abrahamson et al. (2016)-style backarc correction.
/// NZ-specific; only nonzero for [`TectType::Slab`] backarc sites.
///
/// Faithfully reproduces a quirk in the hazardlib source: the "no
/// correction for PGV" comment there relies on `imt.period < 0` for PGV,
/// but hazardlib's `PGV().period` is actually `0.0` (same as PGA), so in
/// practice PGV gets the same period<0.02 correction as PGA. We reproduce
/// that behaviour rather than the comment's stated intent, since Layer 1
/// verification compares against hazardlib's actual output.
fn backarc_term(tect_type: TectType, period: f64, rrup: f64, backarc: bool) -> f64 {
    if tect_type != TectType::Slab || !backarc {
        return 0.0;
    }
    let w_epi_factor = 1.008;
    let min_dist = 85.0;
    let (theta7, theta8) = if period < 0.0 {
        (0.0, 0.0)
    } else if period < 0.02 {
        (1.0988, -1.42)
    } else {
        (
            log_interp(&PERIODS[1..], &THETA7S[1..], period),
            log_interp(&PERIODS[1..], &THETA8S[1..], period),
        )
    };
    let dist = rrup.max(min_dist);
    (theta7 + theta8 * (dist / 40.0).ln()) * w_epi_factor
}

/// NZ-NSHM2022-specific epistemic sigma_mu adjustment (global model only —
/// the only configuration in scope here).
fn sigma_epistemic(tect_type: TectType, period: f64) -> f64 {
    let (se1, se2, t1, t2) = match tect_type {
        TectType::Slab => (0.35, 0.22, 0.15, 2.0),
        TectType::Interface => (0.4, 0.4, 0.2, 0.4),
    };
    if period < t1 {
        se1
    } else if period < t2 {
        let ratio1 = (period / t1).ln();
        let ratio2 = (t2 / t1).ln();
        se1 - (se1 - se2) * (ratio1 / ratio2)
    } else {
        se2
    }
}

/// Original (linear) standard-deviation model (`modified_sigma=False`).
fn stddevs_original(c: &Coeffs, rrup: f64, vs30: f64) -> (f64, f64, f64) {
    let (r1, r2, v1, v2) = (200.0, 500.0, 200.0, 500.0);

    let mut phi_rv = if rrup <= r1 {
        c.phi21
    } else if rrup >= r2 {
        c.phi22
    } else {
        (c.phi22 - c.phi21) / (r2.ln() - r1.ln()) * (rrup.ln() - r1.ln()) + c.phi21
    };

    let clamped_r_ratio = || (r2 / rrup.max(r1).min(r2)).ln() / (r2 / r1).ln();
    if vs30 <= v1 {
        phi_rv += c.phi2v * clamped_r_ratio();
    } else if vs30 < v2 {
        phi_rv += c.phi2v * ((v2 / vs30.min(v2)).ln() / (v2 / v1).ln()) * clamped_r_ratio();
    }

    let phi_tot = phi_rv.sqrt();
    let total = (c.tau.powi(2) + phi_tot.powi(2)).sqrt();
    (total, c.tau, phi_tot)
}

/// NZ "modified sigma" nonlinear standard-deviation model (`modified_sigma=True`).
#[allow(clippy::too_many_arguments)]
fn stddevs_nonlinear(
    c: &Coeffs,
    c_pga: &Coeffs,
    period: f64,
    pgar: f64,
    rrup: f64,
    vs30: f64,
) -> (f64, f64, f64) {
    let (r1, r2) = (200.0, 500.0);
    let phi_amp: f64 = 0.3;

    let phi2_rv = if rrup <= r1 {
        c.phi21
    } else if rrup >= r2 {
        c.phi22
    } else {
        (c.phi22 - c.phi21) / (r2.ln() - r1.ln()) * (rrup.ln() - r1.ln()) + c.phi21
    };
    let phi2_rv_pga = if rrup <= r1 {
        c_pga.phi21
    } else if rrup >= r2 {
        c_pga.phi22
    } else {
        (c_pga.phi22 - c_pga.phi21) / (r2.ln() - r1.ln()) * (rrup.ln() - r1.ln()) + c_pga.phi21
    };

    let phi_lin = phi2_rv.sqrt();
    let phi_lin_pga = phi2_rv_pga.sqrt();
    let phi_b = (phi_lin.powi(2) - phi_amp.powi(2)).sqrt();
    let phi_b_pga = (phi_lin_pga.powi(2) - phi_amp.powi(2)).sqrt();

    let tau_lin = c.tau;
    let tau_lin_pga = c_pga.tau;

    let (rho_w, rho_b) = if period < 0.01 {
        (1.0, 1.0)
    } else {
        (
            log_interp(&PERIODS_AG20, &RHO_WS, period),
            log_interp(&PERIODS_AG20, &RHO_BS, period),
        )
    };

    let f2 = c.f4 * ((c.f5 * (vs30.min(VREF_FNL) - VB)).exp() - (c.f5 * (VREF_FNL - VB)).exp());
    let partial_f_pga = f2 * pgar / (pgar + F3);

    let phi2_nl = phi_lin.powi(2)
        + partial_f_pga.powi(2) * phi_b_pga.powi(2)
        + 2.0 * partial_f_pga * phi_b_pga * phi_b * rho_w;
    let tau2_nl = tau_lin.powi(2)
        + partial_f_pga.powi(2) * tau_lin_pga.powi(2)
        + 2.0 * partial_f_pga * tau_lin_pga * tau_lin * rho_b;

    ((tau2_nl + phi2_nl).sqrt(), tau2_nl.sqrt(), phi2_nl.sqrt())
}

/// Computes mean + standard deviations for a single site/rupture pair.
pub fn compute(
    tect_type: TectType,
    im: Im,
    inp: &Inputs,
    sigma_mu_epsilon: f64,
    modified_sigma: bool,
) -> MeanStd {
    let c = coeffs_for(im);
    let c_pga = PGA_COEFFS;
    let period = im.period();

    let m_b = match tect_type {
        TectType::Interface => MB_DEFAULT_INTERFACE,
        TectType::Slab => MB_DEFAULT_SLAB,
    };

    let (c0, c1, a0, c4, c5, c6) = select(tect_type, &c);
    let (c0_pga, c1_pga, a0_pga, c4_pga, c5_pga, c6_pga) = select(tect_type, &c_pga);

    let fm = magnitude_scaling(c6, c4, c5, inp.mag, m_b);
    let fm_pga = magnitude_scaling(c6_pga, c4_pga, c5_pga, inp.mag, m_b);

    let h = path_term_h(tect_type, inp.mag, m_b);
    let fp = path_term(c1, a0, inp.mag, inp.rrup, h);
    // Backarc term applied to the path function for reference-rock PGA.
    let fp_pga = path_term(c1_pga, a0_pga, inp.mag, inp.rrup, h)
        + backarc_term(tect_type, 0.0, inp.rrup, inp.backarc);

    let fd = if tect_type == TectType::Slab {
        depth_scaling(inp.hypo_depth, c.d, c.m, c.db)
    } else {
        0.0
    };
    let fd_pga = if tect_type == TectType::Slab {
        depth_scaling(inp.hypo_depth, c_pga.d, c_pga.m, c_pga.db)
    } else {
        0.0
    };

    let flin = linear_amplification(inp.vs30, c.s2, c.v2);

    // Reference-rock PGA (backarc already folded into fp_pga).
    let pgar = (fp_pga + fm_pga + c0_pga + fd_pga).exp();
    let fnl = non_linear_term(c.f4, c.f5, inp.vs30, pgar);

    let fba = backarc_term(tect_type, period, inp.rrup, inp.backarc);

    // Basin term (fb) omitted: always 0 in this configuration (no z2pt5).
    let mut mean = fp + fnl + flin + fm + c0 + fd + fba;
    if sigma_mu_epsilon != 0.0 {
        mean += sigma_mu_epsilon * sigma_epistemic(tect_type, period);
    }

    let (total, inter, intra) = if modified_sigma {
        stddevs_nonlinear(&c, &c_pga, period, pgar, inp.rrup, inp.vs30)
    } else {
        stddevs_original(&c, inp.rrup, inp.vs30)
    };

    MeanStd {
        mean,
        total,
        inter,
        intra,
    }
}

/// Layer 1 verification: compares this module's output directly against
/// hazardlib's own independently-sourced reference tables (GNS-produced
/// CSVs, generated from Grace Parker's original R implementation — see
/// `nznshm2022_parker2020_test.py`'s test data), read from a local
/// `oq-engine` checkout rather than vendored into this repo (see the
/// gm_models plan for why). Set `OQ_ENGINE_PATH` to that checkout's root;
/// tests are skipped (with a message, not silently) if it's unset.
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;

    const MAX_DISCREP_PERCENTAGE: f64 = 0.1;

    fn oq_engine_data_dir() -> Option<PathBuf> {
        let base = env::var("OQ_ENGINE_PATH").ok()?;
        Some(
            PathBuf::from(base)
                .join("openquake/hazardlib/tests/gsim/data/nz22/NZNSH2022_PARKER20"),
        )
    }

    fn header_index(headers: &csv::StringRecord, name: &str) -> usize {
        headers
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("column '{name}' not found in {headers:?}"))
    }

    struct RowInputs {
        inputs: Inputs,
        value_start: usize,
    }

    fn row_inputs(
        headers: &csv::StringRecord,
        record: &csv::StringRecord,
        tect_type: TectType,
    ) -> RowInputs {
        let mag: f64 = record[header_index(headers, "rup_mag")].parse().unwrap();
        let rrup: f64 = record[header_index(headers, "dist_rrup")].parse().unwrap();
        let vs30: f64 = record[header_index(headers, "site_vs30")].parse().unwrap();
        let (hypo_depth, backarc) = if tect_type == TectType::Slab {
            let hypo_depth: f64 = record[header_index(headers, "rup_hypo_depth")]
                .parse()
                .unwrap();
            let backarc = record[header_index(headers, "site_backarc")]
                .trim()
                .eq_ignore_ascii_case("true");
            (hypo_depth, backarc)
        } else {
            (0.0, false)
        };
        RowInputs {
            inputs: Inputs {
                mag,
                rrup,
                vs30,
                hypo_depth,
                backarc,
            },
            value_start: header_index(headers, "pga"),
        }
    }

    fn im_for_header(header: &str) -> Im {
        if header == "pga" {
            Im::Pga
        } else {
            Im::Sa(header.parse().unwrap_or_else(|_| {
                panic!("column '{header}' is neither 'pga' nor a parseable SA period")
            }))
        }
    }

    fn check_mean_csv(path: &PathBuf, tect_type: TectType) {
        let mut rdr = csv::Reader::from_path(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let headers = rdr.headers().unwrap().clone();

        let mut n_checked = 0usize;
        let mut max_discrep: f64 = 0.0;
        for result in rdr.records() {
            let record = result.unwrap();
            let row = row_inputs(&headers, &record, tect_type);
            for (col_idx, header) in headers.iter().enumerate().skip(row.value_start) {
                let im = im_for_header(header);
                let expected: f64 = record[col_idx].parse().unwrap();
                let predicted = compute(tect_type, im, &row.inputs, 0.0, false).mean.exp();
                let discrep = (predicted / expected * 100.0 - 100.0).abs();
                max_discrep = max_discrep.max(discrep);
                n_checked += 1;
                assert!(
                    discrep <= MAX_DISCREP_PERCENTAGE,
                    "mean mismatch for IM '{header}' (mag={}, rrup={}, vs30={}): \
                     predicted={predicted}, expected={expected}, discrepancy={discrep:.4}%",
                    row.inputs.mag,
                    row.inputs.rrup,
                    row.inputs.vs30,
                );
            }
        }
        assert!(n_checked > 0, "no rows/columns were checked in {}", path.display());
        eprintln!(
            "{}: checked {n_checked} mean values, max discrepancy {max_discrep:.4}%",
            path.display()
        );
    }

    fn check_stddev_csv(path: &PathBuf, tect_type: TectType, modified_sigma: bool) {
        let mut rdr = csv::Reader::from_path(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let headers = rdr.headers().unwrap().clone();

        let mut n_checked = 0usize;
        let mut max_discrep: f64 = 0.0;
        for result in rdr.records() {
            let record = result.unwrap();
            let row = row_inputs(&headers, &record, tect_type);
            for (col_idx, header) in headers.iter().enumerate().skip(row.value_start) {
                let im = im_for_header(header);
                let expected: f64 = record[col_idx].parse().unwrap();
                let predicted = compute(tect_type, im, &row.inputs, 0.0, modified_sigma).total;
                let discrep = (predicted / expected * 100.0 - 100.0).abs();
                max_discrep = max_discrep.max(discrep);
                n_checked += 1;
                assert!(
                    discrep <= MAX_DISCREP_PERCENTAGE,
                    "total stddev mismatch (modified_sigma={modified_sigma}) for IM '{header}' \
                     (mag={}, rrup={}, vs30={}): predicted={predicted}, expected={expected}, \
                     discrepancy={discrep:.4}%",
                    row.inputs.mag,
                    row.inputs.rrup,
                    row.inputs.vs30,
                );
            }
        }
        assert!(n_checked > 0, "no rows/columns were checked in {}", path.display());
        eprintln!(
            "{}: checked {n_checked} stddev values, max discrepancy {max_discrep:.4}%",
            path.display()
        );
    }

    macro_rules! hazardlib_test {
        ($name:ident, $check:expr) => {
            #[test]
            fn $name() {
                let Some(dir) = oq_engine_data_dir() else {
                    eprintln!(
                        "OQ_ENGINE_PATH not set; skipping hazardlib verification test '{}'. \
                         Set it to a local oq-engine checkout root to run this test.",
                        stringify!($name)
                    );
                    return;
                };
                $check(dir);
            }
        };
    }

    hazardlib_test!(interface_mean_matches_hazardlib, |dir: PathBuf| check_mean_csv(
        &dir.join("PARKER2021_INTERFACE_GLO_GNS_MEAN.csv"),
        TectType::Interface
    ));
    hazardlib_test!(slab_mean_matches_hazardlib, |dir: PathBuf| check_mean_csv(
        &dir.join("PARKER2021_SLAB_GLO_GNS_MEAN.csv"),
        TectType::Slab
    ));
    hazardlib_test!(
        interface_stddev_original_matches_hazardlib,
        |dir: PathBuf| check_stddev_csv(
            &dir.join("PARKER2021_INTERFACE_GLO_GNS_TOTAL_STDDEV_ORIGINAL_SIGMA.csv"),
            TectType::Interface,
            false
        )
    );
    hazardlib_test!(
        interface_stddev_modified_matches_hazardlib,
        |dir: PathBuf| check_stddev_csv(
            &dir.join("PARKER2021_INTERFACE_GLO_GNS_TOTAL_STDDEV_MODIFIED_SIGMA.csv"),
            TectType::Interface,
            true
        )
    );
    hazardlib_test!(slab_stddev_original_matches_hazardlib, |dir: PathBuf| {
        check_stddev_csv(
            &dir.join("PARKER2021_SLAB_GLO_GNS_TOTAL_STDDEV_ORIGINAL_SIGMA.csv"),
            TectType::Slab,
            false,
        )
    });
    hazardlib_test!(slab_stddev_modified_matches_hazardlib, |dir: PathBuf| {
        check_stddev_csv(
            &dir.join("PARKER2021_SLAB_GLO_GNS_TOTAL_STDDEV_MODIFIED_SIGMA.csv"),
            TectType::Slab,
            true,
        )
    });
}
