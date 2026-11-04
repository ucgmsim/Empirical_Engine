//! NZ NSHM2022 modification of Parker et al. (2020) subduction GMPE
//! ("P_21" in `oq_wrapper`), global (GLO) region only.
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.nz22.nz_nshm2022_parker`
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
//!
//! Equation numbers ("Eq. N") refer to Parker et al. (2022), "NGA-Subduction
//! global ground motion models with regional adjustment factors", Earthquake
//! Spectra 38(1). The NZ-specific backarc term and modified sigma model are not
//! in that paper (see Bradley et al., 2024, BSSA, for the NZ NSHM2022 GMCM).
//!
//! OpenQuake uses `period = 0.0` for both PGA and PGV. Where that convention
//! changes the result for PGV (backarc term, rho_W/rho_B), the `match` on
//! [`IM`] says so explicitly.

use crate::coeffs::{Coeffs, PGA_COEFFS, PGV_COEFFS, SA_COEFFS, SA_PERIODS};
use crate::error::GMMError;
use crate::im::IM;
use crate::nz22_const::{PERIODS, PERIODS_AG20, RHO_BS, RHO_WS, THETA7S, THETA8S};
use crate::result::{GMMResult, MeanStd};
use crate::utils::{check_lengths, log_interp};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// b4 in Eq. 2.
const B4: f64 = 0.1;
/// f3 in Eq. 9.
const F3: f64 = 0.05;
/// Constants 200 and 760 in Eq. 10.
const VB: f64 = 200.0;
const VREF_FNL: f64 = 760.0;
/// Vref in Eq. 8.
const VREF: f64 = 760.0;

/// Global mc (Table 3).
const MB_DEFAULT_INTERFACE: f64 = 7.9;
const MB_DEFAULT_SLAB: f64 = 7.6;

/// P21 input columns, one value per rupture-site pair. Slab-only columns
/// exist only on the `Slab` variant. Exposed to Python as
/// `gm_models._core.P21Inputs`, built with `P21Inputs.interface(...)` or
/// `P21Inputs.slab(...)`. Column lengths are checked in [`P21Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub enum P21Inputs {
    Interface {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
    },
    Slab {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
        hypo_depth: Vec<f64>,
        backarc: Vec<bool>,
    },
}

/// A single row of [`P21Inputs`].
#[derive(Debug, Clone, Copy)]
pub enum P21Row {
    Interface {
        mag: f64,
        rrup: f64,
        vs30: f64,
    },
    Slab {
        mag: f64,
        rrup: f64,
        vs30: f64,
        hypo_depth: f64,
        backarc: bool,
    },
}

impl P21Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = P21Row> + '_, GMMError> {
        let n = match self {
            Self::Interface { mag, rrup, vs30 } => check_lengths(&[
                ("mag", mag.len()),
                ("rrup", rrup.len()),
                ("vs30", vs30.len()),
            ]),
            Self::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth,
                backarc,
            } => check_lengths(&[
                ("mag", mag.len()),
                ("rrup", rrup.len()),
                ("vs30", vs30.len()),
                ("hypo_depth", hypo_depth.len()),
                ("backarc", backarc.len()),
            ]),
        }?;
        Ok((0..n).map(move |i| match self {
            Self::Interface { mag, rrup, vs30 } => P21Row::Interface {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
            },
            Self::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth,
                backarc,
            } => P21Row::Slab {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
                hypo_depth: hypo_depth[i],
                backarc: backarc[i],
            },
        }))
    }
}

#[pymethods]
impl P21Inputs {
    /// Subduction interface inputs.
    #[staticmethod]
    fn interface(
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
    ) -> Self {
        Self::Interface {
            mag: mag.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
        }
    }

    /// Subduction slab inputs.
    #[staticmethod]
    fn slab(
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        hypo_depth: PyReadonlyArray1<f64>,
        backarc: PyReadonlyArray1<bool>,
    ) -> Self {
        Self::Slab {
            mag: mag.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            hypo_depth: hypo_depth.as_array().to_vec(),
            backarc: backarc.as_array().to_vec(),
        }
    }
}

impl Coeffs {
    /// Coefficients for `im`. Fails only for SA periods above the largest
    /// tabulated period.
    pub fn for_im(im: IM) -> Result<Self, GMMError> {
        match im {
            IM::PGA {} => Ok(PGA_COEFFS),
            IM::PGV {} => Ok(PGV_COEFFS),
            IM::SA { period } => Self::for_period(period),
        }
    }

    /// Coefficient-table lookup, reproducing OpenQuake's `CoeffsTable.__getitem__`
    /// (`opt=0` path, see `openquake/hazardlib/gsim/coeffs_table.py:237-307`):
    /// exact match if the period is tabulated; otherwise linear interpolation in
    /// `ln(period)` between the nearest tabulated periods below and above; and,
    /// only for periods below the smallest tabulated SA period, linear (not
    /// log-linear) interpolation between PGA (treated as period 0) and the
    /// smallest tabulated SA period.
    fn for_period(period: f64) -> Result<Self, GMMError> {
        let max = SA_PERIODS[SA_PERIODS.len() - 1];
        if period.is_nan() || period > max {
            return Err(GMMError::UnsupportedPeriod { period, max });
        }
        // First index with SA_PERIODS[i] >= period; SA_PERIODS is ascending.
        let i = SA_PERIODS.partition_point(|&p| p < period);
        if SA_PERIODS[i] == period {
            return Ok(SA_COEFFS[i]);
        }
        if i == 0 {
            return Ok(PGA_COEFFS.interp(&SA_COEFFS[0], period / SA_PERIODS[0]));
        }
        let (below, above) = (SA_PERIODS[i - 1].ln(), SA_PERIODS[i].ln());
        let ratio = (period.ln() - below) / (above - below);
        Ok(SA_COEFFS[i - 1].interp(&SA_COEFFS[i], ratio))
    }
}

/// (c0, c1, a0, c4, c5, c6) for the given tectonic type — interface reads the
/// unsuffixed columns, slab reads the `*slab` columns (`SUFFIX` in the
/// Python source).
fn select(row: &P21Row, c: &Coeffs) -> (f64, f64, f64, f64, f64, f64) {
    match row {
        P21Row::Interface { .. } => (c.c0, c.c1, c.a0, c.c4, c.c5, c.c6),
        P21Row::Slab { .. } => (c.c0slab, c.c1slab, c.a0slab, c.c4slab, c.c5slab, c.c6slab),
    }
}

/// Magnitude scaling FM, Eq. 5.
fn magnitude_scaling(c6: f64, c4: f64, c5: f64, mag: f64, m_b: f64) -> f64 {
    let m_diff = mag - m_b;
    if m_diff > 0.0 {
        c6 * m_diff
    } else {
        c4 * m_diff + c5 * m_diff.powi(2)
    }
}

/// Near-source saturation term h, Eq. 4a (interface) and Eq. 4b (slab).
fn path_term_h(row: &P21Row, mag: f64, m_b: f64) -> f64 {
    match row {
        P21Row::Interface { .. } => 10f64.powf(-0.82 + 0.252 * mag),
        P21Row::Slab { .. } => {
            let slope = (35f64.log10() - 3.12f64.log10()) / (m_b - 4.0);
            if mag <= m_b {
                10f64.powf(slope * (mag - m_b) + 35f64.log10())
            } else {
                35.0
            }
        }
    }
}

/// Path term FP, Eq. 2, with R and Rref from Eqs. 3a and 3b.
fn path_term(c1: f64, a0: f64, mag: f64, rrup: f64, h: f64) -> f64 {
    let r = rrup.hypot(h);
    let r_rref = (r / 1f64.hypot(h)).ln();
    c1 * r.ln() + (B4 * mag) * r_rref + a0 * r
}

/// Depth scaling FD, Eq. 6, with db1 = 20 km (subduction slab only; always 0
/// for interface).
fn depth_scaling(hypo_depth: f64, d: f64, m: f64, db: f64) -> f64 {
    if hypo_depth <= 20.0 {
        m * (20.0 - db) + d
    } else if hypo_depth >= db {
        d
    } else {
        m * (hypo_depth - db) + d
    }
}

/// Linear site term Flin, Eq. 8. With `region=None`, OpenQuake's `s1 == s2`,
/// which collapses the three-branch piecewise form into a single clamp.
fn linear_amplification(vs30: f64, s2: f64, v2: f64) -> f64 {
    s2 * (vs30.min(v2) / VREF).ln()
}

/// f2, Eq. 10.
fn f2(f4: f64, f5: f64, vs30: f64) -> f64 {
    f4 * ((f5 * (vs30.min(VREF_FNL) - VB)).exp() - (f5 * (VREF_FNL - VB)).exp())
}

/// Non-linear site term Fnl, Eq. 9 (with f1 = 0).
/// NZ version: no `period >= 3` cutoff, unlike the base
/// `parker_2020._non_linear_term` (see the NZ source's docstring).
fn non_linear_term(f4: f64, f5: f64, vs30: f64, pgar: f64) -> f64 {
    f2(f4, f5, vs30) * ((pgar + F3) / F3).ln()
}

/// BC Hydro 2016 / Abrahamson et al. (2016)-style backarc correction.
/// NZ-specific; applied only to slab backarc sites.
fn backarc_term(im: IM, rrup: f64) -> Result<f64, GMMError> {
    let w_epi_factor = 1.008;
    let min_dist = 85.0;
    let (theta7, theta8) = match im {
        IM::SA { period } if period >= 0.02 => (
            log_interp(&PERIODS[1..], &THETA7S[1..], period)?,
            log_interp(&PERIODS[1..], &THETA8S[1..], period)?,
        ),
        // Includes PGV: OpenQuake's "no correction for PGV" branch tests
        // `imt.period < 0`, but `PGV().period` is 0.0, so PGV gets the PGA
        // correction. Reproduced here since we verify against OpenQuake output.
        _ => (1.0988, -1.42),
    };
    let dist = rrup.max(min_dist);
    Ok((theta7 + theta8 * (dist / 40.0).ln()) * w_epi_factor)
}

/// Epistemic standard deviation sigma_e(T), Eq. 27, with the NZ-NSHM2022
/// coefficients (global model only, the only configuration in scope here).
/// PGA and PGV take the short-period value se1.
fn sigma_epistemic(row: &P21Row, im: IM) -> f64 {
    let (se1, se2, t1, t2) = match row {
        P21Row::Slab { .. } => (0.35, 0.22, 0.15, 2.0),
        P21Row::Interface { .. } => (0.4, 0.4, 0.2, 0.4),
    };
    match im {
        IM::SA { period } if period >= t2 => se2,
        IM::SA { period } if period >= t1 => {
            se1 - (se1 - se2) * (period / t1).ln() / (t2 / t1).ln()
        }
        _ => se1,
    }
}

/// phi^2(Rrup), Eq. 19.
fn phi2_rrup(c: &Coeffs, rrup: f64) -> f64 {
    let (r1, r2): (f64, f64) = (200.0, 500.0);
    if rrup <= r1 {
        c.phi21
    } else if rrup >= r2 {
        c.phi22
    } else {
        (c.phi22 - c.phi21) / (r2.ln() - r1.ln()) * (rrup.ln() - r1.ln()) + c.phi21
    }
}

/// Original (linear) standard-deviation model (`modified_sigma=False`):
/// phi^2(Rrup) from Eq. 19 plus DeltaVar(Vs30) from Eq. 20, then phi (Eq. 18)
/// and total sigma (Eq. 17).
fn stddevs_original(c: &Coeffs, rrup: f64, vs30: f64) -> (f64, f64, f64) {
    let (r1, r2, v1, v2) = (200.0, 500.0, 200.0, 500.0);

    let mut phi2_rv = phi2_rrup(c, rrup);
    let clamped_r_ratio = || (r2 / rrup.max(r1).min(r2)).ln() / (r2 / r1).ln();
    if vs30 <= v1 {
        phi2_rv += c.phi2v * clamped_r_ratio();
    } else if vs30 < v2 {
        phi2_rv += c.phi2v * ((v2 / vs30.min(v2)).ln() / (v2 / v1).ln()) * clamped_r_ratio();
    }

    let phi_tot = phi2_rv.sqrt();
    (c.tau.hypot(phi_tot), c.tau, phi_tot)
}

/// NZ "modified sigma" nonlinear standard-deviation model (`modified_sigma=True`).
/// The linear within-event variance is Eq. 19 (without Eq. 20's DeltaVar), and
/// f2 is Eq. 10. The nonlinear propagation follows Abrahamson and Gulerce
/// (2022), Earthquake Spectra 38(4), "AG22 Eq. N" below, with rho_W/rho_B from
/// their Table E6. Their Eq. 28 (d f_site / d ln PGA) is replaced by the
/// derivative of Parker's Eq. 9.
fn stddevs_nonlinear(
    c: &Coeffs,
    c_pga: &Coeffs,
    im: IM,
    pgar: f64,
    rrup: f64,
    vs30: f64,
) -> Result<(f64, f64, f64), GMMError> {
    let phi_amp2: f64 = 0.3f64.powi(2);

    let phi2_lin = phi2_rrup(c, rrup);
    // AG22 Eq. 26
    let phi_b = (phi2_lin - phi_amp2).sqrt();
    let phi_b_pga = (phi2_rrup(c_pga, rrup) - phi_amp2).sqrt();

    let (rho_w, rho_b) = match im {
        IM::SA { period } if period >= 0.01 => (
            log_interp(&PERIODS_AG20, &RHO_WS, period)?,
            log_interp(&PERIODS_AG20, &RHO_BS, period)?,
        ),
        // Includes PGV, via OpenQuake's `PGV().period == 0.0`.
        _ => (1.0, 1.0),
    };

    // d Fnl / d ln(PGAr) from Eq. 9, in place of AG22 Eq. 28
    let partial_f_pga = f2(c.f4, c.f5, vs30) * pgar / (pgar + F3);

    // AG22 Eqs. 27 and 29
    let phi2_nl = phi2_lin
        + partial_f_pga.powi(2) * phi_b_pga.powi(2)
        + 2.0 * partial_f_pga * phi_b_pga * phi_b * rho_w;
    let tau2_nl = c.tau.powi(2)
        + partial_f_pga.powi(2) * c_pga.tau.powi(2)
        + 2.0 * partial_f_pga * c_pga.tau * c.tau * rho_b;

    Ok(((tau2_nl + phi2_nl).sqrt(), tau2_nl.sqrt(), phi2_nl.sqrt()))
}

/// Mean + standard deviations for `NZNSHM2022_ParkerEtAl2020SInter` /
/// `NZNSHM2022_ParkerEtAl2020SSlab` (global/GLO region), one value per
/// rupture-site pair in `inputs`. The tectonic type is set by how
/// `inputs` was built (`P21Inputs.interface` or `P21Inputs.slab`).
/// Exposed to Python as `gm_models._core.p21`.
#[pyfunction]
pub fn p21(
    inputs: PyRef<'_, P21Inputs>,
    im: IM,
    sigma_mu_epsilon: f64,
    modified_sigma: bool,
) -> PyResult<GMMResult> {
    let coeffs = Coeffs::for_im(im)?;
    Ok(inputs
        .iter()?
        .map(|row| compute(im, &coeffs, &row, sigma_mu_epsilon, modified_sigma))
        .collect::<Result<GMMResult, _>>()?)
}

/// Computes mean + standard deviations for a single site/rupture pair.
///
/// `c` must be `Coeffs::for_im(im)`; it is passed in so callers looping over
/// sites look it up once rather than per site. Errors only if a period-table
/// interpolation falls outside its table, which `Coeffs::for_im`'s period
/// check should already rule out.
pub fn compute(
    im: IM,
    c: &Coeffs,
    row: &P21Row,
    sigma_mu_epsilon: f64,
    modified_sigma: bool,
) -> Result<MeanStd, GMMError> {
    let c_pga = PGA_COEFFS;

    let (mag, rrup, vs30, m_b) = match *row {
        P21Row::Interface { mag, rrup, vs30 } => (mag, rrup, vs30, MB_DEFAULT_INTERFACE),
        P21Row::Slab {
            mag, rrup, vs30, ..
        } => (mag, rrup, vs30, MB_DEFAULT_SLAB),
    };
    let backarc = matches!(row, P21Row::Slab { backarc: true, .. });

    let (c0, c1, a0, c4, c5, c6) = select(row, c);
    let (c0_pga, c1_pga, a0_pga, c4_pga, c5_pga, c6_pga) = select(row, &c_pga);

    let fm = magnitude_scaling(c6, c4, c5, mag, m_b);
    let fm_pga = magnitude_scaling(c6_pga, c4_pga, c5_pga, mag, m_b);

    let h = path_term_h(row, mag, m_b);
    let fp = path_term(c1, a0, mag, rrup, h);
    // NZ backarc term, for the reference-rock PGA (folded into its path term)
    // and for the target IM.
    let (fba_pga, fba) = if backarc {
        (backarc_term(IM::PGA {}, rrup)?, backarc_term(im, rrup)?)
    } else {
        (0.0, 0.0)
    };
    let fp_pga = path_term(c1_pga, a0_pga, mag, rrup, h) + fba_pga;

    let (fd, fd_pga) = match *row {
        P21Row::Slab { hypo_depth, .. } => (
            depth_scaling(hypo_depth, c.d, c.m, c.db),
            depth_scaling(hypo_depth, c_pga.d, c_pga.m, c_pga.db),
        ),
        P21Row::Interface { .. } => (0.0, 0.0),
    };

    let flin = linear_amplification(vs30, c.s2, c.v2);

    // Reference-rock PGA, Eq. 1 with PGA coefficients and FS = 0 (backarc
    // already folded into fp_pga).
    let pgar = (fp_pga + fm_pga + c0_pga + fd_pga).exp();
    let fnl = non_linear_term(c.f4, c.f5, vs30, pgar);

    // Eq. 1, with FS = Flin + Fnl from Eq. 7. Basin term (Fb) omitted: always 0
    // in this configuration (no z2pt5). fba is the NZ backarc addition.
    // Epistemic adjustment of c0 by epsilon * sigma_e (Eq. 27 and text).
    let mean = fp + fnl + flin + fm + c0 + fd + fba + sigma_mu_epsilon * sigma_epistemic(row, im);

    let (total, inter, intra) = if modified_sigma {
        stddevs_nonlinear(c, &c_pga, im, pgar, rrup, vs30)?
    } else {
        stddevs_original(c, rrup, vs30)
    };

    Ok(MeanStd {
        mean,
        total,
        inter,
        intra,
    })
}

/// Layer 1 verification: compares this module's output directly against
/// OpenQuake's own independently-sourced reference tables (GNS-produced
/// CSVs, generated from Grace Parker's original R implementation — see
/// `nznshm2022_parker2020_test.py`'s test data), read from a local
/// `oq-engine` checkout rather than vendored into this repo (see the
/// gm_models plan for why). These tests are `#[ignore]`d so a plain
/// `cargo test` reports them as ignored rather than passed; run them with
/// `OQ_ENGINE_PATH=<oq-engine checkout root> cargo test -- --ignored`.
/// CI runs them against oq-engine master (`openquake-verification.yml`), so a
/// failure there may come from an upstream data change, not from this code.
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;

    const MAX_DISCREP_PERCENTAGE: f64 = 0.1;

    fn oq_engine_data_dir() -> PathBuf {
        let base = env::var("OQ_ENGINE_PATH")
            .expect("OQ_ENGINE_PATH must be set to an oq-engine checkout root");
        PathBuf::from(base).join("openquake/hazardlib/tests/gsim/data/nz22/NZNSH2022_PARKER20")
    }

    fn header_index(headers: &csv::StringRecord, name: &str) -> usize {
        headers
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("column '{name}' not found in {headers:?}"))
    }

    struct RowInputs {
        inputs: P21Row,
        value_start: usize,
    }

    /// Slab CSVs are the only ones with `rup_hypo_depth`/`site_backarc` columns.
    fn row_inputs(headers: &csv::StringRecord, record: &csv::StringRecord) -> RowInputs {
        let mag: f64 = record[header_index(headers, "rup_mag")].parse().unwrap();
        let rrup: f64 = record[header_index(headers, "dist_rrup")].parse().unwrap();
        let vs30: f64 = record[header_index(headers, "site_vs30")].parse().unwrap();
        let inputs = if headers.iter().any(|h| h == "rup_hypo_depth") {
            P21Row::Slab {
                mag,
                rrup,
                vs30,
                hypo_depth: record[header_index(headers, "rup_hypo_depth")]
                    .parse()
                    .unwrap(),
                backarc: record[header_index(headers, "site_backarc")]
                    .trim()
                    .eq_ignore_ascii_case("true"),
            }
        } else {
            P21Row::Interface { mag, rrup, vs30 }
        };
        RowInputs {
            inputs,
            value_start: header_index(headers, "pga"),
        }
    }

    fn im_for_header(header: &str) -> IM {
        if header == "pga" {
            IM::PGA {}
        } else {
            IM::SA {
                period: header.parse().unwrap_or_else(|_| {
                    panic!("column '{header}' is neither 'pga' nor a parseable SA period")
                }),
            }
        }
    }

    fn check_mean_csv(path: &PathBuf) {
        let mut rdr = csv::Reader::from_path(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let headers = rdr.headers().unwrap().clone();

        let mut n_checked = 0usize;
        let mut max_discrep: f64 = 0.0;
        for result in rdr.records() {
            let record = result.unwrap();
            let row = row_inputs(&headers, &record);
            for (col_idx, header) in headers.iter().enumerate().skip(row.value_start) {
                let im = im_for_header(header);
                let expected: f64 = record[col_idx].parse().unwrap();
                let predicted = compute(im, &Coeffs::for_im(im).unwrap(), &row.inputs, 0.0, false)
                    .unwrap()
                    .mean
                    .exp();
                let discrep = (predicted / expected * 100.0 - 100.0).abs();
                max_discrep = max_discrep.max(discrep);
                n_checked += 1;
                assert!(
                    discrep <= MAX_DISCREP_PERCENTAGE,
                    "mean mismatch for IM '{header}' ({:?}): \
                     predicted={predicted}, expected={expected}, discrepancy={discrep:.4}%",
                    row.inputs,
                );
            }
        }
        assert!(
            n_checked > 0,
            "no rows/columns were checked in {}",
            path.display()
        );
        eprintln!(
            "{}: checked {n_checked} mean values, max discrepancy {max_discrep:.4}%",
            path.display()
        );
    }

    fn check_stddev_csv(path: &PathBuf, modified_sigma: bool) {
        let mut rdr = csv::Reader::from_path(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let headers = rdr.headers().unwrap().clone();

        let mut n_checked = 0usize;
        let mut max_discrep: f64 = 0.0;
        for result in rdr.records() {
            let record = result.unwrap();
            let row = row_inputs(&headers, &record);
            for (col_idx, header) in headers.iter().enumerate().skip(row.value_start) {
                let im = im_for_header(header);
                let expected: f64 = record[col_idx].parse().unwrap();
                let predicted = compute(
                    im,
                    &Coeffs::for_im(im).unwrap(),
                    &row.inputs,
                    0.0,
                    modified_sigma,
                )
                .unwrap()
                .total;
                let discrep = (predicted / expected * 100.0 - 100.0).abs();
                max_discrep = max_discrep.max(discrep);
                n_checked += 1;
                assert!(
                    discrep <= MAX_DISCREP_PERCENTAGE,
                    "total stddev mismatch (modified_sigma={modified_sigma}) for IM '{header}' \
                     ({:?}): predicted={predicted}, expected={expected}, \
                     discrepancy={discrep:.4}%",
                    row.inputs,
                );
            }
        }
        assert!(
            n_checked > 0,
            "no rows/columns were checked in {}",
            path.display()
        );
        eprintln!(
            "{}: checked {n_checked} stddev values, max discrepancy {max_discrep:.4}%",
            path.display()
        );
    }

    macro_rules! openquake_test {
        ($name:ident, $check:expr) => {
            #[test]
            #[ignore = "needs OQ_ENGINE_PATH; run with `cargo test -- --ignored`"]
            fn $name() {
                $check(oq_engine_data_dir());
            }
        };
    }

    openquake_test!(interface_mean_matches_openquake, |dir: PathBuf| {
        check_mean_csv(&dir.join("PARKER2021_INTERFACE_GLO_GNS_MEAN.csv"))
    });
    openquake_test!(slab_mean_matches_openquake, |dir: PathBuf| check_mean_csv(
        &dir.join("PARKER2021_SLAB_GLO_GNS_MEAN.csv")
    ));
    openquake_test!(
        interface_stddev_original_matches_openquake,
        |dir: PathBuf| check_stddev_csv(
            &dir.join("PARKER2021_INTERFACE_GLO_GNS_TOTAL_STDDEV_ORIGINAL_SIGMA.csv"),
            false
        )
    );
    openquake_test!(
        interface_stddev_modified_matches_openquake,
        |dir: PathBuf| check_stddev_csv(
            &dir.join("PARKER2021_INTERFACE_GLO_GNS_TOTAL_STDDEV_MODIFIED_SIGMA.csv"),
            true
        )
    );
    openquake_test!(slab_stddev_original_matches_openquake, |dir: PathBuf| {
        check_stddev_csv(
            &dir.join("PARKER2021_SLAB_GLO_GNS_TOTAL_STDDEV_ORIGINAL_SIGMA.csv"),
            false,
        )
    });
    openquake_test!(slab_stddev_modified_matches_openquake, |dir: PathBuf| {
        check_stddev_csv(
            &dir.join("PARKER2021_SLAB_GLO_GNS_TOTAL_STDDEV_MODIFIED_SIGMA.csv"),
            true,
        )
    });
}

/// Property tests over the supported input domain. These need no external
/// data, so they run in a plain `cargo test`.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn im() -> impl Strategy<Value = IM> {
        prop_oneof![
            Just(IM::PGA {}),
            Just(IM::PGV {}),
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    fn row() -> impl Strategy<Value = P21Row> {
        let common = (5.0..9.5f64, 0.1..1000.0f64, 100.0..2000.0f64);
        prop_oneof![
            common
                .clone()
                .prop_map(|(mag, rrup, vs30)| P21Row::Interface { mag, rrup, vs30 }),
            (common, 0.0..200.0f64, any::<bool>()).prop_map(
                |((mag, rrup, vs30), hypo_depth, backarc)| P21Row::Slab {
                    mag,
                    rrup,
                    vs30,
                    hypo_depth,
                    backarc,
                }
            ),
        ]
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            im in im(),
            row in row(),
            sigma_mu_epsilon in -1.2815..=1.2815f64,
            modified_sigma in any::<bool>(),
        ) {
            let c = Coeffs::for_im(im).unwrap();
            let r = compute(im, &c, &row, sigma_mu_epsilon, modified_sigma).unwrap();
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let rel = (r.total.powi(2) - (r.inter.powi(2) + r.intra.powi(2))).abs() / r.total.powi(2);
            prop_assert!(rel < 1e-12, "total^2 != inter^2 + intra^2 (rel err {rel})");
        }

        #[test]
        fn periods_above_table_are_rejected(period in 10.0f64..1e3) {
            prop_assume!(period > SA_PERIODS[SA_PERIODS.len() - 1]);
            prop_assert!(Coeffs::for_im(IM::SA { period }).is_err(), "period {period} accepted");
        }

        #[test]
        fn interpolated_coeffs_lie_between_neighbours(i in 0..SA_PERIODS.len() - 1, t in 0.0..1.0f64) {
            let (lo, hi) = (SA_PERIODS[i], SA_PERIODS[i + 1]);
            let period = (lo.ln() + t * (hi.ln() - lo.ln())).exp();
            let c = Coeffs::for_im(IM::SA { period }).unwrap();
            let (a, b) = (SA_COEFFS[i].tau, SA_COEFFS[i + 1].tau);
            prop_assert!(c.tau >= a.min(b) - 1e-12 && c.tau <= a.max(b) + 1e-12);
        }
    }

    #[test]
    fn tabulated_periods_return_exact_rows() {
        for (&period, row) in SA_PERIODS.iter().zip(SA_COEFFS.iter()) {
            assert_eq!(Coeffs::for_im(IM::SA { period }).unwrap(), *row);
        }
    }

    #[test]
    fn inputs_reject_mismatched_lengths_and_iterate_rows() {
        let slab = |backarc: Vec<bool>| P21Inputs::Slab {
            mag: vec![7.0, 8.0],
            rrup: vec![10.0, 20.0],
            vs30: vec![400.0, 500.0],
            hypo_depth: vec![30.0, 40.0],
            backarc,
        };
        assert!(slab(vec![true]).iter().is_err());
        let rows: Vec<P21Row> = slab(vec![false, true]).iter().unwrap().collect();
        assert!(matches!(
            rows[..],
            [
                P21Row::Slab {
                    mag: 7.0,
                    backarc: false,
                    ..
                },
                P21Row::Slab {
                    mag: 8.0,
                    backarc: true,
                    ..
                },
            ]
        ));
    }

    #[test]
    fn nan_period_is_rejected() {
        assert!(Coeffs::for_im(IM::SA { period: f64::NAN }).is_err());
    }
}
