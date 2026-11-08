//! Abrahamson and Gulerce (2020) subduction GMM, in the two configurations
//! `oq_wrapper` uses: "AG_20_NZ" (`AbrahamsonGulerce2020SInter` / `SSlab` with
//! `region="NZL"`) and "AG_20" (`NZNSHM2022_AbrahamsonGulerce2020SInter` / `SSlab`
//! with its default `region="GLO"`, which adds the NZ backarc term to slab
//! events and an optional epistemic `sigma_mu_epsilon` shift).
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.abrahamson_gulerce_2020`
//! and `openquake.hazardlib.gsim.nz22.nz_nshm2022_abrahamson_gulerce_2020`,
//! specialised to `ergodic=True, apply_usa_adjustment=False,
//! usgs_basin_scaling=False, ak23_bias_adj=False` and to the GLO and NZL
//! regions. Those regions have no basin term (it is only defined for CAS and
//! JPN), no region-specific `a2` adjustment, and a linear-only `phi` model
//! (`phi_lin = phi1`, Eq. 5.2), so all of that is omitted. The NZ22 variant's
//! own `rho_W`/`rho_B` tables (`nz22_const`) are not used: AG20 reads `rhoW` and
//! `rhoB` from its coefficient table.
//!
//! Equation numbers ("Eq. N.N") refer to Abrahamson and Gulerce (2020), PEER
//! Report 2020/25 (and its Erratum). The NZ-specific backarc term and the
//! epistemic adjustment are applied as in Bradley et al. (2024), BSSA.
//!
//! PGV is not defined for this model; `Coeffs::for_im` rejects it.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/ag20_coefficients.rs"));
}

use crate::error::GMMError;
use crate::im::IM;
use crate::nz22_const::backarc_term;
use crate::result::{GMMResult, MeanStd};
use crate::utils::check_lengths;
use coeffs::{Coeffs, PGA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Region-independent constants (OpenQuake's `CONSTS`).
const A3: f64 = 0.1;
const A4: f64 = 0.73;
const A9: f64 = 0.4;
const A45: f64 = 0.34;
const C4: f64 = 10.0;
const N: f64 = 1.18;
const C: f64 = 1.88;
const TAU_LIN: f64 = 0.47;
const PHI_AMP: f64 = 0.3;
/// Vs30 cap in Eq. 3.8 (Erratum).
const VS30_MAX: f64 = 1000.0;

/// Region of the model: the global model, or the New Zealand adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Global,
    NewZealand,
}

/// AG20 input columns, one value per rupture-site pair. Slab-only columns
/// exist only on the `Slab` variant. Exposed to Python as
/// `gm_models._core.AG20Inputs`, built with `AG20Inputs.interface(...)` or
/// `AG20Inputs.slab(...)`. Column lengths are checked in [`AG20Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub enum AG20Inputs {
    Interface {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
    },
    Slab {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
        ztor: Vec<f64>,
        backarc: Vec<bool>,
    },
}

/// A single row of [`AG20Inputs`].
#[derive(Debug, Clone, Copy)]
pub enum AG20Row {
    Interface {
        mag: f64,
        rrup: f64,
        vs30: f64,
    },
    Slab {
        mag: f64,
        rrup: f64,
        vs30: f64,
        ztor: f64,
        backarc: bool,
    },
}

impl AG20Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = AG20Row> + '_, GMMError> {
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
                ztor,
                backarc,
            } => check_lengths(&[
                ("mag", mag.len()),
                ("rrup", rrup.len()),
                ("vs30", vs30.len()),
                ("ztor", ztor.len()),
                ("backarc", backarc.len()),
            ]),
        }?;
        Ok((0..n).map(move |i| match self {
            Self::Interface { mag, rrup, vs30 } => AG20Row::Interface {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
            },
            Self::Slab {
                mag,
                rrup,
                vs30,
                ztor,
                backarc,
            } => AG20Row::Slab {
                mag: mag[i],
                rrup: rrup[i],
                vs30: vs30[i],
                ztor: ztor[i],
                backarc: backarc[i],
            },
        }))
    }
}

#[pymethods]
impl AG20Inputs {
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
        ztor: PyReadonlyArray1<f64>,
        backarc: PyReadonlyArray1<bool>,
    ) -> Self {
        Self::Slab {
            mag: mag.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            ztor: ztor.as_array().to_vec(),
            backarc: backarc.as_array().to_vec(),
        }
    }
}

/// Region-adjusted (a1, a6, a12): the regional base constant (Eq. 3.1), the
/// anelastic attenuation coefficient (Eq. 3.4) and the linear Vs30 coefficient (Eq. 3.7).
fn regional_coeffs(region: Region, c: &Coeffs) -> (f64, f64, f64) {
    match region {
        Region::Global => (c.a1, c.a6, c.a12),
        Region::NewZealand => (c.a35, c.a6 + c.a28, c.a12 + c.a21),
    }
}

/// Shallow site amplification FS, Eq. 3.7 (Erratum), with `pga1000` in g.
fn site_term(c: &Coeffs, a12: f64, vs30: f64, pga1000: f64) -> f64 {
    let vnorm = vs30.min(VS30_MAX) / c.vlin;
    let linear = a12 * vnorm.ln();
    if vnorm >= 1.0 {
        linear + c.b * N * vnorm.ln()
    } else {
        linear + c.b * ((pga1000 + C * vnorm.powf(N)).ln() - (pga1000 + C).ln())
    }
}

/// ln of the median for the coefficients `c`, Eq. 3.1 without the basin term
/// (zero for GLO and NZL). `pga1000` is the reference-rock PGA in g (only used
/// by the nonlinear site term) and `fba` the NZ backarc term.
fn ln_median(c: &Coeffs, region: Region, row: &AG20Row, vs30: f64, pga1000: f64, fba: f64) -> f64 {
    let (mag, rrup) = match *row {
        AG20Row::Interface { mag, rrup, .. } | AG20Row::Slab { mag, rrup, .. } => (mag, rrup),
    };
    let (a1, a6, a12) = regional_coeffs(region, c);
    // Magnitude hinge C1 and slope a4 of Eq. 3.3 (the slab hinge is regional).
    let (c1, a4) = match row {
        AG20Row::Interface { .. } => (c.c1i, A4),
        AG20Row::Slab { .. } => (
            match region {
                Region::Global => 7.5,
                Region::NewZealand => 8.0,
            },
            A4 + A45,
        ),
    };
    // Eq. 3.3. The slope above the hinge (a5) is 0.
    let f_mag = c.a13 * (10.0 - mag).powi(2) + if mag <= c1 { a4 * (mag - c1) } else { 0.0 };
    // Eq. 3.1: magnitude-dependent geometric spreading, and the saturation
    // term hff (Eq. 3.2), which the slab scaling below reuses.
    let hff = C4 * (A9 * (mag - 6.0)).exp();
    let f_r = (c.a2 + A3 * (mag - 7.0)) * (rrup + hff).ln();
    // Eq. 3.5 (Erratum) and Eq. 3.6: slab-only terms.
    let slab = match *row {
        AG20Row::Interface { .. } => 0.0,
        AG20Row::Slab { ztor, .. } => {
            let f_slab = c.a10 + (A4 + A45) * (c1 - 7.5) + c.a14 * (rrup + hff).ln();
            let f_dep = if ztor > 200.0 {
                c.a11 * 150.0
            } else if ztor <= 50.0 {
                c.a8 * (ztor - 50.0)
            } else {
                c.a11 * (ztor - 50.0)
            };
            f_slab + f_dep
        }
    };
    a1 + f_mag + f_r + a6 * rrup + slab + site_term(c, a12, vs30, pga1000) + fba
}

/// Mean + standard deviations for `AbrahamsonGulerce2020SInter` / `SSlab`
/// (`region`) with the NZ22 backarc term and epistemic shift (the shift is
/// ignored for the base model by passing 0, and the backarc term is zero unless
/// the slab row is flagged). One value per rupture-site pair in `inputs`.
/// SA periods outside the table are an error. Exposed to Python as `gm_models._core.ag20`.
#[pyfunction]
pub fn ag20(
    py: Python<'_>,
    inputs: PyRef<'_, AG20Inputs>,
    im: IM,
    new_zealand_region: bool,
    sigma_mu_epsilon: f64,
) -> PyResult<GMMResult> {
    let region = if new_zealand_region {
        Region::NewZealand
    } else {
        Region::Global
    };
    let coeffs = Coeffs::for_im(im)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays, so no Python object
    // is touched while other threads run.
    Ok(py.detach(|| {
        inputs
            .iter()?
            .map(|row| compute(im, &coeffs, region, &row, sigma_mu_epsilon))
            .collect::<Result<GMMResult, GMMError>>()
    })?)
}

/// Computes mean + standard deviations for a single rupture-site pair.
///
/// `c` must be `Coeffs::for_im(im)`; it is passed in so callers looping over
/// sites look it up once rather than per site. Errors only if the backarc
/// term's period interpolation falls outside its table.
pub fn compute(
    im: IM,
    c: &Coeffs,
    region: Region,
    row: &AG20Row,
    sigma_mu_epsilon: f64,
) -> Result<MeanStd, GMMError> {
    let c_pga = &PGA_COEFFS;
    let (rrup, vs30, backarc) = match *row {
        AG20Row::Interface { rrup, vs30, .. } => (rrup, vs30, false),
        AG20Row::Slab {
            rrup,
            vs30,
            backarc,
            ..
        } => (rrup, vs30, backarc),
    };
    // NZ backarc term (slab backarc sites only), for the reference-rock PGA
    // and for the target IM.
    let (fba_pga, fba) = if backarc {
        (backarc_term(IM::PGA {}, rrup)?, backarc_term(im, rrup)?)
    } else {
        (0.0, 0.0)
    };

    // Reference-rock PGA (Vs30 = 1000 m/s, so the site term is linear and
    // `pga1000` is not needed), in g.
    let pga1000 = ln_median(c_pga, region, row, VS30_MAX, 0.0, fba_pga).exp();
    let mut mean = ln_median(c, region, row, vs30, pga1000, fba);
    if sigma_mu_epsilon != 0.0 {
        // Epistemic adjustment, Eq. 6.1.
        let r = rrup.clamp(50.0, 500.0) / 100.0;
        mean += sigma_mu_epsilon * (c.e1 + c.e2 * r + c.e3 * r.powi(2));
    }

    // Linear phi, Eq. 5.2. `max`/`min` rather than `clamp`, which panics for
    // d2 < 0 where `np.clip` does not.
    let phi_lin = |c: &Coeffs| {
        (c.d1 + c.d2 * (rrup - 150.0) / 300.0)
            .max(c.d1)
            .min(c.d1 + c.d2)
            .sqrt()
    };
    let (phi_lin_im, phi_lin_pga) = (phi_lin(c), phi_lin(c_pga));
    let (tau, phi) = if vs30.min(VS30_MAX) < c.vlin {
        // Nonlinear site response, Eqs. 5.7 to 5.9 (Erratum). vnorm <= 1 here.
        let vnorm = vs30.min(VS30_MAX) / c.vlin;
        let dfsite_dlnpga =
            c.b * pga1000 * (-1.0 / (pga1000 + C) + 1.0 / (pga1000 + C * vnorm.powf(N)));
        let phi_b = (phi_lin_im.powi(2) - PHI_AMP.powi(2)).sqrt();
        let phi_b_pga = (phi_lin_pga.powi(2) - PHI_AMP.powi(2)).sqrt();
        // OpenQuake uses tau_lin for both the target and PGA tau here.
        let tau_nl = TAU_LIN.powi(2) * (1.0 + dfsite_dlnpga.powi(2) + 2.0 * dfsite_dlnpga * c.rhob);
        let phi_nl = phi_lin_im.powi(2)
            + dfsite_dlnpga.powi(2) * phi_b_pga.powi(2)
            + 2.0 * dfsite_dlnpga * phi_b_pga * phi_b * c.rhow;
        (tau_nl.sqrt(), phi_nl.sqrt())
    } else {
        (TAU_LIN, phi_lin_im)
    };

    Ok(MeanStd {
        mean,
        total: tau.hypot(phi),
        inter: tau,
        intra: phi,
    })
}

/// Layer 1 verification: compares this module's output directly against
/// OpenQuake's reference tables (Fortran code from N. Abrahamson, and GNS
/// tables for the NZ22 variant), read from the `external/oq-engine`
/// submodule. `#[ignore]`d so a plain `cargo test` works without the
/// submodule; run them with `cargo test -- --ignored`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn check(file: &str, slab: bool, region: Region, pick: fn(&MeanStd) -> f64) {
        check_csv(
            file,
            |rec: &HashMap<&str, &str>| {
                let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
                let (mag, rrup, vs30) = (num("rup_mag"), num("dist_rrup"), num("site_vs30"));
                if slab {
                    AG20Row::Slab {
                        mag,
                        rrup,
                        vs30,
                        ztor: num("rup_ztor"),
                        // Only the NZ22 GNS slab tables have a backarc column.
                        backarc: rec
                            .get("site_backarc")
                            .is_some_and(|b| b.trim().eq_ignore_ascii_case("true")),
                    }
                } else {
                    AG20Row::Interface { mag, rrup, vs30 }
                }
            },
            |im, row| pick(&compute(im, &Coeffs::for_im(im).unwrap(), region, row, 0.0).unwrap()),
        );
    }

    /// AG_20_NZ (base model, NZL) and AG_20 without backarc (base model, GLO).
    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn base_model_matches_openquake() {
        for (code, region) in [("NZL", Region::NewZealand), ("GLO", Region::Global)] {
            for (tect, slab) in [("INTERFACE", false), ("INSLAB", true)] {
                let file = |stat: &str| format!("ag2020/AG2020_{tect}_{code}__{stat}.csv");
                check(&file("MEAN"), slab, region, |r| r.mean.exp());
                check(&file("TOTAL_STDDEV"), slab, region, |r| r.total);
                check(&file("INTER_EVENT_STDDEV"), slab, region, |r| r.inter);
                check(&file("INTRA_EVENT_STDDEV"), slab, region, |r| r.intra);
            }
        }
    }

    /// AG_20 (NZ22 variant, GLO) including the backarc term on slab.
    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn nz22_matches_openquake() {
        for (tect, slab) in [("INTERFACE", false), ("SLAB", true)] {
            let file = |stat: &str| format!("nz22/ag2020/AG20_{tect}_GLO_GNS_{stat}.csv");
            check(&file("MEAN"), slab, Region::Global, |r| r.mean.exp());
            check(&file("TOTAL_STDDEV"), slab, Region::Global, |r| r.total);
        }
    }
}

/// Property tests over the supported input domain. These need no external
/// data, so they run in a plain `cargo test`.
#[cfg(test)]
mod proptests {
    use super::*;
    use approx::relative_eq;
    use coeffs::{SA_COEFFS, SA_PERIODS};
    use proptest::prelude::*;

    fn im() -> impl Strategy<Value = IM> {
        prop_oneof![
            Just(IM::PGA {}),
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    fn row() -> impl Strategy<Value = AG20Row> {
        let common = (5.0..9.5f64, 0.1..1000.0f64, 100.0..2000.0f64);
        prop_oneof![
            common
                .clone()
                .prop_map(|(mag, rrup, vs30)| AG20Row::Interface { mag, rrup, vs30 }),
            (common, 0.0..300.0f64, any::<bool>()).prop_map(
                |((mag, rrup, vs30), ztor, backarc)| AG20Row::Slab {
                    mag,
                    rrup,
                    vs30,
                    ztor,
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
            nz in any::<bool>(),
            sigma_mu_epsilon in -1.2815..=1.2815f64,
        ) {
            let region = if nz { Region::NewZealand } else { Region::Global };
            let c = Coeffs::for_im(im).unwrap();
            let r = compute(im, &c, region, &row, sigma_mu_epsilon).unwrap();
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }

    #[test]
    fn tabulated_periods_return_exact_rows_and_pgv_is_rejected() {
        for (&period, row) in SA_PERIODS.iter().zip(SA_COEFFS.iter()) {
            assert_eq!(Coeffs::for_im(IM::SA { period }).unwrap(), *row);
        }
        assert!(Coeffs::for_im(IM::PGV {}).is_err());
    }

    #[test]
    fn inputs_reject_mismatched_lengths() {
        let slab = AG20Inputs::Slab {
            mag: vec![7.0, 8.0],
            rrup: vec![10.0, 20.0],
            vs30: vec![400.0, 500.0],
            ztor: vec![30.0, 40.0],
            backarc: vec![true],
        };
        assert!(slab.iter().is_err());
    }
}
