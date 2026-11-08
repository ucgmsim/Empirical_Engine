//! Campbell and Bozorgnia (2008) NGA GMPE with the Campbell and Bozorgnia (2010)
//! CAV model ("CB_10" in `oq_wrapper`, because OpenQuake's CB08 class includes
//! the CB10 CAV coefficients).
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.campbell_bozorgnia_2008`
//! (`CampbellBozorgnia2008`, geometric-mean component, default arguments),
//! the only configuration `oq_wrapper` uses (ACTIVE_SHALLOW). The arbitrary
//! component total sigma (`CampbellBozorgnia2008Arbitrary`) and PGD are not
//! ported. There is no epistemic mapping for this model in `oq_wrapper`.
//!
//! Equation numbers ("Eq. N") refer to Campbell and Bozorgnia (2008), Earthquake
//! Spectra 24(1), 139-171. CAV uses the same functional form with its own
//! coefficients.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/cb10_coefficients.rs"));
}

use crate::error::GMMError;
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::utils::check_lengths;
use coeffs::{Coeffs, PGA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// CB10 input columns, one value per rupture-site pair. Angles are in degrees,
/// distances and depths in km. Exposed to Python as `gm_models._core.CB10Inputs`.
/// Column lengths are checked in [`CB10Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub struct CB10Inputs {
    mag: Vec<f64>,
    dip: Vec<f64>,
    rake: Vec<f64>,
    ztor: Vec<f64>,
    rrup: Vec<f64>,
    rjb: Vec<f64>,
    vs30: Vec<f64>,
    z2pt5: Vec<f64>,
}

/// A single row of [`CB10Inputs`].
#[derive(Debug, Clone, Copy)]
pub struct CB10Row {
    pub mag: f64,
    pub dip: f64,
    pub rake: f64,
    pub ztor: f64,
    pub rrup: f64,
    pub rjb: f64,
    pub vs30: f64,
    /// Depth to Vs = 2.5 km/s (km). -999 means unknown (OpenQuake convention),
    /// NaN switches the basin term off (as OpenQuake does).
    pub z2pt5: f64,
}

impl CB10Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = CB10Row> + '_, GMMError> {
        let n = check_lengths(&[
            ("mag", self.mag.len()),
            ("dip", self.dip.len()),
            ("rake", self.rake.len()),
            ("ztor", self.ztor.len()),
            ("rrup", self.rrup.len()),
            ("rjb", self.rjb.len()),
            ("vs30", self.vs30.len()),
            ("z2pt5", self.z2pt5.len()),
        ])?;
        Ok((0..n).map(move |i| CB10Row {
            mag: self.mag[i],
            dip: self.dip[i],
            rake: self.rake[i],
            ztor: self.ztor[i],
            rrup: self.rrup[i],
            rjb: self.rjb[i],
            vs30: self.vs30[i],
            z2pt5: self.z2pt5[i],
        }))
    }
}

#[pymethods]
impl CB10Inputs {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mag: PyReadonlyArray1<f64>,
        dip: PyReadonlyArray1<f64>,
        rake: PyReadonlyArray1<f64>,
        ztor: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        rjb: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        z2pt5: PyReadonlyArray1<f64>,
    ) -> Self {
        let v = |a: PyReadonlyArray1<f64>| a.as_array().to_vec();
        Self {
            mag: v(mag),
            dip: v(dip),
            rake: v(rake),
            ztor: v(ztor),
            rrup: v(rrup),
            rjb: v(rjb),
            vs30: v(vs30),
            z2pt5: v(z2pt5),
        }
    }
}

/// Magnitude term, Eq. 2.
fn magnitude_term(c: &Coeffs, mag: f64) -> f64 {
    let term = if mag > 6.5 {
        c.c2 * (mag - 5.5) + c.c3 * (mag - 6.5)
    } else if mag > 5.5 {
        c.c2 * (mag - 5.5)
    } else {
        0.0
    };
    c.c0 + c.c1 * mag + term
}

/// Style-of-faulting term, Eqs. 4 and 5.
fn style_of_faulting_term(c: &Coeffs, row: &CB10Row) -> f64 {
    let frv = f64::from(row.rake > 30.0 && row.rake < 150.0);
    let fnm = f64::from(row.rake > -150.0 && row.rake < -30.0);
    // The top-of-rupture depth term only applies to reverse faults.
    let ffltz = if frv > 0.0 { row.ztor.min(1.0) } else { 0.0 };
    c.c7 * frv * ffltz + c.c8 * fnm
}

/// Hanging wall term, Eqs. 6 to 10.
fn hanging_wall_term(c: &Coeffs, row: &CB10Row) -> f64 {
    let CB10Row {
        mag,
        dip,
        ztor,
        rrup,
        rjb,
        ..
    } = *row;
    // Eq. 7
    let f_dist = if rjb > 0.0 && ztor < 1.0 {
        let r_max = rrup.max(rjb.hypot(1.0));
        (r_max - rjb) / r_max
    } else if rjb > 0.0 {
        (rrup - rjb) / rrup
    } else {
        1.0
    };
    // Eq. 8
    let f_mag = (2.0 * (mag - 6.0)).clamp(0.0, 1.0);
    // Eq. 9
    let f_depth = if ztor >= 20.0 {
        0.0
    } else {
        (20.0 - ztor) / 20.0
    };
    // Eq. 10
    let f_dip = if dip > 70.0 { (90.0 - dip) / 20.0 } else { 1.0 };
    c.c9 * f_dist * f_mag * f_depth * f_dip
}

/// Basin response term, Eq. 12.
fn basin_term(c: &Coeffs, row: &CB10Row) -> f64 {
    let z2pt5 = if row.z2pt5 == -999.0 {
        // OpenQuake's -999 sentinel: z2pt5 from Vs30 via the Chiou and Youngs
        // (2008) z1pt0 and Campbell and Bozorgnia (2007).
        let z1pt0 = (28.5 - (3.82 / 8.0) * (row.vs30.powi(8) + 378.7f64.powi(8)).ln()).exp();
        (519.0 + 3.595 * z1pt0) / 1000.0
    } else {
        row.z2pt5
    };
    if z2pt5 < 1.0 {
        c.c11 * (z2pt5 - 1.0)
    } else if z2pt5 > 3.0 {
        c.c12 * c.k3 * (-0.75f64).exp() * (1.0 - (-0.25 * (z2pt5 - 3.0)).exp())
    } else {
        0.0
    }
}

/// Shallow site response term, Eq. 11. `pga1100` is PGA on 1100 m/s rock.
fn site_response(c: &Coeffs, vs30: f64, pga1100: f64) -> f64 {
    let stiff_factor = c.c10 + c.k2 * c.n;
    if vs30 >= 1100.0 {
        // Very hard rock takes the constant amplification.
        stiff_factor * (1100.0 / c.k1).ln()
    } else if vs30 < c.k1 {
        let pga_scale = (pga1100 + c.c * (vs30 / c.k1).powf(c.n)).ln() - (pga1100 + c.c).ln();
        c.c10 * (vs30 / c.k1).ln() + c.k2 * pga_scale
    } else {
        stiff_factor * (vs30 / c.k1).ln()
    }
}

/// Mean + standard deviations for OpenQuake's `CampbellBozorgnia2008`, one
/// value per rupture-site pair in `inputs`. SA periods outside the table are an
/// error. Exposed to Python as `gm_models._core.cb10`.
#[pyfunction]
pub fn cb10(py: Python<'_>, inputs: PyRef<'_, CB10Inputs>, im: IM) -> PyResult<GMMResult> {
    let coeffs = Coeffs::for_im(im)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays.
    Ok(py.detach(|| -> Result<GMMResult, GMMError> {
        Ok(inputs
            .iter()?
            .map(|row| compute(im, &coeffs, &row))
            .collect())
    })?)
}

/// Computes mean + standard deviations for a single site/rupture pair.
///
/// `c` must be `Coeffs::for_im(im)`; it is passed in so callers looping over
/// sites look it up once rather than per site.
pub fn compute(im: IM, c: &Coeffs, row: &CB10Row) -> MeanStd {
    let c_pga = &PGA_COEFFS;
    let vs30 = row.vs30;
    // Terms shared by the target IM and the PGA coefficients, as in OpenQuake's
    // `compute` and `_compute_imt1100`.
    let non_site_terms = |c: &Coeffs| {
        magnitude_term(c, row.mag)
            // Eq. 3
            + (c.c4 + c.c5 * row.mag) * row.rrup.hypot(c.c6).ln()
            + style_of_faulting_term(c, row)
            + hanging_wall_term(c, row)
            + basin_term(c, row)
    };

    // PGA on reference rock (Vs30 = 1100 m/s).
    let fsite_1100 = (c_pga.c10 + c_pga.k2 * c_pga.n) * (1100.0 / c_pga.k1).ln();
    let ln_pga1100 = non_site_terms(c_pga) + fsite_1100;
    let pga1100 = ln_pga1100.exp();

    let mut mean = non_site_terms(c) + site_response(c, vs30, pga1100);
    // For 0 < T < 0.25 s, SA cannot fall below PGA at the site.
    // `PGV().period` and `CAV().period` are 0, so they are excluded.
    if matches!(im, IM::SA { period } if period < 0.25) {
        let ln_pga_site = ln_pga1100 - fsite_1100 + site_response(c_pga, vs30, pga1100);
        mean = mean.max(ln_pga_site);
    }

    // Intra-event standard deviation, Eqs. 15 and 17.
    let sig_lnyb = (c.s_lny.powi(2) - c.s_lnaf.powi(2)).sqrt();
    let sig_lnab = (c_pga.s_lny.powi(2) - c.s_lnaf.powi(2)).sqrt();
    let alpha = if vs30 < c.k1 {
        c.k2 * pga1100 * (1.0 / (pga1100 + c.c * (vs30 / c.k1).powf(c.n)) - 1.0 / (pga1100 + c.c))
    } else {
        0.0
    };
    let intra = (sig_lnyb.powi(2)
        + c.s_lnaf.powi(2)
        + alpha.powi(2) * sig_lnab.powi(2)
        + 2.0 * alpha * c.rho * sig_lnyb * sig_lnab)
        .sqrt();
    // Total, Eq. 16.
    MeanStd {
        mean,
        total: intra.hypot(c.t_lny),
        inter: c.t_lny,
        intra,
    }
}

/// Verification against OpenQuake's reference tables (generated from the
/// authors' Fortran implementation), read from the `external/oq-engine`
/// submodule. `#[ignore]`d so a plain `cargo test` works without it. The CAV
/// model is not covered by OpenQuake's reference data.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> CB10Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        CB10Row {
            mag: num("rup_mag"),
            dip: num("rup_dip"),
            rake: num("rup_rake"),
            ztor: num("rup_ztor"),
            rrup: num("dist_rrup"),
            rjb: num("dist_rjb"),
            vs30: num("site_vs30"),
            z2pt5: num("site_z2pt5"),
        }
    }

    fn check(file: &str, pick: fn(&MeanStd) -> f64) {
        check_csv(&format!("CB08/{file}"), row, |im, row| {
            pick(&compute(im, &Coeffs::for_im(im).unwrap(), row))
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        for style in ["RV", "NM", "SS"] {
            check(&format!("CB08_{style}_MEAN.csv"), |r| r.mean.exp());
        }
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn stddevs_match_openquake() {
        for style in ["RV", "NM", "SS"] {
            check(&format!("CB08_{style}_STD_TOTAL.csv"), |r| r.total);
            check(&format!("CB08_{style}_STD_INTER.csv"), |r| r.inter);
            check(&format!("CB08_{style}_STD_INTRA.csv"), |r| r.intra);
        }
    }
}

/// Property tests over the supported input domain. These need no external
/// data, so they run in a plain `cargo test`.
#[cfg(test)]
mod proptests {
    use super::*;
    use approx::relative_eq;
    use proptest::prelude::*;

    fn im() -> impl Strategy<Value = IM> {
        prop_oneof![
            Just(IM::PGA {}),
            Just(IM::PGV {}),
            Just(IM::CAV {}),
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    fn row() -> impl Strategy<Value = CB10Row> {
        (
            (4.0..8.5f64, 10.0..90.0f64, -180.0..180.0f64, 0.0..20.0f64),
            (0.1..300.0f64, 150.0..1500.0f64, 0.0..8.0f64),
        )
            .prop_map(|((mag, dip, rake, ztor), (rrup, vs30, z2pt5))| CB10Row {
                mag,
                dip,
                rake,
                ztor,
                rrup,
                rjb: rrup * 0.9,
                vs30,
                z2pt5,
            })
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(im in im(), row in row()) {
            let r = compute(im, &Coeffs::for_im(im).unwrap(), &row);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }

    #[test]
    fn inputs_reject_mismatched_lengths() {
        let inputs = CB10Inputs {
            mag: vec![7.0],
            dip: vec![45.0],
            rake: vec![90.0],
            ztor: vec![0.0],
            rrup: vec![10.0, 20.0],
            rjb: vec![10.0],
            vs30: vec![400.0],
            z2pt5: vec![1.0],
        };
        assert!(inputs.iter().is_err());
    }
}
