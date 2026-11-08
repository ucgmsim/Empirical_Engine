//! Chiou and Youngs (2014) NGA-West2 GMPE ("CY_14" in `oq_wrapper`), with
//! OpenQuake's `sigma_mu_epsilon` epistemic adjustment.
//!
//! Transcribed from OpenQuake's `openquake.hazardlib.gsim.chiou_youngs_2014`
//! (`ChiouYoungs2014`), specialised to the default constructor arguments, the
//! only configuration `oq_wrapper` instantiates it with (for both active
//! shallow and volcanic tectonic types): `region="CAL"`, `use_hw=True`,
//! `add_delta_c1=False`, `alpha_nm=1`, no host-to-target source/path tables,
//! no USGS or CyberShake basin scaling, no PEER or EMME24 variants. That means
//! the Japan, Italy and Wenchuan branches and the Boore et al. (2022)
//! adjustments are omitted, and the directivity term (`rcdpp`, not an input
//! of the default class) and centred DPP are always 0. The coefficients that
//! OpenQuake takes from its `CONSTANTS` dict (c2, c4, c4a, crb, c11, phi6) are
//! constants here.
//!
//! Equation numbers ("Eq. N") refer to Chiou and Youngs (2014), "Update of the
//! Chiou and Youngs NGA Model for the Average Horizontal Component of Peak
//! Ground Motion and Response Spectra", Earthquake Spectra 30(3).
//!
//! OpenQuake quirk reproduced here: for SA with T <= 0.3 s the median is
//! floored at the PGA median (CY14 p. 1144), before the epistemic term is
//! added. PGV is not floored. `z1pt0 == -999` means "unknown" and is replaced
//! by the Vs30-based estimate (a metre-unit value is expected, as `oq_wrapper`
//! converts km to m).
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

// Tabulated coefficients that happen to look like math constants.
#[allow(clippy::approx_constant)]
pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/cy14_coefficients.rs"));
}

use crate::error::GMMError;
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use crate::utils::{check_lengths, nga_west2_epistemic_sigma};
use coeffs::{Coeffs, PGA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Reference Vs30 (m/s).
const REF_VS: f64 = 1130.0;
/// c2, c4, c4a, crb of Eq. 11 (period independent).
const C2: f64 = 1.06;
const C4: f64 = -2.1;
const C4A: f64 = -0.5;
const CRB: f64 = 50.0;
/// phi6 of Eq. 12.
const PHI6: f64 = 300.0;

/// A single rupture-site pair. `z1pt0` is in metres, `vs30measured` is the
/// "measured" flag of the sigma model (Eq. 13), angles are in degrees.
#[derive(Debug, Clone, Copy)]
pub struct CY14Row {
    pub mag: f64,
    pub dip: f64,
    pub rake: f64,
    pub ztor: f64,
    pub rrup: f64,
    pub rjb: f64,
    pub rx: f64,
    pub vs30: f64,
    pub vs30measured: bool,
    pub z1pt0: f64,
}

/// CY14 input columns, one value per rupture-site pair. Exposed to Python as
/// `gm_models._core.CY14Inputs`. Column lengths are checked in
/// [`CY14Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub struct CY14Inputs {
    mag: Vec<f64>,
    dip: Vec<f64>,
    rake: Vec<f64>,
    ztor: Vec<f64>,
    rrup: Vec<f64>,
    rjb: Vec<f64>,
    rx: Vec<f64>,
    vs30: Vec<f64>,
    vs30measured: Vec<bool>,
    z1pt0: Vec<f64>,
}

impl CY14Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = CY14Row> + '_, GMMError> {
        let n = check_lengths(&[
            ("mag", self.mag.len()),
            ("dip", self.dip.len()),
            ("rake", self.rake.len()),
            ("ztor", self.ztor.len()),
            ("rrup", self.rrup.len()),
            ("rjb", self.rjb.len()),
            ("rx", self.rx.len()),
            ("vs30", self.vs30.len()),
            ("vs30measured", self.vs30measured.len()),
            ("z1pt0", self.z1pt0.len()),
        ])?;
        Ok((0..n).map(move |i| CY14Row {
            mag: self.mag[i],
            dip: self.dip[i],
            rake: self.rake[i],
            ztor: self.ztor[i],
            rrup: self.rrup[i],
            rjb: self.rjb[i],
            rx: self.rx[i],
            vs30: self.vs30[i],
            vs30measured: self.vs30measured[i],
            z1pt0: self.z1pt0[i],
        }))
    }
}

#[pymethods]
impl CY14Inputs {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mag: PyReadonlyArray1<f64>,
        dip: PyReadonlyArray1<f64>,
        rake: PyReadonlyArray1<f64>,
        ztor: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        rjb: PyReadonlyArray1<f64>,
        rx: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        vs30measured: PyReadonlyArray1<bool>,
        z1pt0: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            mag: mag.as_array().to_vec(),
            dip: dip.as_array().to_vec(),
            rake: rake.as_array().to_vec(),
            ztor: ztor.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            rjb: rjb.as_array().to_vec(),
            rx: rx.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            vs30measured: vs30measured.as_array().to_vec(),
            z1pt0: z1pt0.as_array().to_vec(),
        }
    }
}

/// Mean ln(ztor) of the magnitude- and style-dependent ztor model, used to
/// centre ztor in Eq. 11 (`_get_centered_ztor`).
fn mean_ztor(mag: f64, rake: f64) -> f64 {
    let reverse = (30.0..=150.0).contains(&rake);
    let (a, b, m0) = if reverse {
        (2.704, 1.226, 5.849)
    } else {
        (2.673, 1.136, 4.970)
    };
    (a - b * (mag - m0).max(0.0)).max(0.0).powi(2)
}

/// Vs30-dependent mean z1pt0 (m) of the non-Japan relationship, CY14 Eq. 1.
fn mean_z1pt0(vs30: f64) -> f64 {
    ((-7.15 / 4.0)
        * ((vs30.powi(4) + 570.94f64.powi(4)) / (1360f64.powi(4) + 570.94f64.powi(4))).ln())
    .exp()
}

/// ln of the reference-rock median, Eq. 11 (California far-field scaling,
/// hanging wall term on, no directivity term).
fn ln_y_ref(c: &Coeffs, r: &CY14Row) -> f64 {
    let (mag, rrup) = (r.mag, r.rrup);
    let reverse = (30.0..=150.0).contains(&r.rake);
    let normal = (-120.0..=-60.0).contains(&r.rake);
    let coshm = (2.0 * (mag - 4.5).max(0.0)).cosh();
    let cos_dip = r.dip.to_radians().cos();

    let source = if reverse { c.c1a + c.c1c / coshm } else { 0.0 }
        + if normal { c.c1b + c.c1d / coshm } else { 0.0 }
        + (c.c7 + c.c7b / coshm) * (r.ztor - mean_ztor(mag, r.rake))
        + c.c11b / coshm * cos_dip.powi(2);
    let magnitude =
        C2 * (mag - 6.0) + ((C2 - c.c3) / c.cn) * (1.0 + (c.cn * (c.cm - mag)).exp()).ln();
    let path = C4 * (rrup + c.c5 * (c.c6 * (mag - c.chm).max(0.0)).cosh()).ln()
        + (C4A - C4) * rrup.hypot(CRB).ln()
        + (c.cg1 + c.cg2 / (mag - c.cg3).max(0.0).cosh()) * rrup;
    let hanging_wall = if r.rx >= 0.0 {
        let fdist = (1.0 - r.rjb.hypot(r.ztor) / (rrup + 1.0))
            * (c.c9a + (1.0 - c.c9a) * (r.rx / c.c9b).tanh());
        c.c9 * cos_dip * fdist
    } else {
        0.0
    };
    c.c1 + source + magnitude + path + hanging_wall
}

/// Mean (without epistemic term) and total/inter/intra standard deviations
/// for one IM's coefficients, Eqs. 12 and 13.
fn mean_stddevs(c: &Coeffs, r: &CY14Row) -> MeanStd {
    let ln_y_ref = ln_y_ref(c, r);
    let y_ref = ln_y_ref.exp();

    let mut z1pt0 = r.z1pt0;
    let z1pt0_mean = mean_z1pt0(r.vs30);
    if z1pt0 == -999.0 {
        z1pt0 = z1pt0_mean;
    }
    let dz1pt0 = if z1pt0 <= 0.0 {
        0.0
    } else {
        z1pt0 - z1pt0_mean
    };
    let f_basin = c.phi5 * (1.0 - (-dz1pt0 / PHI6).exp());
    let f_lin = c.phi1 * (r.vs30 / REF_VS).ln().min(0.0);
    let f_nl_scaling = c.phi2
        * ((c.phi3 * (r.vs30.min(REF_VS) - 360.0)).exp() - (c.phi3 * (REF_VS - 360.0)).exp());
    let f_nl = ((y_ref + c.phi4) / c.phi4).ln() * f_nl_scaling;

    // Eq. 13: nl0 is the derivative of the nonlinear term w.r.t. ln(y_ref).
    let nl0 = f_nl_scaling * (y_ref / (y_ref + c.phi4));
    let mag_test = (r.mag - 5.0).clamp(0.0, 1.5);
    let tau = c.tau1 + (c.tau2 - c.tau1) / 1.5 * mag_test;
    // OpenQuake adds sig3 or 0.7 (measured) under the square root, not squared.
    let phi_vs = if r.vs30measured { 0.7 } else { c.sig3 };
    let phi = (c.sig1 + (c.sig2 - c.sig1) * mag_test / 1.5) * (phi_vs + (1.0 + nl0).powi(2)).sqrt();
    let inter = ((1.0 + nl0) * tau).abs();

    MeanStd {
        mean: ln_y_ref + f_lin + f_nl + f_basin,
        total: (inter.powi(2) + phi.powi(2)).sqrt(),
        inter,
        intra: phi,
    }
}

/// Mean + standard deviations of `ChiouYoungs2014` (default arguments), one
/// value per rupture-site pair in `inputs`. SA periods outside the table are an
/// error. Exposed to Python as `gm_models._core.cy14`.
#[pyfunction]
pub fn cy14(
    py: Python<'_>,
    inputs: PyRef<'_, CY14Inputs>,
    im: IM,
    sigma_mu_epsilon: f64,
) -> PyResult<GMMResult> {
    let coeffs = Coeffs::for_im(im)?;
    let inputs = &*inputs;
    // Releases the GIL: `inputs` owns copies of the arrays.
    Ok(py.detach(|| {
        inputs
            .iter()?
            .map(|row| Ok(compute(im, &coeffs, &row, sigma_mu_epsilon)))
            .collect::<Result<GMMResult, GMMError>>()
    })?)
}

/// Computes mean + standard deviations for a single rupture-site pair. `c`
/// must be `Coeffs::for_im(im)`.
pub fn compute(im: IM, c: &Coeffs, row: &CY14Row, sigma_mu_epsilon: f64) -> MeanStd {
    let mut r = mean_stddevs(c, row);
    if matches!(im, IM::SA { period } if period <= 0.3) {
        r.mean = r.mean.max(mean_stddevs(&PGA_COEFFS, row).mean);
    }
    r.mean += sigma_mu_epsilon * nga_west2_epistemic_sigma(row.mag, row.rrup);
    r
}

/// Verification against OpenQuake's reference tables (read from the
/// `external/oq-engine` submodule), `#[ignore]`d so a plain `cargo test` works
/// without it. The files are those of OpenQuake's `ChiouYoungs2014TestCase`
/// (default arguments). `CY14_MEDIAN_MS_HW_*` cover the hanging-wall term.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn row(rec: &HashMap<&str, &str>) -> CY14Row {
        let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
        CY14Row {
            mag: num("rup_mag"),
            dip: num("rup_dip"),
            rake: num("rup_rake"),
            ztor: num("rup_ztor"),
            rrup: num("dist_rrup"),
            rjb: num("dist_rjb"),
            rx: num("dist_rx"),
            vs30: num("site_vs30"),
            vs30measured: num("site_vs30measured") != 0.0,
            z1pt0: num("site_z1pt0"),
        }
    }

    fn check(file: &str, pick: fn(&MeanStd) -> f64) {
        check_csv(&format!("NGA/CY14/{file}"), row, |im, row| {
            pick(&compute(im, &Coeffs::for_im(im).unwrap(), row, 0.0))
        });
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn mean_matches_openquake() {
        for file in [
            "CY14_MEDIAN_MS_HW_NM.csv",
            "CY14_MEDIAN_MS_HW_RV.csv",
            "CY14_MEDIAN_MS_HW_SS.csv",
            "cy14_extended_table_mean.csv",
        ] {
            check(file, |r| r.mean.exp());
        }
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn stddevs_match_openquake() {
        check("CY14_INTER_EVENT_SIGMA.csv", |r| r.inter);
        check("CY14_INTRA_EVENT_SIGMA.csv", |r| r.intra);
        for file in [
            "CY14_TOTAL_EVENT_SIGMA.csv",
            "cy14_extended_table_total_stddev.csv",
        ] {
            check(file, |r| r.total);
        }
    }
}

/// Property tests over the supported input domain (no external data).
#[cfg(test)]
mod proptests {
    use super::*;
    use approx::relative_eq;
    use proptest::prelude::*;

    fn im() -> impl Strategy<Value = IM> {
        prop_oneof![
            Just(IM::PGA {}),
            Just(IM::PGV {}),
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    fn row() -> impl Strategy<Value = CY14Row> {
        (
            (3.5..8.5f64, 10.0..90.0f64, -180.0..180.0f64, 0.0..20.0f64),
            (0.1..500.0f64, 0.0..500.0f64, -200.0..200.0f64),
            (150.0..1500.0f64, any::<bool>(), 0.0..2000.0f64),
        )
            .prop_map(
                |((mag, dip, rake, ztor), (rrup, rjb, rx), (vs30, vs30measured, z1pt0))| CY14Row {
                    mag,
                    dip,
                    rake,
                    ztor,
                    rrup,
                    rjb: rjb.min(rrup),
                    rx,
                    vs30,
                    vs30measured,
                    z1pt0,
                },
            )
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            im in im(),
            row in row(),
            sigma_mu_epsilon in -1.2815..=1.2815f64,
        ) {
            let r = compute(im, &Coeffs::for_im(im).unwrap(), &row, sigma_mu_epsilon);
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }

        #[test]
        fn short_period_sa_is_floored_at_pga(row in row(), period in 0.01..=0.3f64) {
            let sa = compute(IM::SA { period }, &Coeffs::for_im(IM::SA { period }).unwrap(), &row, 0.0);
            let pga = compute(IM::PGA {}, &PGA_COEFFS, &row, 0.0);
            prop_assert!(sa.mean >= pga.mean);
        }
    }

    #[test]
    fn inputs_reject_mismatched_lengths() {
        let inputs = CY14Inputs {
            mag: vec![7.0, 8.0],
            dip: vec![90.0],
            rake: vec![0.0, 0.0],
            ztor: vec![0.0, 0.0],
            rrup: vec![1.0, 2.0],
            rjb: vec![1.0, 2.0],
            rx: vec![1.0, 2.0],
            vs30: vec![400.0, 500.0],
            vs30measured: vec![true, false],
            z1pt0: vec![100.0, 200.0],
        };
        assert!(inputs.iter().is_err());
    }
}
