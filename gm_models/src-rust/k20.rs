//! Kuehn et al. (2020) NGA-Subduction GMPE, in the two configurations
//! `oq_wrapper` uses:
//!
//! - "K_20_NZ": `KuehnEtAl2020SInter` / `KuehnEtAl2020SSlab` with
//!   `region="NZL"` ([`k20_nz`]): New Zealand regional coefficients, the NZ
//!   basin term (needs `z1pt0`), the original sigma model.
//! - "K_20": `NZNSHM2022_KuehnEtAl2020SInter` / `NZNSHM2022_KuehnEtAl2020SSlab`
//!   with `region="GLO"` and `modified_sigma=True` ([`k20`]): global
//!   coefficients, the NZ NSHM2022 backarc term for slab sites, the "modified
//!   sigma" nonlinear standard-deviation model, and the epistemic `sigma_mu`
//!   adjustment (`sigma_mu_epsilon`).
//!
//! Both share the base model's mean ([`mean_values`]), so one module serves
//! both; [`Region`] selects the coefficient columns. Specialised to
//! `m_b=None`, `m9_basin_term=None`, `usgs_basin_scaling=False`,
//! `ak23_bias_adj=False`, `ak23_cb14_sig=False`: the magnitude break is the
//! regional default, and the Alaska/USGS options are omitted. Other regions
//! (and the Japan/South America `dm_b` break shift) are omitted. The
//! `sigma_mu` tables are the global ones only, so `sigma_mu_epsilon` must be 0
//! for [`Region::Nzl`] (`k20_nz` always passes 0).
//!
//! The NZ NSHM2022 class never applies a basin term, even for `region="NZL"`
//! (its `get_basin_term` starts from zero depths), so the basin term is driven
//! by `z1pt0` alone: callers that need none pass NaN.
//!
//! Equation numbers ("Eq. N") refer to Kuehn et al. (2021), "Partially
//! Non-Ergodic Ground-Motion Model for Subduction Regions using the
//! NGA-Subduction Database", PEER Report 2020/04, as cited in OpenQuake. The
//! NZ-specific backarc term and modified sigma model are not in that report
//! (see Bradley et al., 2024, BSSA, for the NZ NSHM2022 GMCM).
//!
//! # Verification gap
//!
//! OpenQuake has no reference CSVs for "K_20" itself (GLO coefficients with
//! the backarc term and modified sigma). The `nz22/kuehn2020` CSVs it does have
//! use NZL coefficients, so `nz22_backarc_and_modified_sigma_match_openquake`
//! runs this module's NZ22 code path with [`Region::Nzl`] against them, and
//! the GLO coefficients are checked separately (`global_matches_openquake`).
//! The `oq_wrapper` fixtures cover "K_20" end to end, but only with
//! `backarc=False`. During porting, 300 random scenarios (backarc on and off,
//! M up to 9.3, rrup up to 1250 km, SA up to 12 s, all epistemic branches)
//! were also compared against `oq_wrapper.run_gmm` with no mismatch at
//! rtol 1e-6; that comparison is not part of the test suite.
//!
//! OpenQuake uses `period = 0.0` for PGV as well as PGA. Where that changes
//! the result (backarc term, rho_W/rho_B), the code says so.
//!
//! Derived from OpenQuake hazardlib (Copyright (C) GEM Foundation), licensed
//! under AGPL-3.0-or-later. Transcribed to Rust and modified by ucgmsim, 2026.

pub mod coeffs {
    include!(concat!(env!("OUT_DIR"), "/k20_coefficients.rs"));
}

use crate::error::GMMError;
use crate::im::IM;
use crate::nz22_const::{backarc_term, PERIODS_AG20, RHO_BS, RHO_WS};
use crate::result::{GMMResult, MeanStd};
use crate::utils::{check_lengths, log_interp};
use coeffs::{Coeffs, PGA_COEFFS};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::sync::OnceLock;

/// Constants of Eq. 4.3 to 4.8 (`CONSTS` in OpenQuake): c (Eq. 4.8) and n
/// (Eq. 4.8), delta_m and delta_z (Eq. 4.3), Mref (Eq. 4.4), c_10 (Eq. 4.7).
const C_SITE: f64 = 1.88;
const N_SITE: f64 = 1.18;
const DELTA_M: f64 = 0.1;
const DELTA_Z: f64 = 1.0;
const M_REF: f64 = 6.0;
const C_10: f64 = 0.0;
/// Depth-term reference and break depths (Eq. 4.7), interface then slab.
const Z_REF: [f64; 2] = [15.0, 50.0];
const Z_B: [f64; 2] = [30.0, 80.0];
/// Aleatory amplification variability phi_lnAF (NZ modified sigma).
const PHI_AMP: f64 = 0.3;
/// New Zealand basin-depth model (`Z_MODEL["NZL"]`), Eq. 4.11.
const C_Z: [f64; 4] = [
    6.859789675000001,
    std::f64::consts::LN_10,
    5.745692775,
    0.91563524375,
];

/// Coefficient set: global, or the New Zealand regional columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Region {
    Glo,
    Nzl,
}

/// K20 input columns, one value per rupture-site pair. `backarc` exists only
/// on the `Slab` variant. `z1pt0` is in metres, with -999 meaning "infer from
/// vs30" as in OpenQuake; it only matters for [`k20_nz`]. Exposed to Python as
/// `gm_models._core.K20Inputs`, built with `K20Inputs.interface(...)` or
/// `K20Inputs.slab(...)`. Column lengths are checked in [`K20Inputs::iter`].
#[pyclass(frozen, module = "gm_models._core")]
#[derive(Debug)]
pub enum K20Inputs {
    Interface {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
        ztor: Vec<f64>,
        z1pt0: Vec<f64>,
    },
    Slab {
        mag: Vec<f64>,
        rrup: Vec<f64>,
        vs30: Vec<f64>,
        ztor: Vec<f64>,
        z1pt0: Vec<f64>,
        backarc: Vec<bool>,
    },
}

/// A single row of [`K20Inputs`]. `backarc` is always false for interface rows.
#[derive(Debug, Clone, Copy)]
pub struct K20Row {
    pub slab: bool,
    pub mag: f64,
    pub rrup: f64,
    pub vs30: f64,
    pub ztor: f64,
    pub z1pt0: f64,
    pub backarc: bool,
}

impl K20Inputs {
    /// The rupture-site pairs, in order. Fails if the columns differ in length.
    pub fn iter(&self) -> Result<impl Iterator<Item = K20Row> + '_, GMMError> {
        let (mag, rrup, vs30, ztor, z1pt0, backarc) = match self {
            Self::Interface {
                mag,
                rrup,
                vs30,
                ztor,
                z1pt0,
            } => (mag, rrup, vs30, ztor, z1pt0, None),
            Self::Slab {
                mag,
                rrup,
                vs30,
                ztor,
                z1pt0,
                backarc,
            } => (mag, rrup, vs30, ztor, z1pt0, Some(backarc)),
        };
        let n = check_lengths(&[
            ("mag", mag.len()),
            ("rrup", rrup.len()),
            ("vs30", vs30.len()),
            ("ztor", ztor.len()),
            ("z1pt0", z1pt0.len()),
            ("backarc", backarc.map_or(mag.len(), |b| b.len())),
        ])?;
        Ok((0..n).map(move |i| K20Row {
            slab: backarc.is_some(),
            mag: mag[i],
            rrup: rrup[i],
            vs30: vs30[i],
            ztor: ztor[i],
            z1pt0: z1pt0[i],
            backarc: backarc.is_some_and(|b| b[i]),
        }))
    }
}

#[pymethods]
impl K20Inputs {
    /// Subduction interface inputs.
    #[staticmethod]
    fn interface(
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        ztor: PyReadonlyArray1<f64>,
        z1pt0: PyReadonlyArray1<f64>,
    ) -> Self {
        Self::Interface {
            mag: mag.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            ztor: ztor.as_array().to_vec(),
            z1pt0: z1pt0.as_array().to_vec(),
        }
    }

    /// Subduction slab inputs.
    #[staticmethod]
    fn slab(
        mag: PyReadonlyArray1<f64>,
        rrup: PyReadonlyArray1<f64>,
        vs30: PyReadonlyArray1<f64>,
        ztor: PyReadonlyArray1<f64>,
        z1pt0: PyReadonlyArray1<f64>,
        backarc: PyReadonlyArray1<bool>,
    ) -> Self {
        Self::Slab {
            mag: mag.as_array().to_vec(),
            rrup: rrup.as_array().to_vec(),
            vs30: vs30.as_array().to_vec(),
            ztor: ztor.as_array().to_vec(),
            z1pt0: z1pt0.as_array().to_vec(),
            backarc: backarc.as_array().to_vec(),
        }
    }
}

/// Logistic hinge function, Eq. 4.3.
fn log_hinge(x: f64, x0: f64, a: f64, b0: f64, b1: f64, delta: f64) -> f64 {
    let xdiff = x - x0;
    a + b0 * xdiff + (b1 - b0) * delta * (1.0 + (xdiff / delta).exp()).ln()
}

/// ln of the reference basin depth (m) for `vs30`, Eq. 4.11 (NZ coefficients).
fn ln_z_ref(vs30: f64) -> f64 {
    let e = ((vs30.ln() - C_Z[2]) / C_Z[3]).exp();
    C_Z[0] + (C_Z[1] - C_Z[0]) * e / (1.0 + e)
}

/// Basin response term, Eq. 4.10, for New Zealand (depth to Vs = 1 km/s,
/// `z1pt0`, in metres). -999 means "infer from vs30" in OpenQuake, which makes
/// the term `c_11` (zero basin anomaly). A non-positive or NaN depth gives 0,
/// as does any region but New Zealand.
fn basin_term(c: &Coeffs, region: Region, z1pt0: f64, vs30: f64) -> f64 {
    if region == Region::Glo {
        return 0.0;
    }
    let z = if z1pt0 == -999.0 {
        ln_z_ref(vs30).exp()
    } else {
        z1pt0
    };
    if z > 0.0 {
        c.c_11_nz + c.c_12_nz * (z.ln() - ln_z_ref(vs30))
    } else {
        0.0
    }
}

/// Mean ln ground motion without basin and backarc terms, Eq. 4.1: base term
/// (Eq. 4.1), magnitude scaling (Eq. 4.4), geometric spreading (Eq. 4.5),
/// anelastic attenuation (Eq. 4.9), depth scaling (Eq. 4.7) and the shallow
/// site response (Eq. 4.8), evaluated at `vs30` with the reference-rock PGA
/// `pga1100` driving the nonlinear part.
fn mean_values(c: &Coeffs, row: &K20Row, region: Region, m_b: f64, vs30: f64, pga1100: f64) -> f64 {
    let i = usize::from(row.slab);
    let (c1, c2, c4, c9, dz_b) = if row.slab {
        (
            match region {
                Region::Glo => c.mu_c_1_slab,
                Region::Nzl => c.c_1_slab_reg_nz,
            },
            c.c_2_slab,
            c.c_4_slab,
            c.c_9_slab,
            c.dz_b_slab,
        )
    } else {
        (
            match region {
                Region::Glo => c.mu_c_1_if,
                Region::Nzl => c.c_1_if_reg_nz,
            },
            c.c_2_if,
            c.c_4_if,
            c.c_9_if,
            c.dz_b_if,
        )
    };
    let (c6, c7) = match region {
        Region::Glo => (c.mu_c_6, c.mu_c_7),
        Region::Nzl => (c.c_6_2_reg_nz, c.c_7_reg_nz),
    };

    let fm = log_hinge(row.mag, m_b, c4 * (m_b - M_REF), c4, c.c_5, DELTA_M);
    let h = 10f64.powf(c.c_nft_1 + c.c_nft_2 * (row.mag - 6.0));
    let fgeom = (c2 + c.c_3 * row.mag) * (row.rrup + h).ln();
    let z_break = Z_B[i] + dz_b;
    let fztor = log_hinge(
        row.ztor,
        z_break,
        c9 * (z_break - Z_REF[i]),
        c9,
        C_10,
        DELTA_Z,
    );

    let vs_mod = vs30 / c.k1;
    let fsite = if vs30 > c.k1 {
        c7 * vs_mod.ln() + c.k2 * N_SITE * vs_mod.ln()
    } else {
        c7 * vs_mod.ln()
            + c.k2 * ((pga1100 + C_SITE * vs_mod.powf(N_SITE)).ln() - (pga1100 + C_SITE).ln())
    };

    c1 + fm + fgeom + c6 * row.rrup + fztor + fsite
}

/// NZ NSHM2022 backarc correction, 0 unless `row` is a slab site flagged
/// `backarc`.
fn slab_backarc(im: IM, row: &K20Row) -> Result<f64, GMMError> {
    if row.backarc {
        backarc_term(im, row.rrup)
    } else {
        Ok(0.0)
    }
}

/// Linear-interpolation weights of `x` in the ascending `grid`, clamped to
/// its ends: the lower index and the weight of the upper neighbour.
fn bracket(grid: &[f64], x: f64) -> (usize, f64) {
    let i = grid.partition_point(|&g| g <= x).clamp(1, grid.len() - 1) - 1;
    (i, ((x - grid[i]) / (grid[i + 1] - grid[i])).clamp(0.0, 1.0))
}

/// Within-model epistemic uncertainty sigma_mu tables (global model), embedded
/// from `data/k20_sigma_mu_{if,slab}.csv`: one row per (M, R) pair with
/// columns `M,R,PGA,PGV,<SA periods>`, M-major.
struct SigmaMu {
    mags: Vec<f64>,
    rrups: Vec<f64>,
    periods: Vec<f64>,
    /// `[PGA, PGV, SA...]` per (M, R) pair.
    values: Vec<Vec<f64>>,
}

impl SigmaMu {
    fn parse(csv: &str) -> Self {
        let mut lines = csv.lines();
        let header: Vec<f64> = lines
            .next()
            .unwrap()
            .split(',')
            .skip(4)
            .map(|p| p.parse().unwrap())
            .collect();
        let rows: Vec<Vec<f64>> = lines
            .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
            .collect();
        let mut mags: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        mags.dedup();
        let rrups = rows
            .iter()
            .take(rows.len() / mags.len())
            .map(|r| r[1])
            .collect();
        Self {
            mags,
            rrups,
            periods: header,
            values: rows.into_iter().map(|r| r[2..].to_vec()).collect(),
        }
    }

    /// OpenQuake's `get_sigma_mu_adjustment`: bilinear in (M, R) for PGA and
    /// PGV, trilinear in (M, R, ln T) for SA. OpenQuake pads the tables so
    /// that values outside them are constant, which clamping reproduces.
    fn at(&self, im: IM, mag: f64, rrup: f64) -> f64 {
        let (i, tm) = bracket(&self.mags, mag);
        let (j, tr) = bracket(&self.rrups, rrup);
        let nr = self.rrups.len();
        let corner = |di: usize, dj: usize| &self.values[(i + di) * nr + j + dj];
        let lerp = |a: f64, b: f64, t: f64| a + t * (b - a);
        let bilinear = |k: usize| {
            lerp(
                lerp(corner(0, 0)[k], corner(0, 1)[k], tr),
                lerp(corner(1, 0)[k], corner(1, 1)[k], tr),
                tm,
            )
        };
        match im {
            IM::PGA {} => bilinear(0),
            IM::SA { period } => {
                let ln_t: Vec<f64> = self.periods.iter().map(|p| p.ln()).collect();
                let (k, tt) = bracket(&ln_t, period.ln());
                lerp(bilinear(2 + k), bilinear(2 + k + 1), tt)
            }
            _ => bilinear(1),
        }
    }
}

/// `data/k20_sigma_mu_{if,slab}.csv`, parsed on first use.
fn sigma_mu_table(slab: bool) -> &'static SigmaMu {
    static TABLES: [OnceLock<SigmaMu>; 2] = [OnceLock::new(), OnceLock::new()];
    TABLES[usize::from(slab)].get_or_init(|| {
        SigmaMu::parse(if slab {
            include_str!("../data/k20_sigma_mu_slab.csv")
        } else {
            include_str!("../data/k20_sigma_mu_if.csv")
        })
    })
}

/// NZ "modified sigma" nonlinear standard-deviation model (tau, phi). Sites
/// with `vs30 <= k1` have the nonlinear site term, whose derivative with
/// respect to ln PGA (taken from Abrahamson and Gulerce 2020, with Vsref =
/// k1) propagates the PGA reference-rock variability, following Abrahamson and
/// Gulerce (2022), Earthquake Spectra 38(4); rho_W and rho_B are from their
/// Table E6.
fn nonlinear_stddevs(c: &Coeffs, im: IM, vs30: f64, pga1100: f64) -> Result<(f64, f64), GMMError> {
    if vs30 > c.k1 {
        return Ok((c.tau, c.phi));
    }
    let c_pga = &PGA_COEFFS;
    let (rho_w, rho_b) = match im {
        IM::SA { period } if period >= 0.01 => (
            log_interp(&PERIODS_AG20, &RHO_WS, period)?,
            log_interp(&PERIODS_AG20, &RHO_BS, period)?,
        ),
        // Includes PGV, via OpenQuake's `PGV().period == 0.0`.
        _ => (1.0, 1.0),
    };
    let vnorm = vs30 / c.k1;
    let partial_f_pga = c.k2
        * pga1100
        * (-1.0 / (pga1100 + C_SITE) + 1.0 / (pga1100 + C_SITE * vnorm.powf(N_SITE)));
    let phi_b = (c.phi.powi(2) - PHI_AMP.powi(2)).sqrt();
    let phi_b_pga = (c_pga.phi.powi(2) - PHI_AMP.powi(2)).sqrt();
    let tau2 = c.tau.powi(2)
        + partial_f_pga.powi(2) * c_pga.tau.powi(2)
        + 2.0 * partial_f_pga * c_pga.tau * c.tau * rho_b;
    let phi2 = c.phi.powi(2)
        + partial_f_pga.powi(2) * phi_b_pga.powi(2)
        + 2.0 * partial_f_pga * phi_b_pga * phi_b * rho_w;
    Ok((tau2.sqrt(), phi2.sqrt()))
}

/// Evaluates `inputs` for `region` (shared by [`k20`] and [`k20_nz`]).
fn evaluate(
    py: Python<'_>,
    inputs: &K20Inputs,
    im: IM,
    region: Region,
    sigma_mu_epsilon: f64,
    modified_sigma: bool,
) -> PyResult<GMMResult> {
    let coeffs = Coeffs::for_im(im)?;
    // Releases the GIL: `inputs` owns copies of the arrays, so no Python object
    // is touched while other threads run.
    Ok(py.detach(|| {
        inputs
            .iter()?
            .map(|row| compute(im, &coeffs, &row, region, sigma_mu_epsilon, modified_sigma))
            .collect::<Result<GMMResult, GMMError>>()
    })?)
}

/// Mean + standard deviations for `NZNSHM2022_KuehnEtAl2020SInter` /
/// `NZNSHM2022_KuehnEtAl2020SSlab` (global region, "K_20" in `oq_wrapper`),
/// one value per rupture-site pair in `inputs` (`z1pt0` is ignored). The
/// tectonic type is set by how `inputs` was built. SA periods outside the table
/// are an error. Exposed to Python as `gm_models._core.k20`.
#[pyfunction]
pub fn k20(
    py: Python<'_>,
    inputs: PyRef<'_, K20Inputs>,
    im: IM,
    sigma_mu_epsilon: f64,
    modified_sigma: bool,
) -> PyResult<GMMResult> {
    evaluate(
        py,
        &inputs,
        im,
        Region::Glo,
        sigma_mu_epsilon,
        modified_sigma,
    )
}

/// Mean + standard deviations for `KuehnEtAl2020SInter` / `KuehnEtAl2020SSlab`
/// with `region="NZL"` ("K_20_NZ" in `oq_wrapper`), one value per rupture-site
/// pair in `inputs` (`backarc` is ignored). Exposed to Python as
/// `gm_models._core.k20_nz`.
#[pyfunction]
pub fn k20_nz(py: Python<'_>, inputs: PyRef<'_, K20Inputs>, im: IM) -> PyResult<GMMResult> {
    evaluate(py, &inputs, im, Region::Nzl, 0.0, false)
}

/// Computes mean + standard deviations for a single site/rupture pair.
///
/// `c` must be `Coeffs::for_im(im)`; it is passed in so callers looping over
/// sites look it up once rather than per site. `backarc` is read from `row`
/// whatever the region, and the basin term applies to `Region::Nzl` whenever
/// `row.z1pt0` is not NaN. `modified_sigma` is the NZ NSHM2022 model, the
/// default being the original `sqrt(tau^2 + phi^2)`.
pub fn compute(
    im: IM,
    c: &Coeffs,
    row: &K20Row,
    region: Region,
    sigma_mu_epsilon: f64,
    modified_sigma: bool,
) -> Result<MeanStd, GMMError> {
    let c_pga = &PGA_COEFFS;
    let m_b = match (region, row.slab) {
        (Region::Glo, false) => 7.9,
        (Region::Nzl, false) => 8.3,
        (_, true) => 7.6,
    };

    // PGA on reference rock (Vs30 = 1100 m/s, no nonlinearity), with the
    // backarc term of PGA. No basin term.
    let pga1100 =
        (mean_values(c_pga, row, region, m_b, 1100.0, 0.0) + slab_backarc(IM::PGA {}, row)?).exp();
    // PGA on soil, a floor for PGA and SA(T <= 0.1 s) so SA(T) never falls
    // below PGA.
    let pga_soil = mean_values(c_pga, row, region, m_b, row.vs30, pga1100)
        + basin_term(c_pga, region, row.z1pt0, row.vs30)
        + slab_backarc(IM::PGA {}, row)?;

    let mean_im = mean_values(c, row, region, m_b, row.vs30, pga1100)
        + basin_term(c, region, row.z1pt0, row.vs30)
        + slab_backarc(im, row)?;
    let mean = match im {
        IM::PGA {} => pga_soil,
        IM::SA { period } if period <= 0.1 => mean_im.max(pga_soil),
        _ => mean_im,
    };
    let mean = if sigma_mu_epsilon != 0.0 {
        mean + sigma_mu_epsilon * sigma_mu_table(row.slab).at(im, row.mag, row.rrup)
    } else {
        mean
    };

    let (tau, phi) = if modified_sigma {
        nonlinear_stddevs(c, im, row.vs30, pga1100)?
    } else {
        (c.tau, c.phi)
    };
    Ok(MeanStd {
        mean,
        total: tau.hypot(phi),
        inter: tau,
        intra: phi,
    })
}

/// Layer 1 verification against OpenQuake's reference tables, read from the
/// `external/oq-engine` submodule. `#[ignore]`d so a plain `cargo test` works
/// without the submodule; run them with `cargo test -- --ignored`.
///
/// - `kuehn2020/*_GLOBAL_*` (`KuehnEtAl2020SInter/SSlab`, default constructor)
///   check the shared mean, the original sigma, and `sigma_mu_epsilon = +-1`,
///   which `NZNSHM2022_KuehnEtAl2020*` with `region="GLO"` reproduces exactly
///   when no site is backarc.
/// - `kuehn2020/*_NEWZEALAND_*` (`KuehnEtAl2020*(region="NZL")`) check the
///   K_20_NZ configuration including the basin term.
/// - `nz22/kuehn2020/*_NZL_GNS_*` (`NZNSHM2022_KuehnEtAl2020*(region="NZL")`)
///   check the backarc term and the modified sigma. OpenQuake has no reference
///   tables for the global region of the NZ NSHM2022 classes, so they run the
///   NZL coefficients (the backarc and sigma code is region independent).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::oq_test::check_csv;
    use std::collections::HashMap;

    fn row(slab: bool) -> impl Fn(&HashMap<&str, &str>) -> K20Row {
        move |rec| {
            let num = |k: &str| -> f64 { rec[k].parse().unwrap() };
            K20Row {
                slab,
                mag: num("rup_mag"),
                rrup: num("dist_rrup"),
                vs30: num("site_vs30"),
                ztor: num("rup_ztor"),
                z1pt0: rec
                    .get("site_z1pt0")
                    .map_or(f64::NAN, |v| v.parse().unwrap()),
                backarc: rec
                    .get("site_backarc")
                    .is_some_and(|v| v.parse::<f64>().unwrap() != 0.0),
            }
        }
    }

    fn check(
        file: &str,
        region: Region,
        sigma_mu_epsilon: f64,
        modified_sigma: bool,
        pick: fn(&MeanStd) -> f64,
    ) {
        check_csv(file, row(file.contains("INSLAB")), |im, row| {
            let c = Coeffs::for_im(im).unwrap();
            pick(&compute(im, &c, row, region, sigma_mu_epsilon, modified_sigma).unwrap())
        });
    }

    const TECTS: [&str; 2] = ["INTERFACE", "INSLAB"];

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn global_matches_openquake() {
        for tect in TECTS {
            let f = |name: &str| format!("kuehn2020/KUEHN2020_{tect}_GLOBAL_{name}.csv");
            check(&f("MEAN"), Region::Glo, 0.0, false, |r| r.mean.exp());
            check(&f("epsilon1_MEAN"), Region::Glo, 1.0, true, |r| {
                r.mean.exp()
            });
            check(&f("epsilon-1_MEAN"), Region::Glo, -1.0, true, |r| {
                r.mean.exp()
            });
            check(&f("TOTAL_STDDEV"), Region::Glo, 0.0, false, |r| r.total);
            check(&f("INTER_EVENT_STDDEV"), Region::Glo, 0.0, false, |r| {
                r.inter
            });
            check(&f("INTRA_EVENT_STDDEV"), Region::Glo, 0.0, false, |r| {
                r.intra
            });
        }
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn new_zealand_matches_openquake() {
        for tect in TECTS {
            let f = |name: &str| format!("kuehn2020/KUEHN2020_{tect}_NEWZEALAND_{name}.csv");
            check(&f("MEAN"), Region::Nzl, 0.0, false, |r| r.mean.exp());
            check(&f("TOTAL_STDDEV"), Region::Nzl, 0.0, false, |r| r.total);
            check(&f("INTER_EVENT_STDDEV"), Region::Nzl, 0.0, false, |r| {
                r.inter
            });
            check(&f("INTRA_EVENT_STDDEV"), Region::Nzl, 0.0, false, |r| {
                r.intra
            });
        }
    }

    #[test]
    #[ignore = "needs the external/oq-engine submodule; run with `cargo test -- --ignored`"]
    fn nz22_backarc_and_modified_sigma_match_openquake() {
        for tect in TECTS {
            let f = |name: &str| format!("nz22/kuehn2020/KUEHN20_{tect}_NZL_GNS_{name}.csv");
            check(&f("MEAN"), Region::Nzl, 0.0, false, |r| r.mean.exp());
            check(
                &f("TOTAL_STDDEV_ORIGINAL_SIGMA"),
                Region::Nzl,
                0.0,
                false,
                |r| r.total,
            );
            check(
                &f("TOTAL_STDDEV_MODIFIED_SIGMA"),
                Region::Nzl,
                0.0,
                true,
                |r| r.total,
            );
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
            (0.01..=10.0f64).prop_map(|period| IM::SA { period }),
        ]
    }

    fn row() -> impl Strategy<Value = K20Row> {
        (
            any::<bool>(),
            5.0..9.5f64,
            0.1..1000.0f64,
            100.0..2000.0f64,
            0.0..100.0f64,
            prop_oneof![Just(-999.0), Just(f64::NAN), 0.0..2000.0f64],
            any::<bool>(),
        )
            .prop_map(|(slab, mag, rrup, vs30, ztor, z1pt0, backarc)| K20Row {
                slab,
                mag,
                rrup,
                vs30,
                ztor,
                z1pt0,
                backarc: slab && backarc,
            })
    }

    proptest! {
        #[test]
        fn outputs_are_finite_and_stddevs_consistent(
            im in im(),
            row in row(),
            nzl in any::<bool>(),
            sigma_mu_epsilon in -1.2815..=1.2815f64,
            modified_sigma in any::<bool>(),
        ) {
            let region = if nzl { Region::Nzl } else { Region::Glo };
            let eps = if nzl { 0.0 } else { sigma_mu_epsilon };
            let c = Coeffs::for_im(im).unwrap();
            let r = compute(im, &c, &row, region, eps, modified_sigma).unwrap();
            prop_assert!(r.mean.is_finite());
            prop_assert!(r.inter > 0.0 && r.intra > 0.0);
            let (lhs, rhs) = (r.total.powi(2), r.inter.powi(2) + r.intra.powi(2));
            prop_assert!(relative_eq!(lhs, rhs, max_relative = 1e-12), "total^2 {lhs} != inter^2 + intra^2 {rhs}");
        }
    }

    #[test]
    fn sigma_mu_is_constant_outside_the_tables() {
        let t = sigma_mu_table(false);
        for im in [IM::PGA {}, IM::PGV {}, IM::SA { period: 1.0 }] {
            assert_eq!(t.at(im, 20.0, 5000.0), t.at(im, 9.5, 1000.0));
            assert_eq!(t.at(im, 1.0, 0.1), t.at(im, 4.0, 10.0));
        }
        assert_eq!(
            t.at(IM::SA { period: 100.0 }, 6.0, 50.0),
            t.at(IM::SA { period: 10.0 }, 6.0, 50.0)
        );
    }

    #[test]
    fn inputs_reject_mismatched_lengths() {
        let slab = |backarc: Vec<bool>| K20Inputs::Slab {
            mag: vec![7.0, 8.0],
            rrup: vec![10.0, 20.0],
            vs30: vec![400.0, 500.0],
            ztor: vec![30.0, 40.0],
            z1pt0: vec![f64::NAN; 2],
            backarc,
        };
        assert!(slab(vec![true]).iter().is_err());
        let rows: Vec<K20Row> = slab(vec![false, true]).iter().unwrap().collect();
        assert!(rows[0].slab && !rows[0].backarc && rows[1].backarc);
    }
}
