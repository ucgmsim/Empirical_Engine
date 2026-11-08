//! Numerical helpers shared by all ground motion models.

use crate::error::GMMError;
use crate::im::IM;

/// Linear interpolation of `ys` in `ln(x)`, matching OpenQuake's
/// `scipy.interpolate.interp1d(np.log(xs), ys)` (linear kind, default
/// `bounds_error=True`): `x` outside `[xs[0], xs[last]]` (or NaN) is an error.
/// `xs` must be ascending and positive.
pub fn log_interp(xs: &[f64], ys: &[f64], x: f64) -> Result<f64, GMMError> {
    let (min, max) = (xs[0], xs[xs.len() - 1]);
    if !(min..=max).contains(&x) {
        return Err(GMMError::OutOfInterpolationRange { x, min, max });
    }
    // First index with xs[i] >= x, moved to 1 at x == xs[0] so i - 1 is valid.
    let i = xs.partition_point(|&v| v < x).max(1);
    let (x0, x1) = (xs[i - 1].ln(), xs[i].ln());
    Ok(ys[i - 1] + (x.ln() - x0) / (x1 - x0) * (ys[i] - ys[i - 1]))
}

/// Checks that all `(name, length)` columns have the length of the first,
/// and returns that common length.
pub fn check_lengths(columns: &[(&'static str, usize)]) -> Result<usize, GMMError> {
    let expected = columns[0].1;
    match columns.iter().find(|c| c.1 != expected) {
        Some(&(name, len)) => Err(GMMError::LengthMismatch {
            name,
            len,
            expected,
        }),
        None => Ok(expected),
    }
}

/// ZA_06's opt-in `oq_wrapper` long-period extrapolation (`_oq_run_pSA`): SA at a period
/// above the model's largest tabulated period `max` is evaluated at `max`, and
/// 2 ln(max / period) is added to its ln mean (constant spectral displacement).
/// Returns the IM to evaluate and that ln-mean shift (0 for anything else).
pub fn extrapolate_sa(im: IM, max: f64) -> (IM, f64) {
    match im {
        IM::SA { period } if period > max => (IM::SA { period: max }, 2.0 * (max / period).ln()),
        _ => (im, 0.0),
    }
}

/// Epistemic standard deviation of the NGA-West2 models (ASK14, BSSA14,
/// CB14, CY14) and Bradley (2013), OpenQuake's
/// `abrahamson_2014.get_epistemic_sigma`: the USGS-2014 approach
/// (0.4 sqrt(n / N), n = 2, N = number of events in the magnitude-distance
/// bin) with bin counts from the NZ SMDB v1.0. Magnitudes below 5 take the
/// 5 <= M < 6 values. Models add `sigma_mu_epsilon` times this to the ln mean.
pub fn nga_west2_epistemic_sigma(mag: f64, rrup: f64) -> f64 {
    // Event counts for rrup <= 10, 10 < rrup < 30, rrup >= 30 km.
    let counts: [f64; 3] = if mag < 6.0 {
        [11.0, 38.0, 94.0]
    } else if mag < 7.0 {
        [2.0, 7.0, 13.0]
    } else {
        [2.0, 2.0, 4.0]
    };
    let count = if rrup <= 10.0 {
        counts[0]
    } else if rrup < 30.0 {
        counts[1]
    } else {
        counts[2]
    };
    0.4 * (2.0 / count).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const XS: [f64; 3] = [0.1, 1.0, 10.0];
    const YS: [f64; 3] = [1.0, 2.0, 4.0];

    #[test]
    fn knots_return_table_values() {
        for (&x, &y) in XS.iter().zip(YS.iter()) {
            assert_eq!(log_interp(&XS, &YS, x).unwrap(), y);
        }
    }

    #[test]
    fn interpolates_in_log_space() {
        let mid = (0.1f64 * 1.0).sqrt();
        assert_relative_eq!(log_interp(&XS, &YS, mid).unwrap(), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn extrapolates_only_above_max() {
        let (im, shift) = extrapolate_sa(IM::SA { period: 20.0 }, 10.0);
        assert_eq!(im, IM::SA { period: 10.0 });
        assert_relative_eq!(shift, 2.0 * 0.5f64.ln());
        assert_eq!(
            extrapolate_sa(IM::SA { period: 5.0 }, 10.0),
            (IM::SA { period: 5.0 }, 0.0)
        );
        assert_eq!(extrapolate_sa(IM::PGA {}, 10.0), (IM::PGA {}, 0.0));
    }

    #[test]
    fn mismatched_lengths_are_errors() {
        assert_eq!(check_lengths(&[("a", 3), ("b", 3)]).unwrap(), 3);
        assert!(check_lengths(&[("a", 3), ("b", 2)]).is_err());
    }

    #[test]
    fn outside_range_and_nan_are_errors() {
        for x in [0.09, 10.1, f64::NAN] {
            assert!(log_interp(&XS, &YS, x).is_err());
        }
    }
}
