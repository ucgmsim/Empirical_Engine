//! Numerical helpers shared by all ground motion models.

use crate::error::GMMError;

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((log_interp(&XS, &YS, mid).unwrap() - 1.5).abs() < 1e-12);
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
