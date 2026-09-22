//! Transcribed verbatim from `openquake.hazardlib.gsim.nz22.const`
//! (`openquake/hazardlib/gsim/nz22/const.py`).

/// Correlation-coefficient periods from AG20, used to interpolate `rho_W`/`rho_B`
/// in the NZ "modified sigma" nonlinear standard-deviation model.
pub static PERIODS_AG20: [f64; 24] = [
    0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0,
    2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0,
];

pub static RHO_WS: [f64; 24] = [
    1.0, 0.99, 0.99, 0.97, 0.95, 0.92, 0.9, 0.87, 0.84, 0.82, 0.74, 0.66, 0.59, 0.5, 0.41, 0.33,
    0.3, 0.27, 0.25, 0.22, 0.19, 0.17, 0.14, 0.1,
];

pub static RHO_BS: [f64; 24] = [
    1.0, 0.99, 0.99, 0.985, 0.98, 0.97, 0.96, 0.94, 0.93, 0.91, 0.86, 0.8, 0.78, 0.73, 0.69, 0.62,
    0.56, 0.52, 0.495, 0.43, 0.4, 0.37, 0.32, 0.28,
];

/// Periods for the backarc-term theta7/theta8 interpolation. Note
/// `PERIODS[0] == 0.0` is a placeholder duplicating `PERIODS[1]`'s
/// theta7/theta8 value; hazardlib's own interpolation table drops it
/// (`periods[1:]`), which is what `crate::p21::interp_theta` reproduces.
pub static PERIODS: [f64; 23] = [
    0.0, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 2.5,
    3.0, 4.0, 5.0, 6.0, 7.5, 10.0,
];

pub static THETA7S: [f64; 23] = [
    1.0988, 1.0988, 1.2536, 1.4175, 1.3997, 1.3582, 1.1648, 0.994, 0.8821, 0.7046, 0.5799, 0.5021,
    0.3687, 0.1746, -0.082, -0.2821, -0.4108, -0.4466, -0.4344, -0.4368, -0.4586, -0.4433, -0.4828,
];

pub static THETA8S: [f64; 23] = [
    -1.42, -1.42, -1.65, -1.8, -1.8, -1.69, -1.49, -1.3, -1.18, -0.98, -0.82, -0.7, -0.54, -0.34,
    -0.05, 0.12, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
];
