//! What every ground motion model shares: its Python-facing input pyclass
//! ([`row_inputs!`]) and its evaluation over all rupture-site pairs ([`GMM`],
//! [`evaluate`]).

use crate::error::GMMError;
use crate::im::IM;
use crate::result::{GMMResult, MeanStd};
use pyo3::prelude::*;

/// Defines a model's input pyclass `$inputs` (exposed to Python as
/// `gm_models._core.$inputs`), holding one `$row` per rupture-site pair. Each
/// constructor takes one numpy array per column, checks the lengths match and
/// copies the values into rows. Its Python signature names the columns, which
/// the Python side reads to know what to pass.
///
/// Three forms:
///
/// - Enum rows, one variant (and `#[staticmethod]` constructor) per tectonic
///   type: `Inputs, Row { Variant => ctor { field: ty, .. }, .. }`.
/// - Struct rows with a single constructor (`Inputs(...)` in Python), the row
///   struct generated from the fields: `Inputs, Row { field: ty, .. }`.
/// - Constructors for a hand-written row struct, each taking some columns and
///   setting the remaining fields to constants:
///   `Inputs, Row { ctor(field: ty, ..) { field: value, .. }, .. }`.
#[macro_export]
macro_rules! row_inputs {
    (
        $(#[$meta:meta])*
        $inputs:ident, $row:ident {
            $(
                $(#[$vmeta:meta])*
                $variant:ident => $ctor:ident { $($field:ident : $ty:ty),* $(,)? }
            ),* $(,)?
        }
    ) => {
        #[doc = concat!("A single rupture-site pair of [`", stringify!($inputs), "`].")]
        #[derive(Debug, Clone, Copy)]
        pub enum $row { $($variant { $($field: $ty),* }),* }

        $crate::row_inputs!(@pyclass $(#[$meta])* $inputs, $row);

        #[::pyo3::pymethods]
        impl $inputs {
            $(
                $(#[$vmeta])*
                #[staticmethod]
                #[allow(clippy::too_many_arguments)]
                fn $ctor($($field: ::numpy::PyReadonlyArray1<'_, $ty>),*) -> ::pyo3::PyResult<Self> {
                    $crate::row_inputs!(@rows $row::$variant { $($field),* } {})
                }
            )*
        }
    };
    (
        $(#[$meta:meta])*
        $inputs:ident, $row:ident {
            $(
                $(#[$vmeta:meta])*
                $ctor:ident ( $($field:ident : $ty:ty),* $(,)? ) { $($cfield:ident : $cvalue:expr),* $(,)? }
            ),* $(,)?
        }
    ) => {
        $crate::row_inputs!(@pyclass $(#[$meta])* $inputs, $row);

        #[::pyo3::pymethods]
        impl $inputs {
            $(
                $(#[$vmeta])*
                #[staticmethod]
                #[allow(clippy::too_many_arguments)]
                fn $ctor($($field: ::numpy::PyReadonlyArray1<'_, $ty>),*) -> ::pyo3::PyResult<Self> {
                    $crate::row_inputs!(@rows $row { $($field),* } { $($cfield: $cvalue),* })
                }
            )*
        }
    };
    (
        $(#[$meta:meta])*
        $inputs:ident, $row:ident {
            $($(#[$fmeta:meta])* $field:ident : $ty:ty),* $(,)?
        }
    ) => {
        #[doc = concat!("A single rupture-site pair of [`", stringify!($inputs), "`].")]
        #[derive(Debug, Clone, Copy)]
        pub struct $row { $($(#[$fmeta])* pub $field: $ty),* }

        $crate::row_inputs!(@pyclass $(#[$meta])* $inputs, $row);

        #[::pyo3::pymethods]
        impl $inputs {
            #[new]
            #[allow(clippy::too_many_arguments)]
            fn new($($field: ::numpy::PyReadonlyArray1<'_, $ty>),*) -> ::pyo3::PyResult<Self> {
                $crate::row_inputs!(@rows $row { $($field),* } {})
            }
        }
    };
    (@pyclass $(#[$meta:meta])* $inputs:ident, $row:ident) => {
        $(#[$meta])*
        #[::pyo3::pyclass(frozen, module = "gm_models._core")]
        #[derive(Debug)]
        pub struct $inputs {
            pub rows: Vec<$row>,
        }
    };
    // Body of a constructor: `$field` are the numpy arrays, `$cfield` the
    // fields set to constants.
    (@rows $($row:ident)::+ { $($field:ident),* } { $($cfield:ident : $cvalue:expr),* }) => {{
        $(let $field = $field.as_array();)*
        let n = $crate::utils::check_lengths(&[$((stringify!($field), $field.len())),*])?;
        Ok(Self {
            rows: (0..n)
                .map(|i| $($row)::+ { $($field: $field[i],)* $($cfield: $cvalue),* })
                .collect(),
        })
    }};
}

/// A ground motion model, evaluated one rupture-site pair at a time.
pub trait GMM {
    type Row: Sync;
    /// Everything looked up once per IM, e.g. the coefficient table row.
    type Coeffs: Sync;
    /// Per-call settings: `sigma_mu_epsilon`, `modified_sigma`, region, ...
    type Params: Copy + Send + Sync;

    fn coeffs(im: IM, p: Self::Params) -> Result<Self::Coeffs, GMMError>;
    fn compute(
        im: IM,
        c: &Self::Coeffs,
        row: &Self::Row,
        p: Self::Params,
    ) -> Result<MeanStd, GMMError>;
}

/// Evaluates model `M` for every row, releasing the GIL while it runs (the
/// rows are owned copies, so no Python object is touched).
pub fn evaluate<M: GMM>(
    py: Python<'_>,
    rows: &[M::Row],
    im: IM,
    p: M::Params,
) -> PyResult<GMMResult> {
    let c = M::coeffs(im, p)?;
    Ok(py.detach(|| {
        rows.iter()
            .map(|row| M::compute(im, &c, row, p))
            .collect::<Result<GMMResult, GMMError>>()
    })?)
}
