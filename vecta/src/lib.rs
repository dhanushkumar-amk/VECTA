mod python;
pub mod core;

use pyo3::prelude::*;

/// The top-level Python module for vecta.
///
/// All PyO3 function/class registrations happen here,
/// delegating to `python.rs` for the actual implementations.
#[pymodule]
fn vecta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    Ok(())
}
