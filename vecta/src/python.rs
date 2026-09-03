//! PyO3 bindings layer.
//!
//! This is the ONLY file in the crate allowed to import or use pyo3 types.
//! It acts as a thin bridge between the Python world and the pure-Rust core engine.

use pyo3::prelude::*;

/// Placeholder function — confirms the extension module loads correctly.
#[pyfunction]
fn hello_vecta() -> String {
    "vecta engine initialized".to_string()
}

/// Register all Python-exposed functions and classes onto the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_vecta, m)?)?;
    Ok(())
}
