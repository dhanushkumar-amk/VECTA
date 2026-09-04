pub mod core;
mod python;

use pyo3::prelude::*;

/// The top-level Python module for vecta.
///
/// All PyO3 function/class registrations happen here,
/// delegating to `python.rs` for the actual implementations.
#[pymodule]
fn vecta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::FlatIndex>()?;
    m.add_class::<python::IVFIndex>()?;
    m.add_class::<python::HnswIndex>()?;
    m.add_class::<python::IVFPQIndex>()?;
    m.add_class::<python::MetadataStore>()?;
    python::register(m)?;
    Ok(())
}
