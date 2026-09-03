fn main() {
    // Configures the linker for PyO3 extension modules across platforms (specifically macOS dynamic lookup)
    pyo3_build_config::add_extension_module_link_args();
}
