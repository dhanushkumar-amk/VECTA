//! Integration tests for the core engine.
//!
//! These tests run with `cargo test` and must never depend on Python or PyO3.

#[test]
fn placeholder_sanity_check() {
    assert_eq!(1 + 1, 2, "basic arithmetic must hold");
}
