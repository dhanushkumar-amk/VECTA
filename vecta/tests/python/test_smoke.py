"""
Smoke tests for the vecta Python extension module.

This file grows with each phase — add new assertions here as new
Python-exposed functions and classes are added in src/python.rs.
The CI workflow (ci.yml) runs this via pytest on every push/PR.
"""

import vecta


class TestPhase1:
    """Phase 1: basic module import and placeholder function."""

    def test_import(self):
        """Module should be importable without errors."""
        assert vecta is not None

    def test_hello_vecta(self):
        """hello_vecta() returns the expected initialization string."""
        result = vecta.hello_vecta()
        assert result == "vecta engine initialized"

    def test_hello_vecta_type(self):
        """hello_vecta() returns a Python str."""
        assert isinstance(vecta.hello_vecta(), str)
