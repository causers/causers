"""Tests for package-level functionality."""

import pytest
import causers
from io import StringIO
import sys


class TestPackage:
    """Test suite for package-level functions and attributes."""
    
    def test_package_version(self):
        """Test that package has a version."""
        assert hasattr(causers, '__version__')
        assert isinstance(causers.__version__, str)
        assert len(causers.__version__) > 0
        
    def test_package_exports(self):
        """Test that package exports the expected API."""
        assert hasattr(causers, 'LinearRegressionResult')
        assert hasattr(causers, 'linear_regression')
        
        # Check __all__ is properly defined
        assert hasattr(causers, '__all__')
        assert 'LinearRegressionResult' in causers.__all__
        assert 'linear_regression' in causers.__all__
    
    def test_about_function(self):
        """Test the about() function prints package info."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            causers.about()
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # Check that output contains expected information
        assert "causers version" in output
        assert causers.__version__ in output
        assert "High-performance statistical operations" in output
        assert "Polars DataFrames" in output
        assert "Rust" in output or "PyO3" in output
    
    def test_imports_work(self):
        """Test that we can import the main functions."""
        from causers import LinearRegressionResult, linear_regression
        
        # Check they are callable/instantiable
        assert callable(linear_regression)
        assert LinearRegressionResult is not None
    
    def test_no_unexpected_exports(self):
        """Test that we don't accidentally export internal modules."""
        # Check that private/internal attributes are not in __all__
        public_attrs = set(causers.__all__)
        all_attrs = set(dir(causers))
        
        # Remove standard Python module attributes
        standard_attrs = {'__name__', '__doc__', '__package__', '__loader__', 
                         '__spec__', '__path__', '__file__', '__cached__',
                         '__builtins__', '__version__', '__all__', 'about', '_causers'}
        
        exported_attrs = all_attrs - standard_attrs
        
        # All exported non-standard attributes should be in __all__
        for attr in exported_attrs:
            if not attr.startswith('_'):  # Skip private attributes
                assert attr in public_attrs, f"Public attribute {attr} not in __all__"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])