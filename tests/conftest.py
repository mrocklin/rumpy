"""Pytest configuration and fixtures for rumpy tests."""

import pytest

# Supported dtypes for parametrized tests
DTYPES = ["float32", "float64", "int32", "int64", "bool"]


@pytest.fixture(params=DTYPES)
def dtype(request):
    """Parametrized fixture for all supported dtypes."""
    return request.param
