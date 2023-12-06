"""Test module for litminer."""

from litminer import __author__, __email__, __version__


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Geemi Wellawatte"
    assert __email__ == "gwellawatte@gmail.com"
    assert __version__ == "0.0.0"
