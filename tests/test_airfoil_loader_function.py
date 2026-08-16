"""
Tests for the airfoil .dat loader.

Covers the accurate reading of a file. Rejects files that are
empty, too short, or have unexpected characters.
"""

import pytest

from pyplaneinertia.exceptions import InvalidAirfoil

from .airfoil_loader import load_airfoil_dat

NACA_DAT = """NACA0012
1.0 0.0
0.5 0.06
0.0 0.0
0.5 -0.06
1.0 0.0
"""


def test_reads_valid_dat_file(tmp_path):
    path = tmp_path / "airfoil.dat"
    path.write_text(NACA_DAT)

    airfoil = load_airfoil_dat(path)

    assert airfoil.name == "NACA0012"

    assert airfoil.x.min() == pytest.approx(0.0)
    assert airfoil.x.max() == pytest.approx(1.0)

    assert airfoil.x.shape == airfoil.z.shape

    _, cz = airfoil.airfoil_centroid
    assert cz == pytest.approx(0.0, abs=1e-4)  # Given that NACA is a symetrical airfoil


def test_rejects_defective_line(tmp_path):
    path = tmp_path / "bad_file.dat"
    path.write_text("NACA0012\n 1.0 0.0\n Noise error\n 0.0 0.0\n 0.5 -0.06\n")

    with pytest.raises(InvalidAirfoil):
        load_airfoil_dat(path)


def test_rejects_empty_file(tmp_path):
    path = tmp_path / "empy_file.dat"
    path.write_text("")

    with pytest.raises(InvalidAirfoil):
        load_airfoil_dat(path)


def test_rejects_too_few_points(tmp_path):
    path = tmp_path / "short.dat"
    path.write_text("NACA0012\n1.0 0.0\n 0.0 0.0\n")

    with pytest.raises(InvalidAirfoil):
        load_airfoil_dat(path)
