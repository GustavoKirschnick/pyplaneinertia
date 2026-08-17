"""
Tests dataclasses properties and exception handling.

For the AirfoilCoordinates, the properties are tested against geometric shapes
(rectangle and a dislocated trapezoid) and a symetrical airfoil. SectionGeometry
exception handling is tested, as well as the taper_ratio property.

"""

import numpy as np
import pytest

from pyplaneinertia.exceptions import InvalidGeometry
from pyplaneinertia.models import AirfoilCoordinates, SectionGeometry


def make_rectangle() -> AirfoilCoordinates:
    """Rectangle width of 4.0 vs height of 1.0"""
    x = np.array([0.0, 0.0, 4.0, 4.0])
    z = np.array([0.0, 1.0, 1.0, 0.0])
    return AirfoilCoordinates("rectangle", x, z)


def make_trapezoid() -> AirfoilCoordinates:
    x = np.array([0.0, 1.0, 3.0, 4.0])
    z = np.array([1.0, 3.0, 3.0, 1.0])
    return AirfoilCoordinates("trapezoid", x, z)


def test_airfoil_rectangle_centroid():
    cx, cz = make_rectangle().airfoil_centroid
    # make_rectangle centroid coords (x, z) = (0.5, 2.0)
    assert cz == pytest.approx(0.5)
    assert cx == pytest.approx(2.0)


def test_airfoil_rectangle_gyrational2():
    # $\frac{J}{A} = \frac{\left(w^{2} + h^{2}\right)}{12}$
    expected = (1**2 + 4**2) / 12
    assert make_rectangle().chordwise_gyration2 == pytest.approx(expected)


def test_airfoil_trapezoid_centroid():
    cx, cz = make_trapezoid().airfoil_centroid
    # make_trapezoid centroid coords (x, z) = (1.0 + 8/9, 2.0)
    # zc = \frac{\left(b + 2a \right)h}{3\left(a + b\right)}
    assert cz == pytest.approx(1.0 + 8 / 9)
    assert cx == pytest.approx(2.0)


def test_airfoil_naca0012_zero_z_centroid(naca0012):
    _, cz = naca0012.airfoil_centroid
    assert cz == pytest.approx(0.0)  # Given its symetry in the z-axis


def test_section_geometry_rejects_inverted_taper_ratio(naca0012):
    with pytest.raises(InvalidGeometry):
        SectionGeometry(naca0012, chord_root=0.14, chord_tip=0.18, span=0.3, mass=0.3)


def test_section_geometry_accepts_retangular_taper_ratio(naca0012):
    section_geometry = SectionGeometry(
        naca0012, chord_root=0.14, chord_tip=0.14, span=0.3, mass=0.3
    )
    assert section_geometry.taper_ratio == pytest.approx(1)


def test_section_geometry_accepts_valid_taper_ratio(naca0012):
    section_geometry = SectionGeometry(
        naca0012, chord_root=0.14, chord_tip=0.07, span=0.3, mass=0.3
    )
    assert section_geometry.taper_ratio == pytest.approx(0.5)


def test_section_geometry_rejects_negative_chord_root(naca0012):
    with pytest.raises(InvalidGeometry):
        SectionGeometry(naca0012, chord_root=-0.14, chord_tip=0.07, span=0.3, mass=0.3)


def test_section_geometry_rejects_negative_chord_tip(naca0012):
    with pytest.raises(InvalidGeometry):
        SectionGeometry(naca0012, chord_root=0.14, chord_tip=-0.07, span=0.3, mass=0.3)


def test_section_geometry_rejects_negative_span(naca0012):
    with pytest.raises(InvalidGeometry):
        SectionGeometry(naca0012, chord_root=0.14, chord_tip=0.14, span=-0.3, mass=0.3)


def test_section_geometry_rejects_zero_span(naca0012):
    with pytest.raises(InvalidGeometry):
        SectionGeometry(naca0012, chord_root=0.14, chord_tip=0.14, span=0, mass=0.3)


def test_section_geometry_rejects_negative_mass(naca0012):
    with pytest.raises(InvalidGeometry):
        SectionGeometry(naca0012, chord_root=0.14, chord_tip=0.14, span=0.3, mass=-0.3)


def test_section_geometry_rejects_zero_mass(naca0012):
    with pytest.raises(InvalidGeometry):
        SectionGeometry(naca0012, chord_root=0.14, chord_tip=0.14, span=0.3, mass=0)


def test_section_geometry_valid_taper_ratio(naca0012):
    section_geometry = SectionGeometry(
        naca0012, chord_root=0.14, chord_tip=0.14, span=0.3, mass=0.3
    )
    assert section_geometry.taper_ratio == pytest.approx(1)
