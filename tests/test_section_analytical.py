"""
Analytical tests for Section.

These tests validate the math behind the code, with no dependency
on XFLR5, against closed-form reference values.
"""

from typing import NamedTuple

import pytest

from pyplaneinertia.section import Section


class AnalyticalCase(NamedTuple):
    chord_root: float
    chord_tip: float
    span: float
    mass: float


# Three taper ratio configurations
ANALYTICAL_CASES = [
    AnalyticalCase(chord_root=0.14, chord_tip=0.14, span=0.3, mass=0.3),
    AnalyticalCase(chord_root=0.14, chord_tip=0.12, span=0.3, mass=0.3),
    AnalyticalCase(chord_root=0.14, chord_tip=0.07, span=0.3, mass=0.3),
]


def analytical_span_com(chord_root: float, chord_tip: float, span: float) -> float:
    """
    Spanwise center of mass of a tapered semi-wing.

    Assumes uniform densinty in the wing volume. Applying the following
    equation:
    $y_{c} = \frac{\int[y c(y)^{2}]}{\int[c(y)^{2}]}$.
    """

    a = chord_tip / chord_root - 1
    return span * (0.5 + 2 * a / 3 + a**2 / 4) / (1 + a + a**2 / 3)


@pytest.mark.parametrize("chord_root, chord_tip, span, mass", ANALYTICAL_CASES)
def test_mass_is_conserved(naca0012, make_section, chord_root, chord_tip, span, mass):
    section: Section = make_section(naca0012, chord_root, chord_tip, span, mass)
    panels = section._span_panels(100)

    assert sum(panel.mass for panel in panels) == pytest.approx(section.geometry.mass)


@pytest.mark.parametrize("chord_root, chord_tip, span, mass", ANALYTICAL_CASES)
def test_span_com_matches_closed_form(
    naca0012, make_section, chord_root, chord_tip, span, mass
):
    section: Section = make_section(naca0012, chord_root, chord_tip, span, mass)
    properties = section.properties(100)
    expected = analytical_span_com(chord_root, chord_tip, span)

    assert properties.centroid_local.y == pytest.approx(expected, rel=1e-3)


def test_rectangular_Ixx_matches_flat_plate(naca0012, make_section):
    section: Section = make_section(naca0012, 0.14, 0.14, 0.3, 0.3)
    properties = section.properties(100)
    expected = section.geometry.mass * section.geometry.span**2 / 12

    assert properties.inertia_local.Ixx == pytest.approx(expected)


def test_rectangular_product_of_inertia_is_zero(naca0012, make_section):
    section: Section = make_section(naca0012, 0.14, 0.14, 0.3, 0.3)
    properties = section.properties(100)

    assert properties.inertia_local.Ixz == pytest.approx(0, abs=1e-12)


def test_refinement_converges_to_closed_form_com(naca0012, make_section):
    section: Section = make_section(naca0012, 0.14, 0.07, 0.3, 0.3)
    expected = analytical_span_com(0.14, 0.07, section.geometry.span)
    errors = [
        abs(section.properties(n).centroid_local.y - expected) for n in (10, 100, 1000)
    ]

    assert errors[0] > errors[1] > errors[2]
