"""
Tests Section.properties against XFLR5 reference values.

Covers if the center of mass and mass moments of inertia of 6 test cases
are within an agreement interval between the two outputs. The semi-wing
is modeled as a vertical estabilizer on XFLR5. The outputted values are
mapped to the code's reference frame for the testing.
"""

from typing import NamedTuple

import pytest

from pyplaneinertia.models import SectionProperties


class XFLR5Case(NamedTuple):
    airfoil: str
    chord_root: float
    chord_tip: float
    span: float
    mass: float
    cg_x: float
    cg_y: float
    cg_z: float
    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float


# Test cases: 3 taper ratios, 2 airfoils (symetrical and assymetrical). Same mass and
# span.
XFLR5_CASES = [
    XFLR5Case(
        airfoil="selig1223",
        chord_root=0.14,
        chord_tip=0.14,
        span=0.3,
        mass=0.3,
        cg_x=0.049,
        cg_y=0.150,
        cg_z=0.010,
        Ixx=0.00225,
        Iyy=0.000278,
        Izz=0.00252,
        Ixz=1.56e-20,
    ),
    XFLR5Case(
        airfoil="selig1223",
        chord_root=0.14,
        chord_tip=0.12,
        span=0.3,
        mass=0.3,
        cg_x=0.045,
        cg_y=0.142,
        cg_z=0.009,
        Ixx=0.00224,
        Iyy=0.000243,
        Izz=0.00248,
        Ixz=-5.18e-05,
    ),
    XFLR5Case(
        airfoil="selig1223",
        chord_root=0.14,
        chord_tip=0.07,
        span=0.3,
        mass=0.3,
        cg_x=0.039,
        cg_y=0.118,
        cg_z=0.008,
        Ixx=0.00201,
        Iyy=0.000198,
        Izz=0.0022,
        Ixz=-0.000163,
    ),
    XFLR5Case(
        airfoil="naca0012",
        chord_root=0.14,
        chord_tip=0.14,
        span=0.3,
        mass=0.3,
        cg_x=0.059,
        cg_y=0.150,
        cg_z=-2.42e-16,
        Ixx=0.00225,
        Iyy=0.000323,
        Izz=0.00257,
        Ixz=6.84e-21,
    ),
    XFLR5Case(
        airfoil="naca0012",
        chord_root=0.14,
        chord_tip=0.12,
        span=0.3,
        mass=0.3,
        cg_x=0.055,
        cg_y=0.142,
        cg_z=-2.3e-16,
        Ixx=0.00223,
        Iyy=0.000283,
        Izz=0.00252,
        Ixz=-6.28e-05,
    ),
    XFLR5Case(
        airfoil="naca0012",
        chord_root=0.14,
        chord_tip=0.07,
        span=0.3,
        mass=0.3,
        cg_x=0.0474,
        cg_y=0.118,
        cg_z=-1.9e-16,
        Ixx=0.002,
        Iyy=0.000234,
        Izz=0.00224,
        Ixz=-0.000197,
    ),
]


def _case_id(case):
    return f"{case.airfoil}_ct{case.chord_root}"


@pytest.mark.parametrize("case", XFLR5_CASES, ids=_case_id)
def test_xflr5_match_properties(airfoils, make_section, case):
    section = make_section(
        airfoils[case.airfoil], case.chord_root, case.chord_tip, case.span, case.mass
    )
    properties: SectionProperties = section.properties(
        100
    )  # 100 panels returned agreeable results
    com, inertia = properties.centroid_local, properties.inertia_local

    # Addopting a standard 2% deviation for comparison
    # Center of mass assertions
    assert com.x == pytest.approx(case.cg_x, rel=2e-2)
    assert com.y == pytest.approx(case.cg_y, rel=2e-2)
    assert com.z == pytest.approx(
        case.cg_z, abs=5e-3
    )  # Absolute tolerance given z is almost zero

    # Mass moments of inertia
    assert inertia.Ixx == pytest.approx(case.Ixx, rel=2e-2)
    assert inertia.Iyy == pytest.approx(case.Iyy, rel=2e-2)
    assert inertia.Izz == pytest.approx(case.Izz, rel=2e-2)
