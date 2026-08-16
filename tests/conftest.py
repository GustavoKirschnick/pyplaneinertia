"""
Shared fixtures and helpers for the Section and DistributedSurface tests.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from pyplaneinertia.models import AirfoilCoordinates, SectionGeometry
from pyplaneinertia.section import Section

from .airfoil_loader import load_airfoil_dat

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def airfoils(
    naca0012: AirfoilCoordinates, selig1223: AirfoilCoordinates
) -> dict[str, AirfoilCoordinates]:
    """Name lookup, so the airfoils can be referenced as strings."""
    return {"naca0012": naca0012, "selig1223": selig1223}


@pytest.fixture(scope="session")
def naca0012() -> AirfoilCoordinates:
    """Loads up NACA0012 coordinates from file."""
    return load_airfoil_dat(FIXTURES_DIR / "naca0012.dat")


@pytest.fixture(scope="session")
def selig1223() -> AirfoilCoordinates:
    """Loads up Selig1223 coordinates from file."""
    return load_airfoil_dat(FIXTURES_DIR / "s1223.dat")


@pytest.fixture
def make_section() -> Callable[..., Section]:
    """
    Factory fixture, builds a Section based on test geometrical parameters.
    """

    def _make_section(airfoil, choord_root, chord_tip, span, mass) -> Section:
        geometry = SectionGeometry(
            airfoil_coordinates=airfoil,
            chord_root=choord_root,
            chord_tip=chord_tip,
            span=span,
            mass=mass,
        )

        return Section(geometry)

    return _make_section
