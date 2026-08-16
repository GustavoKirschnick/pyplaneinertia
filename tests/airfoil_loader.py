"""
Function used by the tests to import the airfoil .dat coordinates as an object
"""

from pathlib import Path

import numpy as np

from pyplaneinertia.exceptions import InvalidAirfoil
from pyplaneinertia.models import AirfoilCoordinates

MIN_POINTS = 4


def load_airfoil_dat(path: Path) -> AirfoilCoordinates:
    """
    Reads a .dat file of an airfoil and return it as an AirfoilCoordinates object.

    The first line correspondant to the airfoil name is skipped. The remaining lines
    with the x and z coordinates pairs are read. Blank lines are skipped.

    Args:
        path: path to the .dat file
    Returns:
        The airfoil x and z coordinates as two 1D arrays.
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise InvalidAirfoil(f"Empty file: {path}")
    name = lines[0].strip()
    coords = []

    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()

        if not line:
            continue  # Skips empty lines

        parts = line.split()
        try:
            x, z = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            raise InvalidAirfoil(f"Ill defined coordinate at line {i}: {line}")
        coords.append((x, z))

    if len(coords) < MIN_POINTS:
        raise InvalidAirfoil(
            f"Only {len(coords)} coordinates parsed. File may be corrupted"
        )
    arr = np.array(coords, dtype=float)

    return AirfoilCoordinates(name=name, x=arr[:, 0], z=arr[:, 1])
