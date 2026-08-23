"""
Defines the dataclasses used by the package to model the domain.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .exceptions import InvalidAirfoil, InvalidGeometry


@dataclass(frozen=True)
class AirfoilCoordinates:
    """
    Coordinates of an unit-chord 2D airfoil contour.

    The countor is stored into two 1D separate arrays (x and z).

    Atributes:
        name: airfoil name (e.g. "S1223").
        x: chord-wise coordinates, in fraction of chord [0, 1].
        z: thickness-wise coordinates, in fraction of chord.
    Methods:
        airfoil_centroid: airfoil centroid, in chord fraction.
        chordwise_gyration2: squared radius of gyration, chordwise, around the centroid.
    """

    name: str
    x: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]  # Given the y-axis is on the span-wide direction

    def __post_init__(self) -> None:
        if type(self.x) is not np.ndarray or type(self.z) is not np.ndarray:
            raise InvalidAirfoil("X and Z must be a numpy array")

        if self.x.ndim != 1 or self.z.ndim != 1:
            raise InvalidAirfoil("X and Z must be 1-D")

        if self.x.shape != self.z.shape:
            raise InvalidAirfoil("X and Z must have the same shape")

        if self.x.size <= 3:
            raise InvalidAirfoil("An airfoil must have more than 3 points")

        self.x.flags.writeable = (
            False  # Given that frozen does not cover the array element-wise
        )
        self.z.flags.writeable = False

    def _polygon_moments(self) -> Tuple[float, float, float, float]:
        """
        Computes the area, centroid, and second moment of area from the airfoil polygon.

        Uses Gauss' shoelace formula over the closed contour. The polar moment of area
        around its centroid is defined as:
        $\int[\left(x - x_{c}\right)^{2} + \left(z - z_{c}\right)^{2}] dA$.
        The area sign is kept so polar/signed_area is orientation-independent.
        """

        x, z = self.x, self.z
        x2, z2 = np.roll(x, -1), np.roll(z, -1)
        cross = (
            x * z2 - x2 * z
        )  # Consecutive vertices cross product (double the area size)
        signed_area = 0.5 * cross.sum()
        cx = ((x + x2) * cross).sum() / (
            6 * signed_area
        )  # \frac{\sum \left(A_{i} cx_{i}\right)}{\sum A_{i}}
        cz = ((z + z2) * cross).sum() / (6 * signed_area)
        second_moment_x = (1 / 12) * (
            (x**2 + x * x2 + x2**2) * cross
        ).sum()  # Area moment
        second_moment_z = (1 / 12) * (
            (z**2 + z * z2 + z2**2) * cross
        ).sum()  # Area moment
        polar = (second_moment_x - signed_area * cx**2) + (
            second_moment_z - signed_area * cz**2
        )  # Translating the moments from the origin to the centroid

        return signed_area, cx, cz, polar

    @property
    def airfoil_centroid(self) -> Tuple[float, float]:
        """
        Airfoil area centroid (cx, cz), in chord fraction.
        """

        _, cx, cz, _ = self._polygon_moments()

        return cx, cz

    @property
    def chordwise_gyration2(self) -> float:
        """
        Squared radius of gyration (polar moment / signed_area) of the airfoil about
        its centroid.

        Used to model XFLR5's chordwise mass distribution (proportional to thickness).
        Multiply by chord**2 for the dimensional value.
        """
        signed_area, _, _, polar = self._polygon_moments()
        return polar / signed_area


@dataclass(frozen=True)
class Centroid:
    """A point (x, y, z) in the local frame."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class InertiaTensor:
    """
    Inertia tensor at a point.

    Off-diagonal products default to zero.
    """

    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float = 0.0
    Ixy: float = 0.0
    Iyz: float = 0.0


@dataclass(frozen=True)
class PositionVector:
    """
    Position and incidence value of a body in the global reference frame.

    Atributes:
        x, y, z: position of the body origin (0, 0, 0).
        incidence: incidence angle, in radians.
    """

    x: float
    y: float
    z: float
    incidence: float = 0.0


@dataclass(frozen=True)
class RigidBody:
    """A rigid body, with inertia, centroid, position, and mass values."""

    inertia: InertiaTensor
    centroid: Centroid
    position: PositionVector
    mass: float


@dataclass(frozen=True)
class SpanPanel:
    """
    Single spanwise strip of a section.

    Atributes:
        chord: local chord lenght of the panel.
        area: panel area (chord x panel width).
        mass: panel mass.
        centroid: panel centroid in the local frame.
    """

    chord: float
    area: float
    mass: float
    centroid: Centroid


@dataclass(frozen=True)
class SectionGeometry:
    """
    Geometrical and mass definition of a single lifting surface.

    Atributes:
        airfoil_coordinates: the section's airfoil coordinates.
        chord_root: chord at root (> 0).
        chord_tip: chord at tip (0, chord_root].
        span: section span (> 0).
        mass: sectiom mass (> 0).
    Methods:
        taper_ratio: taper ratio as $\frac{c_{tip}}{c_{root}}$.
    """

    airfoil_coordinates: AirfoilCoordinates
    chord_root: float
    chord_tip: float
    span: float
    mass: float

    def __post_init__(self) -> None:
        if self.chord_tip > self.chord_root:
            raise InvalidGeometry(
                "The chord_tip must be equal or less than the chord_root"
            )

        if self.chord_tip <= 0 or self.chord_root <= 0:
            raise InvalidGeometry("The chord must be greater than zero")

        if self.span <= 0:
            raise InvalidGeometry("The span must be greater than zero")

        if self.mass <= 0:
            raise InvalidGeometry("The mass must be greater than zero")

    @property
    def taper_ratio(self) -> float:
        """
        Computes the taper ratio as chord_tip/chord_root.

        Returns:
            The taper ratio (0, 1].
        """
        return self.chord_tip / self.chord_root


@dataclass(frozen=True)
class SectionProperties:
    """
    Reunites the properties of a section, in its local frame.

    Arguments:
        inertia_local: section's inertia.
        centroid_local: section's centroid.
        geometry: section's geometry descriptor.
        area_local: section's area.
    """

    inertia_local: InertiaTensor
    centroid_local: Centroid
    geometry: SectionGeometry
    area_local: float

@dataclass(frozen=True)
class MassProperties:
    """
    Reunites the mass properties of a body, whether on the local or global coordinate frame.
    """
    mass: float
    center_of_mass: Centroid
    inertia: InertiaTensor
