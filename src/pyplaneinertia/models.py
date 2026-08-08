"""
Defines the dataclasses used by the package to model the domain
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .exceptions import InvalidAirfoil, InvalidGeometry


@dataclass(frozen=True)
class AirfoilCoordinates:
    name: str
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise InvalidAirfoil("X and Y must be 1-D")

        if self.x.shape != self.y.shape:
            raise InvalidAirfoil("X and Y must have the same shape")

        if self.x.size <= 3:
            raise InvalidAirfoil("An airfoil must have more than 3 points")

        self.x.flags.writeable = (
            False  # Given that frozen does not cover the array element-wise
        )
        self.y.flags.writeable = False


@dataclass(frozen=True)
class Centroid:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class InertiaTensor:
    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float
    Ixy: float | None = None
    Iyz: float | None = None


@dataclass(frozen=True)
class PositionVector:
    x: float
    y: float
    z: float
    incidence: float = 0.0


@dataclass(frozen=True)
class RigidBody:
    inertia: InertiaTensor
    centroid: Centroid
    position: PositionVector
    mass: float


@dataclass(frozen=True)
class SpanPanel:
    chord: float
    area: float
    mass: float
    centroid: Centroid


@dataclass(frozen=True)
class SectionGeometry:
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

    # Computes the taper ratio as chord_tip/chord_root
    @property
    def taper_ratio(self) -> float:
        return self.chord_tip / self.chord_root

    # Computes the empirical adjust factor (kappa)
    @property
    def kappa(self) -> float:
        return -0.711 * self.taper_ratio + 0.979


@dataclass(frozen=True)
class SectionProperties:
    inertia_local: InertiaTensor
    centroid_local: Centroid
    geometry: SectionGeometry
    area_local: float
