"""
Computes a lifting-surface mass properties. A surface is built from a list of
Sectionproperties, computing its mass, center of mass, and inertia tensor about its
own center of mass, in the global frame. The Surface can be either:

- HorizontalSurface: span in the global XY plane (e.g wing, horizontal stabilizer).
- VerticalSurface: span along the Z-axis (e.g vertical stabilizer).
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List

import numpy as np

from .combine import combine
from .models import Centroid, MassProperties, PositionVector, SectionProperties
from .transforms import place


def _incidence_rotation(incidence_deg: float) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix for an incidence angle about the y-axis.

    Args:
        incidence_deg: incidence (tilt) angle in degrees.
    Returns:
        The rotation matrix, considering the incidence angle.

    """
    c, s = np.cos(np.radians(incidence_deg)), np.sin(np.radians(incidence_deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Surface(ABC):
    """
    Aggregates a list of Sections that form a Surface into a single set of mass
    properties (via 'combine'), and then maps them from the local coordinate frame
    to the global frame. Subclasses apply the local-to-global rotation.
    """

    def __init__(
        self,
        sections: List[SectionProperties],
        body_position: PositionVector,
        symmetric: bool,
    ):
        self.sections = sections
        self.body_position = body_position
        self.symmetric = symmetric

    def properties(self) -> MassProperties:
        """
        Computes the Surface's mass properties about its center of mass in the global
        frame.

        Aggregates (positioned, mirror) the sections, then rotates and translates the
        result into the global coordinate frame.

        Returns:
            The surface mass, center of mass, and inertia on the global frame.
        """
        local = combine(self._section_parts())
        return place(local, self._rotation(), self.body_position)

    def _section_parts(self) -> List[MassProperties]:
        """
        Returns the sections as mass properties, positioned along the span.

        For a symmetric surface, each positioned section is paired with its mirror
        image across the span centerline (y=0).

        Returns:
            The parts to be aggregated by 'combine'.
        """
        parts = [self._to_mass_properties(s) for s in self._positioned_sections()]
        if self.symmetric:
            parts = parts + [self._mirror_about_y(p) for p in parts]
        return parts

    def _positioned_sections(self):
        """
        Offsets each Section's y-axis centroid by the cummulative inboard span.

        Given that each Section is defined at its local frame, the y-axis has to
        account for the already existing span of each section.

        Returns:
            Sections with their y-axis centroid shifted into the surface frame.
        """
        positioned: List[MassProperties] = []
        offset = 0.0

        for section in self.sections:
            shifted = replace(
                section.centroid_local, y=section.centroid_local.y + offset
            )
            positioned.append(replace(section, centroid_local=shifted))
            offset += section.geometry.span

        return positioned

    @staticmethod
    def _to_mass_properties(section: SectionProperties) -> MassProperties:
        """
        Converts the properties of a section to mass properties for aggregation.

        Args:
            section: the section with its local center of mass and inertia.
        Returns:
            The section's mass, local center of mass, and local inertia as
            MassProperties.
        """

        return MassProperties(
            mass=section.geometry.mass,
            center_of_mass=section.centroid_local,
            inertia=section.inertia_local,
        )

    @staticmethod
    def _mirror_about_y(part: MassProperties) -> MassProperties:
        """
        Returns the mirrored mass properties of a part across the span centerline (y=0).
        Used when the surface is symmetrical.

        The centroid's y coordinate is negated, as well as the products of inertia with
        y components.

        Args:
            part: mass properties of one half of a symetric surface.
        Returns:
            Mirrored mass properties.
        """
        i = part.inertia
        return MassProperties(
            mass=part.mass,
            center_of_mass=Centroid(
                part.center_of_mass.x, -part.center_of_mass.y, part.center_of_mass.z
            ),
            inertia=replace(i, Ixy=-i.Ixy, Iyz=-i.Iyz),
        )

    @abstractmethod
    def _rotation(self) -> np.ndarray:
        """
        Returns a 3x3 rotation matrix that maps the local coordinate frame to the global
        one.

        This matrix is applied both to the inertia tensor and to the centroid.
        """
        pass


class HorizontalSurface(Surface):
    """
    Surface that runs along the XY plane.

    Its local frame already matches the global one, so only the incidence is applied on
    the rotation.
    """

    def _rotation(self) -> np.ndarray:
        return _incidence_rotation(self.body_position.incidence)


class VerticalSurface(Surface):
    """
    Surface that runs along the XZ plane.

    The span (local y-axis) maps to the global Z-axis via a rotation, that also
    considers the incidence.
    """

    def _rotation(self):
        swap_y_z = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        return _incidence_rotation(self.body_position.incidence) @ swap_y_z
