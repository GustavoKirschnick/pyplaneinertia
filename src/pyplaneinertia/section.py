"""
Performs section-level modeling by dividing a lifting surface (SectionGeometry)
into spanwise panels. The Section's mass, center of mass, and inertia tensor are
evaluated on the local frame.

Local frame convention:
    - x: chord-wise
    - y: span-wise
    - z: thickness-wise
"""

from typing import List

from .combine import combine
from .models import (
    Centroid,
    InertiaTensor,
    MassProperties,
    SectionGeometry,
    SectionProperties,
    SpanPanel,
)


class Section:
    """
    Discritezes a single lifting surface into spanwise panels

    Computes the physical properties of a SectionGeometry by numerical integration over
    a configurable number of panels
    """

    def __init__(self, section_geometry: SectionGeometry):

        self.geometry = section_geometry

    def properties(self, n_panel: int) -> SectionProperties:
        """
        Returns the inertia properties of a section as an object for downstream
        consumption.

        Args:
            n_panel: number of panels to divide the section span into (e.g 100).
        Returns:
            The properties of a section at its CoM.
        """

        panels: List[SpanPanel] = self._span_panels(n_panel)  # Discritizes the geometry
        parts: List[MassProperties] = [
            self._panel_mass_properties(panel, n_panel) for panel in panels
        ]  # Consolidate each panel mass properties
        combined = combine(
            parts
        )  # Consolidates the properties on the Section CoM from all the panels

        return SectionProperties(
            inertia_local=combined.inertia,
            centroid_local=combined.center_of_mass,
            geometry=self.geometry,
            area_local=sum(panel.area for panel in panels),
        )

    def _span_panels(self, n_panel: int) -> List[SpanPanel]:
        """
        Divides the SectionGeometry into n spanwise panels, following XFLR5 mass model

        Mass is distributed assuming uniform density across the wing volume: each
        panel's mass is proportional to its local volume, which scales with the chord
        squared. The masses are normalized so their sum equals to self.geometry.mass.
        Each panel is sampled at its spanwise midpoint.

        For each panel, the chord, area, mass, and centroid are computed.

        Args:
            n_panel: number of panels to divide the span into.
        Return:
            The panels, ordered from chord-root to chord-tip.
        """

        delta = self.geometry.span / n_panel
        midpoints = [(i + 0.5) * delta for i in range(n_panel)]

        airfoil_cx, airfoil_cz = self.geometry.airfoil_coordinates.airfoil_centroid

        chords = [self._chord(y) for y in midpoints]  # calls helper function
        weights = [
            c**2 for c in chords
        ]  # proportional to the local volume (uniform density)
        total_weight = sum(weights)

        panels: List[SpanPanel] = []  # initializes empty list

        for i, y in enumerate(midpoints):
            chord = chords[i]
            area = chord * delta
            mass = (
                self.geometry.mass * weights[i] / total_weight
            )  # Σ mass == self.geometry.mass
            centroid = Centroid(x=airfoil_cx * chord, y=y, z=airfoil_cz * chord)
            panels.append(
                SpanPanel(chord=chord, area=area, mass=mass, centroid=centroid)
            )

        return panels

    def _panel_mass_properties(self, panel: SpanPanel, n_panel: int) -> MassProperties:
        """
        Agregates the panel mass properties: mass, center of mass, inertia at its OWN
        frame.

        Args:
            panel: the input panel.
            n_panel: the total number of panels.
        Return:
            The mass properties of a panel.
        """
        delta = self.geometry.span / n_panel
        gyration2 = (
            self.geometry.airfoil_coordinates.chordwise_gyration2
        )  # Chordwise distribution of the airfoil area, around the centroid
        Ixx = (1 / 12) * panel.mass * delta**2
        Iyy = panel.mass * gyration2 * panel.chord**2
        Izz = Ixx + Iyy

        return MassProperties(
            mass=panel.mass,
            center_of_mass=panel.centroid,
            inertia=InertiaTensor(Ixx=Ixx, Iyy=Iyy, Izz=Izz),
        )

    def _chord(self, y: float) -> float:
        """
        Returns the local chord of a panel at a spanwise position.

        Args:
            y: spanwise position measured from the section's root.
        Returns:
            The chord lenght at 'y'.
        """
        return self.geometry.chord_root + (
            self.geometry.chord_tip - self.geometry.chord_root
        ) * (y / (self.geometry.span))
