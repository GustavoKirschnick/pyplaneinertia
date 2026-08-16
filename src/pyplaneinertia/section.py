"""
Performs section-level modeling by dividing a lifting surface (SectionGeometry)
into spanwise panels, computing its mass, center of mass, and inertia tensor in
the local frame.

Local frame convention:
    - x: chord-wise
    - y: span-wise
    - z: thickness-wise
"""

from typing import List

from .models import Centroid, InertiaTensor, SectionGeometry, SectionProperties, SpanPanel


class Section:
    """
    Discritezes a single lifting surface into spanwise panels

    Computes the physical properties of a SectionGeometry by numerical integration over
    a configurable number of panels
    """

    def __init__(self, section_geometry: SectionGeometry):

        self.geometry = section_geometry

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

    def center_of_mass_local(self, panels: List[SpanPanel]) -> Centroid:
        """
        Computes the section's center of mass as a mass-weighted mean between the
        panels.

        Each panel contributes with its centroid weighted by its mass, resulting in
        the section's center of mass.

        Args:
            panels: the section's panels, as returned by '_span_panels'.
        Return:
            The center of mass in the local frame.
        """

        x_ce = (
            sum(panel.mass * panel.centroid.x for panel in panels) / self.geometry.mass
        )
        y_ce = (
            sum(panel.mass * panel.centroid.y for panel in panels) / self.geometry.mass
        )
        z_ce = (
            sum(panel.mass * panel.centroid.z for panel in panels) / self.geometry.mass
        )

        return Centroid(x=x_ce, y=y_ce, z=z_ce)

    def inertia_local(
        self, panels: List[SpanPanel], center_of_mass: Centroid
    ) -> InertiaTensor:
        """
        Computes the section's inertia at its center of mass.

        Each span panel contributes with its own inertia (modeled as a thin plate)
        translated to the section's center of mass.

        $I_{cm} = I_{self} + m * d^{2}$

        Args:
            panels: the section's panels, as returned by '_span_panels'.
            center_of_mass: the section's center of mass, as returned by
            'center_of_mass_local'.
        Return:
            The inertia at the center of mass in the local frame.
        """
        
        Ixx = Iyy = Ixz = 0
        delta = self.geometry.span / len(panels)
        gyration2 = self.geometry.airfoil_coordinates.chordwise_gyration2 # Chordwise distribution of the airfoil area, around the centroid

        for panel in panels:
            dx = panel.centroid.x - center_of_mass.x
            dy = panel.centroid.y - center_of_mass.y
            dz = panel.centroid.z - center_of_mass.z
            Ixx += (1 / 12) * panel.mass * delta**2 + panel.mass * (dz**2 + dy**2)
            Iyy += (panel.mass * gyration2 * panel.chord**2) + panel.mass * (dz**2 + dx**2)
            Ixz += -panel.mass * dx * dy  # minor fix needed

        return InertiaTensor(
            Ixx=Ixx,
            Iyy=Iyy,
            Izz=Ixx + Iyy,
            Ixz=Ixz,
        )

    def _chord(self, y: float) -> float:
        """
        Returns the local chord of a panel at a spanwise position.

        Args:
            y: spanwise position measured from the section's root
        Returns:
            The chord lenght at 'y'
        """
        return self.geometry.chord_root + (
            self.geometry.chord_tip - self.geometry.chord_root
        ) * (y / (self.geometry.span))

    def properties(self, n_panel: int) ->SectionProperties:
        """
        Returns the inertia properties of a section as an object for downstream consumption

        Args:
            n_panel: number of panels to divide the section span into (e.g 100)
        Returns:
            The properties of a section at its CoM
        """

        panels: List[SpanPanel] = self._span_panels(n_panel) # Discritizes the geometry
        centroid = self.center_of_mass_local(panels) # Compute the Section geometry
        inertia = self.inertia_local(panels, centroid) # Compute the Section inertia
        area = sum(panel.area for panel in panels) # Sum up the total panel area

        return SectionProperties(
            inertia_local=inertia,
            centroid_local=centroid,
            geometry=self.geometry,
            area_local=area,
            )