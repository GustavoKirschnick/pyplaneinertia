"""
Mass property aggregation, used at every level (Section, Surface, Aircraft) of the model.

Folds panels into a section, sections into a surface, and surfaces/point masses into the
aircraft.
"""

from typing import List
from .models import Centroid, InertiaTensor, MassProperties

def combine(parts: List[MassProperties]) ->MassProperties:
    """
    Aggregates several mass properties into one.

    Computes the mass-weighted center of mass and inertia about it, transposing each
    part's inertia with the parallel-axis theorem. Every part's inertia MUST be about
    its own center of mass, and all parts must be expressed under one coordinate frame.
    
    Args:
        parts: the components to aggregate (panels, sections, surfaces, ...)
    Retuns:
        The consolidated mass, cebter of mass, and inertia tensor of the whole
    """
    total_mass = sum(part.mass for part in parts)

    cog = Centroid(
        x= sum(part.mass * part.center_of_mass.x for part in parts)/total_mass,
        y= sum(part.mass * part.center_of_mass.y for part in parts)/total_mass,
        z= sum(part.mass * part.center_of_mass.z for part in parts)/total_mass,
    )

    Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0

    for part in parts: # Translates each part's inertia to the common center of mass
        dx = part.center_of_mass.x - cog.x
        dy = part.center_of_mass.y - cog.y
        dz = part.center_of_mass.z - cog.z

        Ixx += part.inertia.Ixx + part.mass * (dz**2 + dy**2)
        Iyy += part.inertia.Iyy + part.mass * (dz**2 + dx**2)
        Izz += part.inertia.Izz + part.mass * (dx**2 + dy**2)
        Ixy += part.inertia.Ixy - part.mass * (dx * dy)
        Ixz += part.inertia.Ixz - part.mass * (dx * dz)
        Iyz += part.inertia.Iyz - part.mass * (dy * dz)

    return MassProperties(
        mass= total_mass,
        center_of_mass= cog,
        inertia= InertiaTensor(Ixx=Ixx, Iyy=Iyy, Izz= Izz, Ixz=Ixz, Ixy=Ixy, Iyz=Iyz),
    )

