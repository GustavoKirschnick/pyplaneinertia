"""
Coordinate-frame transforms for mass properties.
"""

import numpy as np

from .models import Centroid, InertiaTensor, MassProperties, PositionVector


def tensor_to_matrix(t: InertiaTensor) -> np.ndarray:
    """
    Returns the symmetric 3x3 inertia matrix of an InertiaTensor.
    """
    return np.array(
        [[t.Ixx, t.Ixy, t.Ixz], [t.Ixy, t.Iyy, t.Iyz], [t.Ixz, t.Iyz, t.Izz]]
    )


def matrix_to_tensor(m: np.ndarray) -> InertiaTensor:
    """
    Returns an InertiaTensor from a 3x3 inertia matrix. Opposite of "tensor_to_matrix".
    """
    return InertiaTensor(
        Ixx=m[0, 0], Iyy=m[1, 1], Izz=m[2, 2], Ixy=m[0, 1], Ixz=m[0, 2], Iyz=m[1, 2]
    )


def place(
    props: MassProperties, R: np.ndarray, position: PositionVector
) -> MassProperties:
    """
    Express local mass properties in the global coordinate frame.

    The rotation R acts both on the cenetr of mass and the inertia tensor, since there
    is a change of axes. The "position" only translates the center of mass, not the
    inertia about its center of mass.

    Args:
        props: mass properties in the body's local frame.
        R: 3x3 rotation matrix mappijng the local axes to the global one.
        position: where the body's origin sits in the global frame.
    Returns:
        The mass properties in the global frame.
    """

    com = props.center_of_mass
    com_rot = R @ np.array([com.x, com.y, com.z])
    com_global = Centroid(
        x=com_rot[0] + position.x, y=com_rot[1] + position.y, z=com_rot[2] + position.z
    )
    inertia_global = matrix_to_tensor(R @ tensor_to_matrix(props.inertia) @ R.T)
    return MassProperties(
        mass=props.mass,
        center_of_mass=com_global,
        inertia=inertia_global,
    )
