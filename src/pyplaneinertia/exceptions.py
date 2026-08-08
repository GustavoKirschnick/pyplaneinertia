"""
Defines custom exceptions for the package
"""

class PyPlaneInertiaError(Exception):
    """Base for all exceptions from the PyPlaneInertia package"""

class InvalidAirfoil(PyPlaneInertiaError, ValueError):
    """Ill defined airfoil coordinates"""


class InvalidGeometry(PyPlaneInertiaError, ValueError):
    """Ill defined chord, span, or mass"""
