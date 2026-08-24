# pyplaneinertia

Inertia properties of aircraft lifting surfaces for stability and performance analysis.

[![CI](https://github.com/GustavoKirschnick/pyplaneinertia/actions/workflows/ci.yml/badge.svg)](https://github.com/GustavoKirschnick/pyplaneinertia/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.13)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

**Under active development**: The `Section` layer has been completed and validated against [XFLR5](https://www.flow5.tech/xflr5/xflr5.html). Higher layers are under development. Check [Status](#status).

`pyplaneinertia` computes the **mass, center of mass, and inertia tensor** of an aircraft's lifting surfaces (wings, horizontal and vertical stabilizer) and point masses, in the global body frame, which are needed inputs for external dynamical stability analysis (e.g AVL) and take-off performance analysis.

It originated as a refactor of the inertia module of an aircraft multidisciplinary design optimization (MDO) tool built for the **Céu Azul Aeronaves** Aerodesign team, rebuilt from scratch with a clean, tested, validated architecture.

## Status

| Layer        | Implemented | Tested | Validated (XFLR5) |
|--------------|:-----------:|:------:|:-----------------:|
| `models`     | ✅          | ✅     | —                 |
| `Section`    | ✅          | ✅     | ✅  (~2% error)          |
|`Surface`    | ✅          | ⏳     | ⏳  (under progress) |
| `PointMass`  | ❌          | ❌     | —                 |
| `Aircraft`   | ❌          | ❌     | —                 |

✅ Done • ⏳ Under current development • ❌​ Not (yet) implemented • — Not applicable

## Installation

Not yet published to PyPI. Install from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/GustavoKirschnick/pyplaneinertia.git
cd pyplaneinertia
uv sync --dev
```

## Quick start

Build the surface from the bottom up: **airfoil -> section -> surface**. The package takes ready-made data objects and returns mass properties, reading the airfoil is the caller's responsibility and is intentionally kept outside the package.

```python
import numpy as np

from pyplaneinertia import (
    AirfoilCoordinates,
    Section,
    SectionGeometry,
    HorizontalSurface,
    PositionVector,
)

# 1. Airfoil coordinates (unit chord!), from your own .dat reader
airfoil = AirfoilCoordinates(name="NACA0012", x=x_coords, z=z_coords)

# 2. Set up your geometry. chord root/tip, span, and mass

first_section = SectionGeometry(
    airfoil_coordinates=airfoil,
    chord_root=0.20,
    chord_tip=0.20,
    span=0.6,
    mass=0.4,
)

second_section = SectionGeometry(
    airfoil_coordinates=airfoil,
    chord_root=0.20,
    chord_tip=0.10,
    span=0.3,
    mass=0.12,
)

# 3. Section properties (mass, center of mass, local inertia)

sections = [
    Section(first_section).properties(n_panel=100),
    Section(second_section).properties(n_panel=100),
]

# 4. A symmetric wing, mounted at the origin with no incidence.

wing = HorizontalSurface(
    section=sections,
    body_position=PositionVector(x=0.0, y=0.0, z=0.0, incidence=0.0),
    symmetric=True,
)

props = wing.properties()
print(props.mass)  # total mass
print(props.center_of_mass)  # center of mass in the global frame
print(props.inertia)  # inertia tensor about the wing's center of mass
```

## Architecture

Each layer consumes the output of the one below it:

```
AirfoilCoordinates --> Section --> Surface --> Aircraft
(airfoil geometry)(panels discretization)(sections)(components)
```

- **Section** discretizes a tapered section into spanwise panels, distributes mass by the XFLR5-style volume model (mass ∝ chord²), and integrates the local center of mass and inertia;
- **Surface** positions sections along the span, mirrors them for a symmetric surface, aggregates them, and maps the result to the global frame. `HorizontalSurface` and `VerticalSurface` differ only in the mapping; 
- **`combine`** is the shared aggregation core: Mass-weighted center of mass plus parallel-axis theorem. Reused at every level, so the translation is defined once.

## Validation

Results are checked and validated in two independent ways.

**Closed-form** (tight tolerance): The spanwise center of mass of a rectangular tapered wing and the inertia of a rectangular flat plate both have exact solutions the code must reproduce. 

**XFLR5** (external software used as reference): The surface is modeled in XFLR5 with the exact same geometry. The inertia tensor and center of mass computed by the code were validated against the XFLR5 results.

Representative results:

```python
section = SectionGeometry(
    airfoil_coordinates=naca0012,
    chord_root=0.14,
    chord_tip=0.14,
    span=0.3,
    mass=0.3,
)
```

| Quantity | XFLR5 | Implemented | Delta (%) |
|--------------|:-----------:|:------:|:-----------------:|
|I_{xx}|2.25e-3|2.25e-3| 0|
|I_{yy}|3.23e-4|3.29e-4|~1.93|
|I_{zz}|2.57e-3|2.58e-3|~0.36|
|CoM (x)|0.059| 0.0588|~0.33|

## Design decisions

- **Pure domain, no I/O:** The package receives dataclasses and returns properties. File reading (airfoil `.dat`) belongs to adapters, outside the package, so the numeric core stays trivially testable.
- **Immutable value objects:** Every domain type is a frozen dataclass. "Changes" produce new objects, never mutation.
- **One aggregation, reused:** The parallel-axis translation lives only in `combine`, shared from sections to the aircraft, standardizing the translation logic.
- **Validated:** Numerics are validated against closed-form solutions and an external tool.

## Development

```bash
uv run pytest # test suite (with coverage)
uv run ruff check # lint
uv run ruff format # format
```

Continuous integration runs lint and tests every push and pull request.

## Roadmap

- [ ] Test suite for `Surface`
- [ ] Implement `PointMass` and `Aircraft`
- [ ] Test suite for `PointMass` and `Aircraft`
- [ ] End-to-end validation of a full aircraft against XFLR5

## Limitation

Intended for **thin lifting surfaces under near-planar assumptions**. Mass is modeled as uniform in the surface volume, so inertia figures are order-of-magnitude estimates, which is the same modeling regime XFLR5 uses for its inertia evaluation, and suitable for preliminary analysis.

## License

MIT, see [LICENSE](LICENSE)