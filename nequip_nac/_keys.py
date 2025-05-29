from typing import Final, List

from nequip.data import register_fields, ABBREV

NAC_KEY: Final[str] = "nac"
ENERGY_0_KEY: Final[str] = "energy_0"
ENERGY_1_KEY: Final[str] = "energy_1"
FORCE_0_KEY: Final[str] = "force_0"
FORCE_1_KEY: Final[str] = "force_1"
PER_ATOM_ENERGY_0_KEY: Final[str] = "per_atom_energy_0"
PER_ATOM_ENERGY_1_KEY: Final[str] = "per_atom_energy_1"

register_fields(
    graph_fields=[ENERGY_0_KEY, ENERGY_1_KEY],
    node_fields=[NAC_KEY, FORCE_0_KEY, FORCE_1_KEY, PER_ATOM_ENERGY_0_KEY, PER_ATOM_ENERGY_1_KEY],
)

ALL_ENERGY_E0_KEYS: Final[List[str]] = [
    PER_ATOM_ENERGY_0_KEY,
    ENERGY_0_KEY,
    FORCE_0_KEY,
]

ALL_ENERGY_E1_KEYS: Final[List[str]] = [
    ENERGY_1_KEY,
    PER_ATOM_ENERGY_1_KEY,
    FORCE_1_KEY,
]

ABBREV.update({
    NAC_KEY: "nac",
    ENERGY_0_KEY: "e0",
    ENERGY_1_KEY: "e1",
    FORCE_0_KEY: "f0",
    FORCE_1_KEY: "f1",
    PER_ATOM_ENERGY_0_KEY: "E0pA",
    PER_ATOM_ENERGY_1_KEY: "E1pA",
    })