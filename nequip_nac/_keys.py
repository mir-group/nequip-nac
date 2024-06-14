from typing import Final

from nequip.data import register_fields
from nequip.train._key import ABBREV

STATE_KEY: Final[str] = "state"
NAC_KEY: Final[str] = "nac"

register_fields(
    long_fields=[STATE_KEY],
    #graph_fields=[STATE_KEY],
    node_fields=[NAC_KEY],
)

ABBREV.update({NAC_KEY: "nac"})
