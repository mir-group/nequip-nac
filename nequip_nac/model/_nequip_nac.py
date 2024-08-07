from typing import Optional
import logging

from nequip.data import AtomicDataDict, AtomicDataset
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from nequip.model import builder_utils

from nequip_nac.nn import NACProcessor

from .. import _keys


def EnergyNACModel(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer

    # .update also maintains insertion order
    layers.update(
        {
            # -- output block --
            "output_hidden_to_2scalars_1vector": (
                AtomwiseLinear,
                dict(irreps_out="2x0e + 1x1o"),
            ),
            "process_energy_nac": NACProcessor,
        }
    )

    layers["total_energy_0_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=_keys.PER_ATOM_ENERGY_0_KEY,
            out_field=_keys.ENERGY_0_KEY,
        ),
    )

    layers["total_energy_1_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=_keys.PER_ATOM_ENERGY_1_KEY,
            out_field=_keys.ENERGY_1_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
