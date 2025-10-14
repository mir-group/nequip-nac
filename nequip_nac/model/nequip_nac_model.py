import math


import logging

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModel,
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
    PerTypeScaleShift,
    ApplyFactor,
)
from nequip.nn.embedding import (
    NodeTypeEmbed,
    PolynomialCutoff,
    EdgeLengthNormalizer,
    BesselEdgeLengthEncoding,
    SphericalHarmonicEdgeAttrs,
)

from nequip.model.utils import model_builder

from .. import _keys
from nequip_nac.nn import NACProcessor, NACForceOutput

from typing import Optional, Dict, Union, Sequence, Callable


@model_builder
def NequIPNACEnergyModel(
    num_layers: int = 4,
    l_max: int = 1,
    parity: bool = True,
    num_features: int = 32,
    radial_mlp_depth: int = 2,
    radial_mlp_width: int = 64,
    **kwargs,
) -> GraphModel:
    """NequIP GNN model that predicts energies for two electronic states and NACs.

    This follows the pattern of NequIPGNNEnergyModel but with modifications for NAC prediction.
    """
    # === sanity checks and warnings ===
    assert (
        num_layers > 0
    ), f"at least one convnet layer required, but found `num_layers={num_layers}`"

    # === spherical harmonics ===
    irreps_edge_sh = repr(
        o3.Irreps.spherical_harmonics(lmax=l_max, p=-1 if parity else 1)
    )

    # === convnet ===
    # convert a single set of parameters uniformly for every layer
    feature_irreps_hidden = repr(
        o3.Irreps(
            [
                (num_features, (l, p))
                for p in ((1, -1) if parity else (1,))
                for l in range(l_max + 1)
            ]
        )
    )

    feature_irreps_hidden_list = [feature_irreps_hidden] * num_layers
    radial_mlp_depth_list = [radial_mlp_depth] * num_layers
    radial_mlp_width_list = [radial_mlp_width] * num_layers

    # === build model ===
    model = FullNequIPNACEnergyModel(
        irreps_edge_sh=irreps_edge_sh,
        type_embed_num_features=num_features,
        feature_irreps_hidden=feature_irreps_hidden_list,
        radial_mlp_depth=radial_mlp_depth_list,
        radial_mlp_width=radial_mlp_width_list,
        **kwargs,
    )
    return model


@model_builder
def NequIPNACModel(**kwargs) -> GraphModel:
    """NequIP GNN model that predicts energies, forces, and NACs.

    This is the main model builder that should be used in configuration files.
    """
    return NACForceOutput(func=NequIPNACEnergyModel(**kwargs))


@model_builder
def FullNequIPNACEnergyModel(
    r_max: float,
    type_names: Sequence[str],
    # convnet params
    radial_mlp_depth: Sequence[int],
    radial_mlp_width: Sequence[int],
    feature_irreps_hidden: Sequence[Union[str, o3.Irreps]],
    # irreps and dims
    irreps_edge_sh: Union[int, str, o3.Irreps],
    type_embed_num_features: int,
    # edge length encoding
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    num_bessels: int = 8,
    bessel_trainable: bool = False,
    polynomial_cutoff_p: int = 6,
    # edge sum normalization
    avg_num_neighbors: Optional[float] = None,
    # per atom energy params
    per_type_energy_0_scales: Optional[Union[float, Dict[str, float]]] = None,
    per_type_energy_0_shifts: Optional[Union[float, Dict[str, float]]] = None,
    per_type_energy_0_scales_trainable: Optional[bool] = False,
    per_type_energy_0_shifts_trainable: Optional[bool] = False,
    per_type_energy_1_scales: Optional[Union[float, Dict[str, float]]] = None,
    per_type_energy_1_shifts: Optional[Union[float, Dict[str, float]]] = None,
    per_type_energy_1_scales_trainable: Optional[bool] = False,
    per_type_energy_1_shifts_trainable: Optional[bool] = False,
    nac_scale: Union[float, Dict[str, float]] = 1.0,
    # == things that generally shouldn't be changed ==
    # convnet
    convnet_resnet: bool = False,
    convnet_nonlinearity_type: str = "gate",
    convnet_nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
    convnet_nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
) -> GraphModel:
    """NequIP GNN model that predicts energies for two electronic states and NACs based on a more extensive set of arguments."""
    # === sanity checks and warnings ===
    assert all(
        tn.isalnum() for tn in type_names
    ), "`type_names` must contain only alphanumeric characters"

    # require every convnet layer to be specified explicitly in a list
    # infer num_layers from the list size
    assert (
        len(radial_mlp_depth) == len(radial_mlp_width) == len(feature_irreps_hidden)
    ), f"radial_mlp_depth: {radial_mlp_depth}, radial_mlp_width: {radial_mlp_width}, feature_irreps_hidden: {feature_irreps_hidden} should all have the same length"
    num_layers = len(radial_mlp_depth)

    if avg_num_neighbors is None:
        logging.warning(
            "Found `avg_num_neighbors=None` -- it is recommended to set `avg_num_neighbors` for normalization and better numerics during training."
        )
    if per_type_energy_0_scales is None:
        logging.warning(
            "Found `per_type_energy_0_scales=None` -- it is recommended to set `per_type_energy_0_scales` for better numerics during training."
        )
    if per_type_energy_0_shifts is None:
        logging.warning(
            "Found `per_type_energy_0_shifts=None` -- it is HIGHLY recommended to set `per_type_energy_0_shifts` as it determines the per-atom energies approaching the isolated atom regime."
        )

    # === encode and embed features ===
    # == edge tensor embedding ==
    spharm = SphericalHarmonicEdgeAttrs(
        irreps_edge_sh=irreps_edge_sh,
    )
    # == edge scalar embedding ==
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
        irreps_in=spharm.irreps_out,
    )
    bessel_encode = BesselEdgeLengthEncoding(
        num_bessels=num_bessels,
        trainable=bessel_trainable,
        cutoff=PolynomialCutoff(polynomial_cutoff_p),
        edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=edge_norm.irreps_out,
    )
    # for backwards compatibility of NequIP's bessel encoding
    factor = ApplyFactor(
        in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        factor=(2 * math.pi) / (r_max * r_max),
        irreps_in=bessel_encode.irreps_out,
    )
    # == node scalar embedding ==
    type_embed = NodeTypeEmbed(
        type_names=type_names,
        num_features=type_embed_num_features,
        irreps_in=factor.irreps_out,
    )
    modules = {
        "spharm": spharm,
        "edge_norm": edge_norm,
        "bessel_encode": bessel_encode,
        "factor": factor,
        "type_embed": type_embed,
    }
    prev_irreps_out = type_embed.irreps_out

    # === convnet layers ===
    for layer_i in range(num_layers):
        current_convnet = ConvNetLayer(
            irreps_in=prev_irreps_out,
            feature_irreps_hidden=feature_irreps_hidden[layer_i],
            convolution_kwargs={
                "radial_mlp_depth": radial_mlp_depth[layer_i],
                "radial_mlp_width": radial_mlp_width[layer_i],
                "avg_num_neighbors": avg_num_neighbors,
                # to ensure isolated atom limit
                "use_sc": layer_i != 0,
            },
            resnet=(layer_i != 0) and convnet_resnet,
            nonlinearity_type=convnet_nonlinearity_type,
            nonlinearity_scalars=convnet_nonlinearity_scalars,
            nonlinearity_gates=convnet_nonlinearity_gates,
        )
        prev_irreps_out = current_convnet.irreps_out
        modules.update({f"layer{layer_i}_convnet": current_convnet})

    # === readout ===
    # NAC readout - takes last convnet output and produces 2 scalars + 1 vector
    nac_readout = AtomwiseLinear(
        irreps_out="2x0e + 1x1o",
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.NODE_FEATURES_KEY,
        irreps_in=prev_irreps_out,
    )
    # NAC processor - splits output into 2 per-atom energy and NAC components
    nac_processor = NACProcessor(
        nac_scale=nac_scale,
        irreps_in=nac_readout.irreps_out,
    )
    modules.update({"nac_readout": nac_readout, "nac_processor": nac_processor})

    # === Per-type scaling and shifting for state 0 ===
    per_type_energy_0_scale_shift = PerTypeScaleShift(
        type_names=type_names,
        field=_keys.PER_ATOM_ENERGY_0_KEY,
        out_field=_keys.PER_ATOM_ENERGY_0_KEY,
        scales=per_type_energy_0_scales,
        shifts=per_type_energy_0_shifts,
        scales_trainable=per_type_energy_0_scales_trainable,
        shifts_trainable=per_type_energy_0_shifts_trainable,
        irreps_in=nac_processor.irreps_out,
    )

    modules.update(
        {
            "per_type_energy_0_scale_shift": per_type_energy_0_scale_shift,
        }
    )
    # === sum to total energy for state 0 ===
    total_energy_0_sum = AtomwiseReduce(
        irreps_in=per_type_energy_0_scale_shift.irreps_out,
        reduce="sum",
        field=_keys.PER_ATOM_ENERGY_0_KEY,
        out_field=_keys.ENERGY_0_KEY,
    )
    modules.update({"total_energy_0_sum": total_energy_0_sum})

    # === Per-type scaling and shifting for state 1 ===
    per_type_energy_1_scale_shift = PerTypeScaleShift(
        type_names=type_names,
        field=_keys.PER_ATOM_ENERGY_1_KEY,
        out_field=_keys.PER_ATOM_ENERGY_1_KEY,
        scales=per_type_energy_1_scales,
        shifts=per_type_energy_1_shifts,
        scales_trainable=per_type_energy_1_scales_trainable,
        shifts_trainable=per_type_energy_1_shifts_trainable,
        irreps_in=total_energy_0_sum.irreps_out,
    )

    modules.update(
        {
            "per_type_energy_1_scale_shift": per_type_energy_1_scale_shift,
        }
    )
    # === sum to total energy for state 1 ===
    total_energy_1_sum = AtomwiseReduce(
        irreps_in=per_type_energy_1_scale_shift.irreps_out,
        reduce="sum",
        field=_keys.PER_ATOM_ENERGY_1_KEY,
        out_field=_keys.ENERGY_1_KEY,
    )
    modules.update({"total_energy_1_sum": total_energy_1_sum})

    # === assemble in SequentialGraphNetwork ===
    model = SequentialGraphNetwork(modules)

    return model
