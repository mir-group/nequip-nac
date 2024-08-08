from typing import List, Optional, Union

import torch

from nequip.nn import GraphModuleMixin
from nequip.data import AtomicDataDict, AtomicDataset
from nequip.model._scaling import _PerSpeciesRescale, GlobalRescale

from .. import _keys

def RescaleEnergyEtcE0(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
):
    return GlobalRescale(
        model=model,
        config=config,
        dataset=dataset,
        initialize=initialize,
        module_prefix="global_rescale_E0",
        default_scale=(
            f"dataset_{_keys.FORCE_0_KEY}_rms"
            if _keys.FORCE_0_KEY in model.irreps_out
            else f"dataset_{_keys.ENERGY_0_KEY}_std"
        ),
        default_shift=None,
        default_scale_keys=_keys.ALL_ENERGY_E0_KEYS,
        default_shift_keys=[],
    )

def RescaleEnergyEtcE1(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
):
    return GlobalRescale(
        model=model,
        config=config,
        dataset=dataset,
        initialize=initialize,
        module_prefix="global_rescale_E1",
        default_scale=(
            f"dataset_{_keys.FORCE_1_KEY}_rms"
            if _keys.FORCE_1_KEY in model.irreps_out
            else f"dataset_{_keys.ENERGY_1_KEY}_std"
        ),
        default_shift=None,
        default_scale_keys=_keys.ALL_ENERGY_E1_KEYS,
        default_shift_keys=[],
    )

def RescaleEnergyEtcNAC(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
):
    return GlobalRescale(
        model=model,
        config=config,
        dataset=dataset,
        initialize=initialize,
        module_prefix="global_rescale_NAC",
        default_scale=(
            f"dataset_{_keys.NAC_KEY}_rms"
        ),
        default_shift=None,
        default_scale_keys=[_keys.NAC_KEY],
        default_shift_keys=[],
    )



def NACPerSpeciesRescaleE0(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
):
    """Add per-atom rescaling (and shifting) for per-atom energies."""
    module_prefix = "per_species_rescale_E0"

    # Check for common double shift mistake with defaults
    if "RescaleEnergyEtc" in config.get("model_builders", []):
        # if the defaults are enabled, then we will get bad double shift
        # THIS CHECK IS ONLY GOOD ENOUGH FOR EMITTING WARNINGS
        has_global_shift = config.get("global_rescale_shift", None) is not None
        if has_global_shift:
            if config.get(module_prefix + "_shifts", True) is not None:
                # using default of per_atom shift
                raise RuntimeError(
                    "A global_rescale_shift was provided, but the default per-atom energy shift was not disabled."
                )
        del has_global_shift

    return _PerSpeciesRescale(
        scales_default=None,
        shifts_default=f"dataset_per_atom_{_keys.ENERGY_0_KEY}_mean",
        field=_keys.PER_ATOM_ENERGY_0_KEY,
        out_field=_keys.PER_ATOM_ENERGY_0_KEY,
        module_prefix=module_prefix,
        insert_before="total_energy_0_sum",
        model=model,
        config=config,
        initialize=initialize,
        dataset=dataset,
    )



def NACPerSpeciesRescaleE1(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
):
    """Add per-atom rescaling (and shifting) for per-atom energies."""
    module_prefix = "per_species_rescale_E1"

    # Check for common double shift mistake with defaults
    if "RescaleEnergyEtc" in config.get("model_builders", []):
        # if the defaults are enabled, then we will get bad double shift
        # THIS CHECK IS ONLY GOOD ENOUGH FOR EMITTING WARNINGS
        has_global_shift = config.get("global_rescale_shift", None) is not None
        if has_global_shift:
            if config.get(module_prefix + "_shifts", True) is not None:
                # using default of per_atom shift
                raise RuntimeError(
                    "A global_rescale_shift was provided, but the default per-atom energy shift was not disabled."
                )
        del has_global_shift

    return _PerSpeciesRescale(
        scales_default=None,
        shifts_default=f"dataset_per_atom_{_keys.ENERGY_1_KEY}_mean",
        field=_keys.PER_ATOM_ENERGY_1_KEY,
        out_field=_keys.PER_ATOM_ENERGY_1_KEY,
        module_prefix=module_prefix,
        insert_before="total_energy_1_sum",
        model=model,
        config=config,
        initialize=initialize,
        dataset=dataset,
    )

