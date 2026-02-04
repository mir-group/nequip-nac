# This file is a part of the `nequip-nac` package. Please see LICENSE and README at the root for information on using it.
from typing import Dict, List
from nequip.data import DataStatisticsManager, PerAtomModifier, NumNeighbors
from nequip.data.stats import Mean, RootMeanSquare
from nequip_nac._keys import (
    ENERGY_0_KEY,
    ENERGY_1_KEY,
    FORCE_0_KEY,
    FORCE_1_KEY,
    NAC_KEY,
)


def TwoStateDataStatisticsManager(
    dataloader_kwargs: Dict = {},
    type_names: List[str] = None,
):
    """:class:`~nequip.data.DataStatisticsManager` wrapper for two-state NAC datasets.

    This wrapper computes dataset statistics for training two-state NAC models.
    The computed statistics include:

    - ``num_neighbors_mean``: Average number of neighbors
    - ``per_atom_energy_0_mean``: Mean per-atom energy for state 0
    - ``per_atom_energy_1_mean``: Mean per-atom energy for state 1
    - ``per_type_force_0_rms``: RMS of forces for state 0, per atom type
    - ``per_type_force_1_rms``: RMS of forces for state 1, per atom type
    - ``nac_rms``: RMS of non-adiabatic couplings

    These statistics can be interpolated in the model configuration for
    initialization and normalization.

    Args:
        dataloader_kwargs (dict): Arguments for :class:`torch.utils.data.DataLoader`
            for dataset statistics computation. The ``batch_size`` should be as large
            as possible without triggering OOM.
        type_names (list): List of atom type names (must match the model's ``type_names``).
            Required for per-type statistics.

    Returns:
        DataStatisticsManager: Configured statistics manager.

    Example:

    .. code-block:: yaml

        data:
          _target_: nequip.data.datamodule.ASEDataModule
          # ... other data parameters ...

          stats_manager:
            _target_: nequip_nac.data.TwoStateDataStatisticsManager
            type_names: ${model_type_names}
            dataloader_kwargs:
              batch_size: ${batch_size}

        training_module:
          model:
            _target_: nequip_nac.model.NequIPNACModel

            # use computed statistics
            avg_num_neighbors: ${training_data_stats:num_neighbors_mean}
            per_type_energy_0_shifts: ${training_data_stats:per_atom_energy_0_mean}
            per_type_energy_0_scales: ${training_data_stats:per_type_force_0_rms}
            per_type_energy_1_shifts: ${training_data_stats:per_atom_energy_1_mean}
            per_type_energy_1_scales: ${training_data_stats:per_type_force_1_rms}
            nac_scale: ${training_data_stats:nac_rms}
    """
    metrics = [
        {
            "name": "num_neighbors_mean",
            "field": NumNeighbors(),
            "metric": Mean(),
        },
        {
            "name": "per_atom_energy_0_mean",
            "field": PerAtomModifier(ENERGY_0_KEY),
            "metric": Mean(),
        },
        {
            "name": "per_atom_energy_1_mean",
            "field": PerAtomModifier(ENERGY_1_KEY),
            "metric": Mean(),
        },
        {
            "name": "per_type_force_0_rms",
            "field": FORCE_0_KEY,
            "metric": RootMeanSquare(),
            "per_type": True,
        },
        {
            "name": "per_type_force_1_rms",
            "field": FORCE_1_KEY,
            "metric": RootMeanSquare(),
            "per_type": True,
        },
        {
            "name": "nac_rms",
            "field": NAC_KEY,
            "metric": RootMeanSquare(),
        },
    ]
    return DataStatisticsManager(metrics, dataloader_kwargs, type_names)
