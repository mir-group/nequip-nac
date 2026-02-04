# This file is a part of the `nequip-nac` package. Please see LICENSE and README at the root for information on using it.
from typing import Dict, Optional
from nequip.train import (
    MetricsManager,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    MaximumAbsoluteError,
)
from nequip.data import PerAtomModifier
from nequip_nac._keys import (
    ENERGY_0_KEY,
    ENERGY_1_KEY,
    FORCE_0_KEY,
    FORCE_1_KEY,
    NAC_KEY,
)

# default coefficient keys for validation
_TWO_STATE_EF_METRICS_COEFFS_KEYS = frozenset(
    [
        "per_atom_energy_0",
        "per_atom_energy_1",
        "total_energy_0",
        "total_energy_1",
        "forces_0",
        "forces_1",
        "nac",
    ]
)


def TwoStateEnergyForceLoss(
    coeffs: Dict[str, float],
    per_atom_energy: bool = True,
    type_names=None,
) -> MetricsManager:
    """:class:`~nequip.train.MetricsManager` wrapper for two-state energy, forces, and NAC loss.

    This wrapper creates MSE metrics for:
    - ``energy_0`` (per-atom or total)
    - ``energy_1`` (per-atom or total)
    - ``force_0``
    - ``force_1``
    - ``nac``

    Args:
        coeffs (dict): Coefficients for each loss component. Valid keys are:
            ``per_atom_energy_0``, ``per_atom_energy_1``, ``total_energy_0``,
            ``total_energy_1``, ``forces_0``, ``forces_1``, ``nac``.
            The energy keys depend on the ``per_atom_energy`` argument.
        per_atom_energy (bool): If ``True``, normalize energies by number of atoms.
            Defaults to ``True``.

    Returns:
        MetricsManager: Configured metrics manager for loss computation.

    Example:

    .. code-block:: yaml

        loss:
          _target_: nequip_nac.train.TwoStateEnergyForceLoss
          per_atom_energy: true
          coeffs:
            per_atom_energy_0: 5.0
            per_atom_energy_1: 5.0
            forces_0: 1.0
            forces_1: 1.0
            nac: 5.0
    """
    metrics = []

    # energy_0
    if per_atom_energy:
        assert "per_atom_energy_0" in coeffs, (
            "`per_atom_energy_0` must be in coeffs when per_atom_energy=True"
        )
        metrics.append(
            {
                "name": "E0pA_MSE",
                "field": PerAtomModifier(ENERGY_0_KEY),
                "coeff": coeffs["per_atom_energy_0"],
                "metric": MeanSquaredError(),
            }
        )
    else:
        assert "total_energy_0" in coeffs, (
            "`total_energy_0` must be in coeffs when per_atom_energy=False"
        )
        metrics.append(
            {
                "name": "E0_MSE",
                "field": ENERGY_0_KEY,
                "coeff": coeffs["total_energy_0"],
                "metric": MeanSquaredError(),
            }
        )

    # energy_1
    if per_atom_energy:
        assert "per_atom_energy_1" in coeffs, (
            "`per_atom_energy_1` must be in coeffs when per_atom_energy=True"
        )
        metrics.append(
            {
                "name": "E1pA_MSE",
                "field": PerAtomModifier(ENERGY_1_KEY),
                "coeff": coeffs["per_atom_energy_1"],
                "metric": MeanSquaredError(),
            }
        )
    else:
        assert "total_energy_1" in coeffs, (
            "`total_energy_1` must be in coeffs when per_atom_energy=False"
        )
        metrics.append(
            {
                "name": "E1_MSE",
                "field": ENERGY_1_KEY,
                "coeff": coeffs["total_energy_1"],
                "metric": MeanSquaredError(),
            }
        )

    # force_0
    assert "forces_0" in coeffs, "`forces_0` must be in coeffs"
    metrics.append(
        {
            "name": "F0_MSE",
            "field": FORCE_0_KEY,
            "coeff": coeffs["forces_0"],
            "metric": MeanSquaredError(),
        }
    )

    # force_1
    assert "forces_1" in coeffs, "`forces_1` must be in coeffs"
    metrics.append(
        {
            "name": "F1_MSE",
            "field": FORCE_1_KEY,
            "coeff": coeffs["forces_1"],
            "metric": MeanSquaredError(),
        }
    )

    # nac
    assert "nac" in coeffs, "`nac` must be in coeffs"
    metrics.append(
        {
            "name": "NAC_MSE",
            "field": NAC_KEY,
            "coeff": coeffs["nac"],
            "metric": MeanSquaredError(),
        }
    )

    return MetricsManager(metrics, type_names=type_names)


def TwoStateEnergyForceMetrics(
    coeffs: Optional[Dict[str, float]] = None,
    per_atom_energy: bool = True,
    type_names=None,
) -> MetricsManager:
    """:class:`~nequip.train.MetricsManager` wrapper for two-state energy, forces, and NAC validation metrics.

    This wrapper creates comprehensive validation metrics (MAE, RMSE, MaxAbsError) for:
    - ``energy_0`` (per-atom or total)
    - ``energy_1`` (per-atom or total)
    - ``force_0``
    - ``force_1``
    - ``nac``

    Args:
        coeffs (dict, optional): Coefficients for weighted sum computation. Valid keys are:
            ``per_atom_energy_0``, ``per_atom_energy_1``, ``total_energy_0``,
            ``total_energy_1``, ``forces_0``, ``forces_1``, ``nac``.
            The energy keys depend on the ``per_atom_energy`` argument.
            If ``None``, all coefficients are set to 0.
        per_atom_energy (bool): If ``True``, normalize energies by number of atoms.
            Defaults to ``True``.

    Returns:
        MetricsManager: Configured metrics manager for validation.

    Example:

    .. code-block:: yaml

        val_metrics:
          _target_: nequip_nac.train.TwoStateEnergyForceMetrics
          per_atom_energy: true
          coeffs:
            per_atom_energy_0: 5.0
            per_atom_energy_1: 5.0
            forces_0: 1.0
            forces_1: 1.0
            nac: 5.0
    """
    if coeffs is None:
        coeffs = {}

    # validate coefficient keys
    for key in coeffs.keys():
        assert key in _TWO_STATE_EF_METRICS_COEFFS_KEYS, (
            f"Invalid coefficient key: {key}. Valid keys are: {_TWO_STATE_EF_METRICS_COEFFS_KEYS}"
        )

    metrics = []

    # energy_0 metrics
    if per_atom_energy:
        e0_field = PerAtomModifier(ENERGY_0_KEY)
        e0_coeff = coeffs.get("per_atom_energy_0", 0)
        e0_prefix = "E0pA"
    else:
        e0_field = ENERGY_0_KEY
        e0_coeff = coeffs.get("total_energy_0", 0)
        e0_prefix = "E0"

    metrics.extend(
        [
            {
                "name": f"{e0_prefix}_MAE",
                "field": e0_field,
                "coeff": e0_coeff,
                "metric": MeanAbsoluteError(),
            },
            {
                "name": f"{e0_prefix}_RMSE",
                "field": e0_field,
                "coeff": 0,
                "metric": RootMeanSquaredError(),
            },
            {
                "name": f"{e0_prefix}_MaxAE",
                "field": e0_field,
                "coeff": 0,
                "metric": MaximumAbsoluteError(),
            },
        ]
    )

    # energy_1 metrics
    if per_atom_energy:
        e1_field = PerAtomModifier(ENERGY_1_KEY)
        e1_coeff = coeffs.get("per_atom_energy_1", 0)
        e1_prefix = "E1pA"
    else:
        e1_field = ENERGY_1_KEY
        e1_coeff = coeffs.get("total_energy_1", 0)
        e1_prefix = "E1"

    metrics.extend(
        [
            {
                "name": f"{e1_prefix}_MAE",
                "field": e1_field,
                "coeff": e1_coeff,
                "metric": MeanAbsoluteError(),
            },
            {
                "name": f"{e1_prefix}_RMSE",
                "field": e1_field,
                "coeff": 0,
                "metric": RootMeanSquaredError(),
            },
            {
                "name": f"{e1_prefix}_MaxAE",
                "field": e1_field,
                "coeff": 0,
                "metric": MaximumAbsoluteError(),
            },
        ]
    )

    # force_0 metrics
    f0_coeff = coeffs.get("forces_0", 0)
    metrics.extend(
        [
            {
                "name": "F0_MAE",
                "field": FORCE_0_KEY,
                "coeff": f0_coeff,
                "metric": MeanAbsoluteError(),
            },
            {
                "name": "F0_RMSE",
                "field": FORCE_0_KEY,
                "coeff": 0,
                "metric": RootMeanSquaredError(),
            },
            {
                "name": "F0_MaxAE",
                "field": FORCE_0_KEY,
                "coeff": 0,
                "metric": MaximumAbsoluteError(),
            },
        ]
    )

    # force_1 metrics
    f1_coeff = coeffs.get("forces_1", 0)
    metrics.extend(
        [
            {
                "name": "F1_MAE",
                "field": FORCE_1_KEY,
                "coeff": f1_coeff,
                "metric": MeanAbsoluteError(),
            },
            {
                "name": "F1_RMSE",
                "field": FORCE_1_KEY,
                "coeff": 0,
                "metric": RootMeanSquaredError(),
            },
            {
                "name": "F1_MaxAE",
                "field": FORCE_1_KEY,
                "coeff": 0,
                "metric": MaximumAbsoluteError(),
            },
        ]
    )

    # nac metrics
    nac_coeff = coeffs.get("nac", 0)
    metrics.extend(
        [
            {
                "name": "NAC_MAE",
                "field": NAC_KEY,
                "coeff": nac_coeff,
                "metric": MeanAbsoluteError(),
            },
            {
                "name": "NAC_RMSE",
                "field": NAC_KEY,
                "coeff": 0,
                "metric": RootMeanSquaredError(),
            },
            {
                "name": "NAC_MaxAE",
                "field": NAC_KEY,
                "coeff": 0,
                "metric": MaximumAbsoluteError(),
            },
        ]
    )

    return MetricsManager(metrics, type_names=type_names)
