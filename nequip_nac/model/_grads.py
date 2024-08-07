from ..nn import NACForceOutput as NACForceOutputModule
from .. import _keys
from nequip.nn import GraphModuleMixin
from nequip.data import AtomicDataDict

def NACForceOutput(model: GraphModuleMixin) -> NACForceOutputModule:
    r"""Add forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """

    return NACForceOutputModule(
        func=model,
    )
