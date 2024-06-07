import torch
import torch.nn.functional

from e3nn.io import CartesianTensor

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from nequip_nac._keys import STATE_KEY, NAC_KEY


class NACProcessor(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out={
                AtomicDataDict.PER_ATOM_ENERGY_KEY: "1x0e",
                NAC_KEY: "1x1o",
            },
        )
        self.cart_tensor = CartesianTensor("i")

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[AtomicDataDict.NODE_FEATURES_KEY]  # shape (N_atoms, 5)

        # map per_frame state to per_node
        state_idx = torch.index_select(
            data[STATE_KEY], 0, data[AtomicDataDict.BATCH_KEY]
        )

        # for now assume state is 0 or 1
        # will select one of the energies based on the state
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = torch.gather(features, -1, state_idx)

        # extract and convert NAC from spherical tensor to Cartesian tensor
        data[NAC_KEY] = self.cart_tensor.to_cartesian(
            torch.narrow(features, -1, 2, 3)
        )  # shape (N_atoms, 3)
        return data
