import torch
import torch.nn.functional

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from nequip_nac import _keys


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
                _keys.NAC_KEY: "1x1o",
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get data with batch key
        data = AtomicDataDict.with_batch(data)
        features = data[AtomicDataDict.NODE_FEATURES_KEY]  # shape (N_atoms, 5)

        # map per_frame state to per_node
        if _keys.STATE_KEY not in data:
            state_idx = torch.zeros_like(data[AtomicDataDict.BATCH_KEY])
        else:
            state_idx = torch.index_select(
                data[_keys.STATE_KEY], 0, data[AtomicDataDict.BATCH_KEY]
            )

        # for now assume state is 0 or 1
        # will select one of the energies based on the state
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = torch.gather(
            features, -1, state_idx.unsqueeze(-1)
        )

        # extract and convert NAC from spherical tensor to Cartesian tensor

        data[_keys.NAC_KEY] = torch.narrow(
            features, -1, 2, 3
        )  # shape (N_atoms, 3), only for l=1 no need for to_cartesian()
        return data
