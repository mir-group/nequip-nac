import torch
import torch.nn.functional

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from nequip_nac import _keys
import pdb


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
                _keys.PER_ATOM_ENERGY_0_KEY: "1x0e",
                _keys.PER_ATOM_ENERGY_1_KEY: "1x0e",
                _keys.NAC_KEY: "1x1o",
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get data with batch key
        # nequip 0.7.1 expects both BATCH_KEY and NUM_NODES_KEY to be present in batched data.
        data = AtomicDataDict.with_batch_(data)

        features = data[AtomicDataDict.NODE_FEATURES_KEY]  # shape (N_atoms, 5)

        data[_keys.PER_ATOM_ENERGY_0_KEY] = features[:, 0].unsqueeze(-1)  # (N_atoms, 1)
        #print("\n Feature \n", features[:, 1].shape)
        delta_e_per_atom = features[:, 1].unsqueeze(-1)  # (N_atoms, 1)
        data[_keys.PER_ATOM_ENERGY_1_KEY] = data[_keys.PER_ATOM_ENERGY_0_KEY] + delta_e_per_atom
        derivative_coupling = torch.narrow(features, -1, 2, 3)  # (Num_atoms, 3)
        #pdb.set_trace()

        # (N_atoms, 1) * (N_atoms, 3) -> (N_atoms, 3)
        data[_keys.NAC_KEY] = delta_e_per_atom * derivative_coupling
        
        return data
