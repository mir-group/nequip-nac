import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


@compile_mode("script")
class NACForceOutput(GraphModuleMixin, torch.nn.Module):
    r"""Compute forces from energies of two electronic states.

    This class wraps an energy model that predicts energies for multiple electronic states
    and computes the corresponding forces using automatic differentiation.

    Args:
        func: the energy model to wrap
        num_states: number of electronic states (default: 2)
        energy_key_pattern: pattern for energy field names, formatted with state index
        force_key_pattern: pattern for force field names, formatted with state index

    """

    def __init__(
        self,
        func: GraphModuleMixin,
        num_states: int = 2,
        energy_key_pattern: str = "energy_{}",
        force_key_pattern: str = "force_{}",
    ):
        super().__init__()
        self.func = func

        self.num_states = num_states
        self.energy_keys = [energy_key_pattern.format(i) for i in range(num_states)]
        self.force_keys = [force_key_pattern.format(i) for i in range(num_states)]

        # Validate that all energy fields are scalars
        energy_irreps_check = {
            energy_key: Irreps("0e") for energy_key in self.energy_keys
        }

        # Check and init irreps
        self._init_irreps(
            irreps_in=self.func.irreps_in.copy(),
            my_irreps_in=energy_irreps_check,
            irreps_out=self.func.irreps_out.copy(),
        )

        # Define irreps of forces for all states
        force_irreps = {force_key: "1x1o" for force_key in self.force_keys}
        self.irreps_out.update(force_irreps)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # Set requires grad for positions
        pos = data[AtomicDataDict.POSITIONS_KEY]
        pos.requires_grad_(True)

        # Run the energy model
        data = self.func(data)

        # Calculate forces for all states
        for i in range(self.num_states):
            # Retain graph for next state
            if i < len(self.energy_keys) - 1:
                retain_graph = True
            else:
                retain_graph = self.training

            force_grads = torch.autograd.grad(
                outputs=[data[self.energy_keys[i]].sum()],
                inputs=[pos],
                create_graph=self.training,
                retain_graph=retain_graph,
            )[0]

            if force_grads is None:
                raise RuntimeError(f"Failed to compute gradient for state {i}")

            data[self.force_keys[i]] = torch.neg(force_grads)

        # Unset requires_grad
        pos.requires_grad_(False)

        return data
