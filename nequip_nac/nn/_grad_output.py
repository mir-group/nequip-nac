from typing import List, Union, Optional

import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from .. import _keys 

import pdb

@compile_mode("script")
class NACForceOutput(GraphModuleMixin, torch.nn.Module):
    r"""Wrap a model and include as an output its gradient.

    Args:
        func: the model to wrap
        of: the name of the output field of ``func`` to take the gradient with respect to. The field must be a single scalar (i.e. have irreps ``0e``)
        wrt: the input field(s) of ``func`` to take the gradient of ``of`` with regards to.
        out_field: the field in which to return the computed gradients. Defaults to ``f"d({of})/d({wrt})"`` for each field in ``wrt``.
        sign: either 1 or -1; the returned gradient is multiplied by this.
        e.g,
        of=AtomicDataDict.TOTAL_ENERGY_KEY,
        wrt=AtomicDataDict.POSITIONS_KEY,
        out_field=AtomicDataDict.FORCE_KEY,
    """

    def __init__(
        self,
        func: GraphModuleMixin,
    ):
        super().__init__()

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            irreps_out=func.irreps_out,
        )
        self.func = func

        # define irreps of energy and force
        self.irreps_out.update(
            {_keys.FORCE_0_KEY: '1x1o', _keys.FORCE_1_KEY: '1x1o'}
        )
        

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        # set req grad
        data[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
        pos_tensors = data[AtomicDataDict.POSITIONS_KEY]
        
        # run func
        data = self.func(data)
        # Get grads
        force_0_grads = torch.autograd.grad(
            [data[_keys.ENERGY_0_KEY].sum()],
            [pos_tensors],
            create_graph=self.training,
            retain_graph=True,  # needed to allow gradients of this output during training
        )[0]

        data[_keys.FORCE_0_KEY] = torch.neg(force_0_grads)

        force_1_grads = torch.autograd.grad(
            [data[_keys.ENERGY_1_KEY].sum()],
            [pos_tensors],
            create_graph=self.training,
        )[0]

        data[_keys.FORCE_1_KEY] = torch.neg(force_1_grads)

        # unset requires_grad_
        data[AtomicDataDict.POSITIONS_KEY].requires_grad_(False)

        return data


