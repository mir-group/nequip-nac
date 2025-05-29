import torch
from typing import Optional, List, Union

from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_field_type
from nequip.nn._graph_mixin import GraphModuleMixin
from nequip.utils.global_dtype import _GLOBAL_DTYPE


class PerTypeScaleShift(GraphModuleMixin, torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.
    
    Extended version that supports both scalar and vector fields.
    For vector fields, only scaling is supported (no shifts).

    Note that scaling/shifting is always done casting into the global dtype (``float64``), 
    even if ``model_dtype`` is a lower precision.
    """

    field: str
    out_field: str
    has_scales: bool
    has_shifts: bool
    scales_trainable: bool
    shifts_trainable: bool
    is_vector_field: bool

    def __init__(
        self,
        type_names: List[str],
        field: str,
        out_field: Optional[str] = None,
        scales: Optional[Union[float, List[float]]] = None,
        shifts: Optional[Union[float, List[float]]] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        irreps_in={},
    ):
        super().__init__()
        self.type_names = type_names
        self.num_types = len(type_names)

        # === fields and irreps ===
        self.field = field
        self.out_field = field if out_field is None else out_field
        assert get_field_type(self.field) == "node"
        assert get_field_type(self.out_field) == "node"

        # Check if the field is scalar or vector
        field_irreps = irreps_in[self.field] if self.field in irreps_in else "0e"
        self.is_vector_field = "1o" in str(field_irreps)
        
        if self.is_vector_field and shifts is not None:
            raise ValueError("Shifts are not supported for vector fields, only scaling is allowed")
        
        # Set up irreps - for vector fields, use actual irreps; for scalars, enforce 0e
        my_irreps_in = {self.field: field_irreps if self.is_vector_field else "0e"}
        
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in=my_irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        # === dtype ===
        self.out_dtype = _GLOBAL_DTYPE

        # === preprocess scales and shifts ===
        if isinstance(scales, float):
            scales = [scales]
        if isinstance(shifts, float):
            shifts = [shifts]

        # === scales ===
        self.has_scales = scales is not None
        self.scales_trainable = scales_trainable
        if self.has_scales:
            scales = torch.as_tensor(scales, dtype=self.out_dtype)
            if self.scales_trainable and scales.numel() == 1:
                # effective no-op if self.num_types == 1
                scales = (
                    torch.ones(self.num_types, dtype=scales.dtype, device=scales.device)
                    * scales
                )
            assert scales.shape == (self.num_types,) or scales.numel() == 1
            if self.scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)
        else:
            self.register_buffer("scales", torch.Tensor())
        self.scales_shortcut = self.scales.numel() == 1

        # === shifts (disabled for vector fields) ===
        self.has_shifts = shifts is not None and not self.is_vector_field
        self.shifts_trainable = shifts_trainable and not self.is_vector_field
        if self.has_shifts:
            shifts = torch.as_tensor(shifts, dtype=self.out_dtype)
            if self.shifts_trainable and shifts.numel() == 1:
                # effective no-op if self.num_types == 1
                shifts = (
                    torch.ones(self.num_types, dtype=shifts.dtype, device=shifts.device)
                    * shifts
                )
            assert shifts.shape == (self.num_types,) or shifts.numel() == 1
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)
        else:
            self.register_buffer("shifts", torch.Tensor())
        self.shifts_shortcut = self.shifts.numel() == 1

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # shortcut if no scales or shifts found (only dtype promotion performed)
        if not (self.has_scales or self.has_shifts):
            data[self.out_field] = data[self.field].to(self.out_dtype)
            return data

        # === set up ===
        in_field = data[self.field]
        types = data[AtomicDataDict.ATOM_TYPE_KEY].view(-1)

        if self.has_scales:
            if self.scales_shortcut:
                scales = self.scales
            else:
                scales = torch.index_select(self.scales, 0, types).view(-1, 1)
        else:
            scales = self.scales  # dummy for torchscript

        if self.has_shifts:
            if self.shifts_shortcut:
                shifts = self.shifts
            else:
                shifts = torch.index_select(self.shifts, 0, types).view(-1, 1)
        else:
            shifts = self.shifts  # dummy for torchscript

        # === explicit cast ===
        in_field = in_field.to(self.out_dtype)

        # === scale/shift ===
        if self.has_scales and self.has_shifts:
            # we can used an FMA for performance
            # addcmul computes
            # input + tensor1 * tensor2 elementwise
            # it will promote to widest dtype, which comes from shifts/scales
            in_field = torch.addcmul(shifts, scales, in_field)
        else:
            # fallback path for mix of enabled shifts and scales
            # multiplication / addition promotes dtypes already, so no cast is needed
            if self.has_scales:
                in_field = scales * in_field
            if self.has_shifts:
                in_field = shifts + in_field

        data[self.out_field] = in_field
        return data

    def __repr__(self) -> str:
        def _format_type_vals(vals: List[float], type_names: List[str], element_formatter: str = ".6f") -> str:
            if vals is None or not vals:
                return f"[{', '.join(type_names)}: None]"
            if len(vals) == 1:
                return (f"[{', '.join(type_names)}: {{:{element_formatter}}}]").format(vals[0])
            elif len(vals) == len(type_names):
                return (
                    "["
                    + ", ".join(
                        f"{{{i}[0]}}: {{{i}[1]:{element_formatter}}}" for i in range(len(vals))
                    )
                    + "]"
                ).format(*zip(type_names, vals))
            else:
                raise ValueError(
                    f"Don't know how to format vals=`{vals}` for types {type_names} with element_formatter=`{element_formatter}`"
                )
        
        field_type = "vector" if self.is_vector_field else "scalar"
        scales_str = _format_type_vals(self.scales.tolist() if self.scales.numel() > 0 else [], self.type_names)
        shifts_str = _format_type_vals(self.shifts.tolist() if self.shifts.numel() > 0 else [], self.type_names)
        
        return f"{self.__class__.__name__} ({field_type} field)\n  scales: {scales_str}\n  shifts: {shifts_str}"