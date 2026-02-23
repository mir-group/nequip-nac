import pytest
from nequip.utils.unittests.model_tests_basic import EnergyModelTestsMixin


COMMON_CONFIG = {
    "_target_": "nequip_nac.model.NequIPNACModel",
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
    "avg_num_neighbors": 10.0,
    "num_bessels": 8,
    "bessel_trainable": False,
    "polynomial_cutoff_p": 6,
    "num_layers": 2,
    "l_max": 2,
    "parity": True,
    "num_features": 16,
    "radial_mlp_depth": 2,
    "radial_mlp_width": 16,
}

minimal_config0 = dict(
    **COMMON_CONFIG,
)

minimal_config1 = dict(
    per_edge_type_cutoff={"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9},
    **COMMON_CONFIG,
)


class TestNequIPNAC(EnergyModelTestsMixin):
    """Test suite for NequIP-NAC models"""

    @pytest.fixture
    def strict_locality(self):
        return False

    def total_energy_keys(self):
        """Override to test NAC-specific energy keys."""
        from nequip_nac._keys import ENERGY_0_KEY, ENERGY_1_KEY

        return [ENERGY_0_KEY, ENERGY_1_KEY]

    def per_atom_energy_keys(self):
        """Override to test NAC-specific per-atom energy keys."""
        from nequip_nac._keys import PER_ATOM_ENERGY_0_KEY, PER_ATOM_ENERGY_1_KEY

        return [PER_ATOM_ENERGY_0_KEY, PER_ATOM_ENERGY_1_KEY]

    def force_keys(self):
        """Override to test NAC-specific force keys."""
        from nequip_nac._keys import FORCE_0_KEY, FORCE_1_KEY

        return [FORCE_0_KEY, FORCE_1_KEY]

    @pytest.fixture(
        params=[
            minimal_config0,
            minimal_config1,
        ],
        scope="class",
    )
    def config(
        self,
        request,
    ):
        config = request.param
        config = config.copy()
        return config

    def test_import(self):
        """Test that the nequip_nac module can be imported"""
        import nequip_nac

        assert hasattr(nequip_nac, "__version__")

    def test_keys(self):
        """Test that NAC-specific keys are properly defined"""
        from nequip_nac._keys import (
            NAC_KEY,
            ENERGY_0_KEY,
            ENERGY_1_KEY,
            FORCE_0_KEY,
            FORCE_1_KEY,
        )

        assert NAC_KEY is not None
        assert ENERGY_0_KEY is not None
        assert ENERGY_1_KEY is not None
        assert FORCE_0_KEY is not None
        assert FORCE_1_KEY is not None
