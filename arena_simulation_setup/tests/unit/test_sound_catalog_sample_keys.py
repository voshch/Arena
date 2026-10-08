from __future__ import annotations

import pytest

from arena_simulation_setup.tree.assets.sound_catalog import sample_key, split_sample_key


@pytest.mark.parametrize(("asset_id", "variant_id"), [("footstep", "footstep_default_01"), ("lab/chime", "chime_02"), ("motor", "jackal_drivetrain")])
def test_sample_key_round_trips_asset_and_variant(asset_id: str, variant_id: str) -> None:
    key = sample_key(asset_id, variant_id)

    assert key == f"{asset_id}#{variant_id}"
    assert split_sample_key(key) == (asset_id, variant_id)


@pytest.mark.parametrize("key", ["", "footstep", "#footstep_default_01", "footstep#", "#"])
def test_split_sample_key_rejects_keys_without_asset_or_variant(key: str) -> None:
    with pytest.raises(KeyError, match="is not '<asset id>#<variant id>'"):
        split_sample_key(key)
