from __future__ import annotations

import pytest

from forgeworld.tokenizers import TokenBatch, TokenBatchError


def test_token_batch_validates_shapes() -> None:
    batch = TokenBatch(
        values=((1.0, 2.0), (3.0, 4.0)),
        timestamps=("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"),
        semantic_ids=("current", "current"),
        asset_ids=("asset_a", "asset_a"),
        masks=((True, True), (True, False)),
        units=("A", "A"),
    )

    assert batch.token_count == 2
    assert batch.feature_count == 2


def test_token_batch_rejects_mask_shape_mismatch() -> None:
    with pytest.raises(TokenBatchError):
        TokenBatch(
            values=((1.0, 2.0),),
            timestamps=("2026-01-01T00:00:00Z",),
            semantic_ids=("current",),
            asset_ids=("asset_a",),
            masks=((True,),),
            units=("A",),
        )
