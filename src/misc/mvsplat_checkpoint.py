"""Strict encoder loading for official and locally fine-tuned MVSplat weights."""
from collections.abc import Mapping

OFFICIAL_FILE_ID = "1Y5OAJN9AX5yMjkbyxag5OnwacdjPnl-N"
OFFICIAL_SHA256 = "83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a"


def load_mvsplat_weights(encoder, checkpoint):
    # Official RE10K files contain only encoder.*; our Lightning files also
    # contain frozen LPIPS/decoder state. Those use the current shared protocol.
    # Resume goes through Lightning instead, restoring the full training state.
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("Expected a MVSplat state_dict mapping")
    state = {k.removeprefix("encoder."): v for k, v in state.items()
             if k.startswith("encoder.")}
    if not state:
        raise ValueError("Expected encoder.* keys in a MVSplat checkpoint")
    return encoder.load_state_dict(state, strict=True)
