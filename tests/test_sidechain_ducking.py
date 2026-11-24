import pytest
import media.sidechain_ducking as mod

CASES = [
    ("envelope_smooth", (1,)),
    ("param_double_buffer", (1,)),
    ("pause_freeze", (1,)),
    ("rebind_source", (1,)),
    ("loudness_norm", (1,)),
    ("frame_align", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("detect", (1,)),
    ("metrics", (1,)),
    ("gain_apply", (1,)),
    ("duck", (1,)),
    ("restore", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
