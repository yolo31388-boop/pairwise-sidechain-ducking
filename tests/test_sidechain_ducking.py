import pytest
from audio import sidechain_ducking as mod

CASES = [
    ("rms", 'm.rms_level([3, 4]) == approx(3.535534, rel=1e-5)'),
    ("hold", 'm.hold_samples(5, 48000) == 240'),
    ("attack", 'm.attack_samples(1, 48000) == 48'),
    ("release", 'm.release_samples(10, 48000) == 480'),
    ("stereo", 'm.stereo_duck(1, 2, 0.5) == approx((0.5, 1.0))'),
    ("align", 'm.delay_align(5, 3) == 2'),
    ("mic", 'm.per_mic_envelope([0.2, 0.8, 0.1]) == approx(0.8)'),
    ("dbfs", 'm.dbfs_to_linear(-6) == approx(0.501187, rel=1e-5)'),
    ("release_s", 'm.release_smooth(0, 1, 0.2) == approx(0.2)'),
    ("fade", 'm.fade_in(0, 1, 0.2) == approx(0.2)'),
    ("downmix", 'm.downmix_center(1, 2, 3) == approx(1.75)'),
    ("duck", 'm.ducking_gain(2, 1, 1) == approx(0.5)'),
    ("block", 'm.block_key(10, 4) == 2'),
    ("bypass", 'm.bypass(0) is True'),
    ("id", 'm.duck_id("m1", 3) == "m1:3"')
]

@pytest.mark.parametrize("name,expr", CASES, ids=[c[0] for c in CASES])
def test_contract(name, expr):
    assert eval(expr, {"m": mod, "approx": pytest.approx})
