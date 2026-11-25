"""侧链闪避六项修复的行为验证（附加测试，不替换原始契约测试）。

这些测试从题目描述的六个实际问题出发，检查可观察的数值性质，
而不是只检查函数能返回非 None。
"""

import math

import pytest

import media.sidechain_ducking as sd


ALL_ENTRIES = [
    "envelope_smooth", "param_double_buffer", "pause_freeze", "rebind_source",
    "loudness_norm", "frame_align", "determinism", "validation", "audit",
    "fallback", "detect", "metrics", "gain_apply", "duck", "restore",
]


@pytest.mark.parametrize("name", ALL_ENTRIES)
def test_entry_accepts_seed_one(name):
    assert getattr(sd, name)(1) is not None


# --------------------------------------------------------------- 1. 包络平滑

def test_attack_is_gradual_not_instant_jump():
    # 0 -> 1 的阶跃：第一下只能走一部分，且随后继续上行，不能一步到 1。
    out = sd.envelope_smooth([0.0, 1.0, 1.0, 1.0], 50.0, 200.0, 10.0)
    assert 0.0 < out[1] < 0.95
    assert out[1] < out[2] < out[3]
    assert out[3] <= 1.0 + 1e-9


def test_release_is_gradual_not_instant_zero():
    # attack 极快先顶到 1，随后给 0：release 段必须单调回落且第一步不到 0。
    out = sd.smooth_levels([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                           attack_ms=1.0, release_ms=100.0, step_ms=10.0)
    assert out[2] > 0.99
    assert out[3] < out[2]
    assert out[3] > out[4] > out[5] > out[6]
    assert out[3] > 0.5           # release 100ms / step 10ms：第一步仍在 0.9 附近
    assert out[6] > 0.0           # 没有瞬间归零


def test_envelope_report_damps_jump():
    rep = sd.envelope_smooth(1)
    assert rep["attack_gradual"]
    assert rep["release_gradual"]
    assert rep["max_smoothed_jump"] < rep["max_raw_jump"]
    assert rep["no_instant_jump"]


def test_pipeline_frame_jump_far_smaller_than_raw_block_jump():
    p = sd.run_pipeline(1)
    assert p["max_raw_jump"] > 0.5                    # 原始块增益确实会猛跳
    assert p["max_smooth_frame_jump"] < 0.02          # 逐帧增益几乎平滑
    assert p["max_smooth_frame_jump"] < 0.05 * p["max_raw_jump"]


# --------------------------------------------------------------- 2. 参数双缓冲

def test_param_buffer_stages_then_publishes_atomically():
    buf = sd.ParamBuffer({"depth_db": 14, "threshold_db": -30})
    v0 = buf.version
    buf.update(depth_db=20, threshold_db=-24)

    active, v = buf.snapshot()
    assert active["depth_db"] == 14                  # publish 前 active 不变
    assert v == v0 and buf.dirty

    published = buf.publish()
    assert published["depth_db"] == 20
    active2, v2 = buf.snapshot()
    assert active2["threshold_db"] == -24
    assert v2 == v0 + 1


def test_param_double_buffer_report_consistent_no_torn_reads():
    rep = sd.param_double_buffer(1, iterations=120)
    assert rep["staging_isolated"]
    assert rep["atomic_swap"]
    assert rep["torn_reads"] == 0
    assert rep["consistent"]


# --------------------------------------------------------------- 3. 暂停冻结

def test_pause_freezes_envelope_and_ignores_wall_duration():
    rep = sd.pause_freeze(1, pause_wall_ms=5000.0)
    assert rep["frozen"]
    assert rep["decay_during_pause"] == 0.0
    assert rep["envelope_right_after_resume"] == rep["envelope_at_pause"]
    # 错误的墙钟算法会把 5 秒暂停计入 release，包络几乎归零。
    assert rep["buggy_wallclock_envelope_after_resume"] < 1e-6
    assert rep["envelope_at_pause"] > 0.99


def test_engine_pause_emits_silence_and_holds_state():
    eng = sd.DuckingEngine(1)
    eng.bind_source("track-A")
    eng.process([0.5] * sd.DEFAULTS["block_size"], block_rms=[0.9])
    env_before = eng.follower.env
    cursor_before = eng.frame_cursor

    eng.pause()
    silent = eng.process([0.4] * 16, block_rms=[0.9])
    assert silent == [0.0] * 16
    assert eng.follower.env == env_before            # 包络冻结
    assert eng.frame_cursor == cursor_before         # 音频时钟不前进

    eng.resume()
    out = eng.process([0.4] * 16, block_rms=[0.0])
    assert len(out) == 16 and all(math.isfinite(x) for x in out)


# --------------------------------------------------------------- 4. 重绑信号源

def test_rebind_report_follows_new_source_and_rejects_stale():
    rep = sd.rebind_source(1)
    assert rep["rebound"]
    assert rep["new_generation"] == rep["old_generation"] + 1
    assert rep["stale_block_ignored"]
    assert rep["stale_blocks_rejected"] == 1
    assert rep["gain_follows_new_source"]
    assert rep["min_gain_new_track"] > rep["min_gain_old_track"] + 0.2


def test_binder_generation_filters_old_blocks():
    binder = sd.SourceBinder()
    first = binder.bind("A")
    assert binder.is_current(first["generation"])
    second = binder.rebind("B")
    assert not binder.is_current(first["generation"])
    assert binder.is_current(second["generation"])
    assert binder.source_id == "B"


# --------------------------------------------------------------- 5. 参考响度归一化

def test_loudness_norm_consistent_but_peak_norm_not():
    rep = sd.loudness_norm(1)
    assert rep["peak_norm_depth_spread_db"] > 3.0
    assert rep["reference_norm_depth_spread_db"] < 0.5
    assert rep["consistent"]


def test_normalize_to_reference_hits_target_loudness():
    # 一段偏小的信号，归一化后 RMS 响度应落到参考值（容差 0.2 dB）。
    sig = [0.05 * math.sin(2 * math.pi * 440 * i / 48000) for i in range(4800)]
    res = sd.normalize_to_reference(sig, -18.0)
    assert abs(res["output_loudness_db"] - (-18.0)) < 0.2


# --------------------------------------------------------------- 6. 帧对齐

def test_expand_block_gains_exact_mapping():
    gains = [1.0, 0.5, 0.8]
    fg = sd.expand_block_gains(gains, 4)
    assert fg == [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8]


def test_frame_align_non_block_offset():
    # offset=2：绝对帧 4 属于第 1 块。
    fg = sd.frame_align([1.0, 0.5], 4, n_frames=8, frame_offset=2)
    assert fg[2] == 0.5 and fg[3] == 0.5
    assert len(fg) == 8


def test_frame_align_report_zero_error():
    rep = sd.frame_align(1)
    assert rep["aligned"]
    assert rep["max_alignment_error"] == 0.0
    assert rep["length_matches"]


def test_gain_apply_is_sample_aligned():
    audio = [1.0, 2.0, 3.0, 4.0]
    gains = [1.0, 0.5, 0.5, 1.0]
    out = sd.gain_apply(audio, gains)
    assert len(out) == len(audio)
    for a, g, o in zip(audio, gains, out):
        assert o == pytest.approx(a * g)


def test_pipeline_output_frame_aligned():
    p = sd.run_pipeline(1)
    assert len(p["output"]) == p["n"] == len(p["gains"])
    assert p["alignment_error"] == 0.0


# --------------------------------------------------------------- 其余入口

def test_duck_below_threshold_is_unity_and_loud_ducks():
    assert sd.duck(0.001, -30, 18, 14) == pytest.approx(1.0)
    g = sd.duck(1.0, -30, 18, 14)
    assert g == pytest.approx(10 ** (-14 / 20), abs=1e-6)


def test_restore_returns_monotonic_report():
    rep = sd.restore(1)
    assert rep["restored"]
    assert rep["monotonic_restore"]
    assert rep["final_gain"] >= 0.999


def test_determinism_identical_runs():
    rep = sd.determinism(1)
    assert rep["identical"]
    assert rep["runs"] == 2


def test_validation_accepts_defaults_rejects_bad():
    rep = sd.validation(1)
    assert rep["ok"]
    assert rep["bad_config_rejected"]
    assert len(rep["bad_config_errors"]) >= 4


def test_audit_has_block_events():
    rep = sd.audit(1)
    assert rep["blocks"] > 0
    assert len(rep["events"]) == rep["blocks"]
    assert {"block", "rms_db", "raw_gain", "smooth_gain"} <= set(rep["events"][0])


def test_fallback_all_cases_safe():
    rep = sd.fallback(1)
    assert rep["all_safe"]
    assert rep["case_count"] >= 4


def test_metrics_all_requirements_met():
    rep = sd.metrics(1)
    assert all(rep["checks"].values()), rep["checks"]
    assert rep["all_requirements_met"]
