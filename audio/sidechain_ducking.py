import math

def rms_level(samples):
    """RMS 侧链。"""
    raise NotImplementedError("TODO: rms_level")

def hold_samples(milliseconds, sample_rate):
    """hold 时间。"""
    raise NotImplementedError("TODO: hold_samples")

def attack_samples(milliseconds, sample_rate):
    """attack 采样点。"""
    raise NotImplementedError("TODO: attack_samples")

def release_samples(milliseconds, sample_rate):
    """release 采样点。"""
    raise NotImplementedError("TODO: release_samples")

def stereo_duck(left, right, gain):
    """立体声联动。"""
    raise NotImplementedError("TODO: stereo_duck")

def delay_align(key_delay, signal_delay):
    """延迟对齐。"""
    raise NotImplementedError("TODO: delay_align")

def per_mic_envelope(levels):
    """每路麦克风包络。"""
    raise NotImplementedError("TODO: per_mic_envelope")

def dbfs_to_linear(db):
    """dBFS 转线性。"""
    raise NotImplementedError("TODO: dbfs_to_linear")

def release_smooth(current, target, step):
    """释放平滑。"""
    raise NotImplementedError("TODO: release_smooth")

def fade_in(current, target, step):
    """结束渐入。"""
    raise NotImplementedError("TODO: fade_in")

def downmix_center(center, left, right):
    """下混声道矩阵。"""
    raise NotImplementedError("TODO: downmix_center")

def ducking_gain(level, threshold, ratio):
    """ducking 增益。"""
    raise NotImplementedError("TODO: ducking_gain")

def block_key(index, block):
    """块键。"""
    raise NotImplementedError("TODO: block_key")

def bypass(quality):
    """降级旁路。"""
    raise NotImplementedError("TODO: bypass")

def duck_id(track, index):
    """duck id。"""
    raise NotImplementedError("TODO: duck_id")
