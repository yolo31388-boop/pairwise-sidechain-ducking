"""直播混音侧链 ducking 的基础构件。

所有时间参数以毫秒给出，并在使用时按采样率换算成采样点，避免在
44.1 kHz 与 48 kHz 下得到不同的压缩曲线。
"""

import math


def _ms_to_samples(milliseconds, sample_rate):
    """毫秒换算为采样点，四舍五入到最近的整数采样点。"""
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须为正数")
    if milliseconds < 0:
        raise ValueError("时间不能为负数")
    return int(round(float(milliseconds) * float(sample_rate) / 1000.0))


def _move_towards(current, target, step):
    """按固定步长把 current 拉向 target，不会越过 target。"""
    step = abs(float(step))
    if current < target:
        return min(target, current + step)
    if current > target:
        return max(target, current - step)
    return float(current)


def rms_level(samples):
    """返回一段样本的 RMS 幅度。

    使用均方根而不是瞬时峰值：语音停顿时短促的峰值不应被当成
    句末，背景音乐因此不会在换气间隙突然跳回。
    """
    values = [float(s) for s in samples]
    if not values:
        return 0.0
    mean_square = math.fsum(v * v for v in values) / len(values)
    return math.sqrt(mean_square)


def hold_samples(milliseconds, sample_rate):
    """hold 时间对应的采样点数。"""
    return _ms_to_samples(milliseconds, sample_rate)


def attack_samples(milliseconds, sample_rate):
    """attack 时间对应的采样点数，至少 1 个采样点。"""
    return max(1, _ms_to_samples(milliseconds, sample_rate))


def release_samples(milliseconds, sample_rate):
    """release 时间对应的采样点数，至少 1 个采样点。"""
    return max(1, _ms_to_samples(milliseconds, sample_rate))


def stereo_duck(left, right, gain):
    """对左右声道施加同一个增益，保持立体声联动。

    侧链键检测到人声时，右声道也必须一起压下，不能只压左声道。
    """
    gain = float(gain)
    return (float(left) * gain, float(right) * gain)


def delay_align(key_delay, signal_delay):
    """返回侧链键还需要补的延迟，使键与主干信号对齐。

    键比信号早到时需要补延迟；键已经晚于信号时无需（也无法）提前，
    返回 0。
    """
    return max(0, float(key_delay) - float(signal_delay))


def per_mic_envelope(levels):
    """每路麦克风独立包络的最大值。

    各麦克风分别跟踪自己的包络，聚合时取最大值，这样某一路的
    咳嗽/噪声不会把其它麦克风的增益一起拉低。
    """
    values = [float(v) for v in levels]
    if not values:
        return 0.0
    return max(values)


def dbfs_to_linear(db):
    """dBFS 转线性幅度。"""
    db = float(db)
    if db == -math.inf:
        return 0.0
    return 10.0 ** (db / 20.0)


def release_smooth(current, target, step):
    """释放阶段的平滑：每个采样点只向目标移动一步。

    避免增益快速回升导致的泵动（pumping）。
    """
    return _move_towards(float(current), float(target), step)


def fade_in(current, target, step):
    """旁白结束后的渐入：按步长平滑地回到目标增益。

    与释放平滑分开，保证音乐渐入不会过慢也不会突跳。
    """
    return _move_towards(float(current), float(target), step)


def downmix_center(center, left, right):
    """5.1 下混：中置按 -6 dB 均分到左右后再取双声道平均。

    每个输出声道 = 0.5 * 侧声道 + 0.5 * 中置，函数返回左右两个
    输出声道的平均。中置只计一次，避免对白忽大忽小。
    """
    left_out = 0.5 * float(left) + 0.5 * float(center)
    right_out = 0.5 * float(right) + 0.5 * float(center)
    return (left_out + right_out) / 2.0


def ducking_gain(level, threshold, ratio):
    """根据线性幅度返回 ducking 增益。

    level 与 threshold 使用同一套线性幅度单位（dBFS 需先经
    :func:`dbfs_to_linear` 转换）。信号低于阈值时不压缩；高于阈值时
    在 dB 域按 ``ratio`` 的斜率压下，ratio 越大压得越深。
    """
    level = float(level)
    threshold = float(threshold)
    ratio = float(ratio)
    if level <= 0.0:
        return 1.0
    if threshold <= 0.0:
        return 0.0
    if level <= threshold or ratio <= 0.0:
        return 1.0
    gain = (threshold / level) ** ratio
    return max(0.0, min(1.0, gain))


def block_key(index, block):
    """把采样点索引映射到处理块编号。"""
    if block <= 0:
        raise ValueError("block 必须为正数")
    return int(index) // int(block)


def bypass(quality):
    """性能档判断：最低档跳过增强处理。

    即使降级也只跳过 RMS/hold/平滑等增强，基础 ducking 仍通过
    :func:`ducking_gain` 生效，避免音量瞬间突变。
    """
    return quality <= 0


def duck_id(track, index):
    """生成某条轨道某个处理块的唯一 duck 标识。"""
    return f"{track}:{index}"
