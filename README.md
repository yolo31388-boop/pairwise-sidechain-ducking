# pairwise-sidechain-ducking

## 领域

游戏、图形与多媒体

## 主题

音频侧链闪避

## 说明

包络平滑、参数双缓冲与音频帧对齐。

初始状态下 `media/sidechain_ducking.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
