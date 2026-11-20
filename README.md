# pairwise-sidechain-ducking

直播混音里主持人一停顿背景音乐就猛跳回来，侧链检测用瞬时峰值，语音间隙被当成结束；attack 和 release 按固定采样点写死，48k 与 44.1k 下压缩曲线不同；立体声只压左声道，右声道人声还在；回声和延迟效果没对齐侧链键，压的是旧信号；多路麦克风共享一个包络，一个人咳嗽全体降音；阈值标的是 dBFS，输入却是线性幅度，触发点完全不对；释放阶段增益回升过快，出现泵动；旁白结束后音乐渐入太慢，观众以为断流；5.1 下混时中置被重复计入左右，对白忽大忽小；性能档关闭 ducking，音量瞬间突变。audio/sidechain_ducking.py 是核心，tests/test_sidechain_ducking.py 15个红。要修的是：侧链用 RMS 加 hold；attack/release 按采样率换算；立体声联动；延迟对齐键信号；每路麦克风独立包络；dBFS 与线性统一；释放加平滑；结束后按曲线渐入；下混按声道矩阵；降级保留基础 ducking。

## 运行

```bash
python -m pytest tests -q
```

初始状态：15 个用例失败。修复实现使全部用例通过，不要修改 `tests/` 下的测试。

子方向：见平台；任务类型：Bugfix；语言/框架：Python 3, pytest；环境可复现等级：无外部依赖。
