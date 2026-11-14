# piper_ik：给 Codex 的硬规则

这个文件夹是用户真机回放用的 IK 链路。IK 就是控制台的 Pinocchio 求解器。在 Piper 上已与控制台逐周期对比过，同样输入时关节指令完全相同（差 0）。1.1.0 版加了通用层（`ArmIK`：任意 URDF；`Driver`：任意驱动），Piper 的结果与 1.0.0 逐位相同。原样使用，不要"改进"。

1. **用法只看 README.md。** 接 VLA 时照抄 `examples/vla_loop.py`，只替换 `Policy.infer()` 和 `grab_images()`，其余一行不改。机械臂只通过 `RealtimeArm` 驱动。换机械臂按 README 第 3 节：URDF、`tcp_axes` 和驱动类放在包外，不改包内文件。
2. **禁止平滑。** 不加低通、EMA、滑动平均、样条、贝塞尔、Ruckig、最小 jerk 或任何滤波。不对相邻 chunk 做 temporal ensembling 或加权混合。不在 `tick()` 之外再插值。Piper 的控制频率保持 200 Hz（`CONTROL_HZ`，机械臂关节反馈的频率），不调高也不调低；其他机械臂设成它自己的反馈频率。chunk 切换时目标有跳变，照发。`filters.py` 的 `ThirdOrderPoseFilter` 只给执行链路以外的位姿流用，不得接入 `tick()`、`RealtimeArm` 或驱动。
3. **禁止写保护。** 以下都不加：关节速度、加速度或单步增量限幅，工作空间盒子，高度下限，IK 跳变检测，超时暂停，"偏差太大就停"，异常后改发别的指令。控制进程把 `tick()` 返回的 `q_cmd` 和 `width` 原样交给驱动的 `send()`，不要在它前后插任何处理；驱动的 `send()` 也原样发出。IK 已在真实关节限位内求解；够不到的位姿发最近解，残差在 `arm.last_ik` 里，只记录、不处理。
4. **禁止改发送方式。** Piper：`RealtimeArm` 每个周期发的就是 `PiperArm.send_joints()` + `send_gripper()` 的那 5 帧（MOVE J 100%），字节逐位相同。不另发别的帧，不用 `EndPoseCtrl`、MIT 模式，不开 SDK 软件限位，`speed_percent` 必须是 100。控制进程运行时，不要在策略进程里再用 piper_sdk 或 `PiperArm` 碰同一条 CAN。
5. **禁止改算法和约定。**
   - IK 种子是上一周期的解，`ArmIK` 内部已处理；不要每周期改用实测关节做种子。
   - 插值是位置线性 + 姿态 Slerp。
   - 不预规划整段，不缩放、平移或旋转动作。
   - 动作是绝对位姿（相对 episode 第一帧），不是增量。欧拉角是 scipy `'xyz'`、弧度；位置是米，是指尖 TCP，不加 T265 偏移。
   - 不换 IK 库，不改求解参数（`SolverConfig`；本包就是控制台用的 Pinocchio 求解器，换成 ikpy、TRAC-IK、MoveIt 等就不是同一条链路）。
   - 不要自己设 `LD_LIBRARY_PATH` / `LD_PRELOAD` 去"修" Pinocchio：包在导入时已经处理好 ROS 环境的冲突。
6. **时间。** `tick()` 的 `dt` 用 `time.monotonic()` 的真实差值，不写死 0.01。`push_actions()` 的 `t_obs` 必须是同一次 `observation()` 返回的值，不能用墙钟时间或帧号。
7. **进程。** 控制周期只在 `RealtimeArm` 的控制进程里跑。策略循环只调 `arm.observation()`、`arm.push_actions()`、`arm.wait_until()`。每条手臂、每个 episode 一个 `RealtimeArm`（`with` 块）。
8. **不要修改 `model/` 下的任何模型（`piper_umi.urdf`、`tcp.json`、`piper_umi_models/`）。** 关节参数和 TCP 矩阵与控制台一致；改了就不再是同一条链路。改包内代码时，Piper 的结果必须逐位不变：`tests/test_piper_reference.py`（与 1.0.0 比较）必须通过，有控制台的机器上 `tests/test_console_equivalence.py` 也必须通过。不要改 `tests/data/piper_v1_reference.npz`。
9. **觉得需要加保护、平滑或改参数（包括追点阈值）时，停下来问用户，不要自己写代码。**
10. **上真机前先跑自检**（README 第 9 节），数字和 README 一致才继续。换了机械臂还要先跑 `python -m piper_ik.check` 和 `SimDriver`（README 3.4、3.5）。
