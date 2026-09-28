# 真机与 MuJoCo

默认不连接真机；运行时只通过官方 `piper_sdk` 和 SocketCAN，**不需要 ROS**。本次整理只做 fake SDK 测试，没有执行真机动作。

## 启动

在仓库根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r requirements-hardware.txt
ip -details link show can0
./run.sh --sim mujoco --ik pinocchio --allow-hardware --can can0
# 换 RL：退出上一实例后，将 --ik pinocchio 改成 --ik rl
```

CAN 应为 UP、bitrate 1000000。`ERROR-ACTIVE` 是 CAN 正常错误计数区间的状态名，本身不表示故障；它也不能证明六个电机反馈正常。

未配置 CAN 时，先停用其他控制程序，再执行：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

## 网页操作

打开 http://127.0.0.1:8765，按顺序操作。回零前确认路径无障碍、机械臂有支撑，TCP 和关节零点与模型一致。

| 操作 | 实际行为 |
| --- | --- |
| 接管键盘 | 获取本页控制权；另一页面/终端不再拥有控制权 |
| 连接 | 打开 SDK 并读取关节反馈，不自动使能 |
| 使能 | 清除软件停止、请求六轴使能并检查反馈；从实测关节初始化控制器 |
| 回零 | 绕过 RL / Pinocchio，缓速发送 SDK 六关节零位；不是末端 xyz 归零 |
| 键盘微调 | 基座轴增量 → 绝对 TCP → 平滑 → IK → 连续绝对关节命令 |
| 暂停 / 立即停止 | 停止发送运动并调用 SDK 停止；不等价于六轴已失能 |
| 清除故障 | 清除停止和软件错误；不会自动使能或回零 |
| 返回安全位置并失能 | 回到本次连接首次成功使能的关节姿态，到位且稳定保持 3 秒后请求失能 |
| 断开 | 结束本控制连接；不是自动执行安全位置失能流程 |

“安全位置”只是记录姿态的名称，软件没有碰撞检测，不能自行证明这条返回路径安全。断开 CAN 后记录清除。回零目标为 `-joint_zero_offsets_rad`；安全姿态与 `initial_q` 均不是回零目标。

切换浏览器窗口只释放按键，已接受目标继续跟踪；关闭控制页面/连接断开会停止。目标与实测的瞬时跟踪偏差不会单独触发锁死，但反馈过期、通信失败、关节越界仍会停止。

## 终端可选

先启动网页后端，再在第二终端执行：

```bash
./console.sh --port 8765
```

逐条输入 `connect` → `arm` → `home`。`status` 看反馈；`pin` / `rl` 切求解器；`w` / `a` 等发送一次增量；`stop` 停止；`clear` 清除；`disable` 返回记录姿态并失能。终端会自动接管，网页需点击“接管键盘”才能重新控制。

## 限速、加速度与抖动

默认 `max_joint_velocity_rad_s=null` 不添加统一速度上限，使能时从 SDK 读取逐轴上限供 Ruckig 使用。仍保留软件加速度 1 rad/s²、jerk 10 rad/s³，SDK 速度百分比为 5。回零另用 0.08 rad/s 上限。

软件 Ruckig 加速度与电机驱动内部加速度是两层参数。降低驱动加速度不替代连续参考轨迹；把目标步长调大也不能保证没有超调。

可选工具 [set_driver_acceleration.py](../tools/set_driver_acceleration.py) 用 SDK 读写逐轴驱动参数。`--help` 不访问 CAN；**不带 `--apply` 的正常运行仍会发送读取请求**。带 `--apply` 会写入驱动参数，只允许降低，并执行静止检查和读回验证。这不是安装或启动的必要步骤；不要在另一进程控制同一 CAN 时运行。参数原始单位和固件差异见工具注释，不要把原始 `100` 当作 100 rad/s²。

## 常见问题

| 现象 | 检查 |
| --- | --- |
| 8765 占用 | `ss -ltnp 'sport = :8765'` 查对应进程，回原终端 Ctrl+C；不要另起真机实例抢 CAN |
| 使能后不能回零 | 看 `hardware_armed`、`feedback_valid` 与错误消息；不能仅看按钮点击成功 |
| 已停止 | 清除故障 → 使能 → 回零；记录具体失败消息再排查 |
| 显示追踪但机械臂不动 | 区分目标、发送关节和实测关节；检查六轴使能、反馈时间和 SDK 状态 |

[返回 README](../README.md) · [Pinocchio / 平滑参数](PINOCCHIO.md)
