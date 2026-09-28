# RL 求解器与训练

[README](../README.md) · [Pinocchio](PINOCCHIO.md)

## 使用

```bash
pip install -e '.[rl]'
```

```python
from piper_ik import PiperIK, RLConfig

ik = PiperIK("rl", rl=RLConfig(
    max_iterations=20,
    time_budget_s=0.008,
    position_tolerance=0.003,
    orientation_tolerance=0.02617993878,  # 1.5°，以 rad 传入
))
result = ik.solve(target, seed=q)
```

这里的 RL 是 **PPO 运动学 IK**：由当前关节和目标位姿预测关节增量。没有图像、任务示教或动力学训练。对外接口仍返回绝对关节解；TCP 转换和误差验收由 SDK 完成。

当前权重只适配包内 Piper 模型和 tool0 帧；工具长度通过 `tcp` 传入。不可将修改连杆或关节轴后的模型直接交给旧权重。

## 推理配置

| `RLConfig` 参数 | 默认 | 意义 |
| --- | --- | --- |
| `weights` | 包内 checkpoint | 可设为自己的可信 checkpoint 路径，无需覆盖原文件 |
| `max_iterations` | 20 | 每次 solve 的最大策略迭代次数；增大通常增加耗时 |
| `time_budget_s` | 0.008 | 在迭代边界检查的软预算；单次推理可能超时，None 为不设时间预算 |
| `position_tolerance` | 0.003 m | 位置停止／验收阈值 |
| `orientation_tolerance` | 1.5° 对应 rad | 姿态停止／验收阈值；调小不代表旧权重就能达到 |

权重是 26→256→256→6 的 Tanh actor，在 CPU 推理。库不修改调用进程的 PyTorch 全局线程数；小网络在过多线程时可能更慢，应用可自行比较 `torch.set_num_threads(1)` 的效果。

## 观测与动作

实现：[部署 actor](../piper_ik/_policy.py)、[训练网络](../training/train_rl.py)、[训练环境](../training/warp_env.py)。

| 观测切片 | 内容 |
| --- | --- |
| `[0:3]` / `[7:10]` | 当前 / 目标 tool0 的基座绝对位置，m |
| `[3:7]` / `[10:14]` | 当前 / 目标旋转，四元数 **wxyz**，规范到 w ≥ 0 |
| `[14:20]` | 当前六个绝对关节角，rad |
| `[20:23]` | 基座位置误差 × 20 |
| `[23:26]` | `Log(R_goal @ R_current.T)` × 3，基座旋转向量 |

```text
zero_obs = obs，但目标 pose 改成当前 pose，误差置零
action = clip(actor(obs) − actor(zero_obs), −1, 1)
q_next = joint_limit_clip(q + 0.2 × action)
```

减去零误差输出保证误差为零时确定性动作也为零，不保证其他情况下没有振荡。0.2 是内部单次迭代角增量尺度，**不是关节速度**。

## Reward

修改位置：`training/warp_env.py → WarpIKEnv.step()`，以下为 `--precision --centered`：

```text
p = 位置误差，m；o = 姿态误差，rad
C = log(1 + (p/0.003)²) + 0.5 × log(1 + (o/radians(1.5))²)
r = C_before − C_after − 0.2 × C_after − 0.1 + 100 × success
success = p < 0.003 AND o < radians(1.5)
done = success OR age >= 200
```

| 修改项 | 作用与影响 |
| --- | --- |
| 位置尺度 `.003` | 越小，同样位置误差受到更大惩罚；同时检查独立的 `position_tol` |
| `orientation_tol` | 既是成功阈值也是姿态代价归一化尺度；调小提高训练难度 |
| 姿态系数 `.5` | 调大更关注旋转精度；`cost0`、`cost` 两处同步修改 |
| 进展项 `cost0-cost` | 当前系数 1；增大鼓励更快降低误差 |
| 残差惩罚 `-.2*cost` | 加大可减少停滞，但改变与终点奖励的比例 |
| 每步惩罚 `-.1` | 加大绝对值偏好更少迭代，并非真实运动时间 |
| 成功奖励 `100` | 加大强化到位；成功后立即终止，没有持续保持训练 |
| episode 上限 `200` | 增大让困难目标尝试更多步，也增加训练成本 |

当前 reward 没有碰撞、速度、加速度或动作变化率惩罚。新增动作平滑惩罚时需维护并 reset 上一动作；若加入观测，还必须同时修改训练与部署的输入维度。改 reward 后需要重新训练，旧权重不变。

## 训练参数

| 位置 | 默认 | 调整意义 |
| --- | --- | --- |
| `WarpIKEnv.reset()` | precision 下 50% 近目标初态 | 增大 `near_fraction` 强化局部精修，减少远距离探索 |
| reset 关节扰动 | 对数均匀 0.005–0.25 rad | 决定局部问题难度 |
| 观测缩放 / 动作尺度 | 20、3 / 0.2 rad | 环境与部署必须同步修改，重训后验证 |
| `main()` 学习率 | precision 5e-4，降至约 20% | 增大可能更快但不稳定；resume 按新运行重启调度 |
| horizon / envs | 32 / 2048 | 每次更新 65,536 transitions，更多 envs 占用更多显存 |
| epochs / minibatch | 4 / min(8192,batch) | 每批重复优化次数与批量；增加 epochs 可能过拟合 rollout |
| GAE γ / λ | 0.99 / 0.95 | 回报时间范围与偏差/方差权衡 |
| PPO clip / critic 系数 | 0.8–1.2 / 0.5 | 更新幅度与价值学习权重 |
| 梯度上限 / target KL | 0.5 / 0.03 | 限制更新，KL 过大提前结束当前更新 |
| entropy | 计算但不加入 loss | 当前没有显式熵奖励 |

时间上限与成功均按终止处理，GAE 不对超时 bootstrap。非 precision baseline 使用另一套 reward，且不能直接装入当前部署 actor；可部署训练需同时指定 `--precision --centered`。

## 训练命令

仓库根目录执行。建议单独 Python 3.10 CUDA 环境，GPU 依赖固定在 `training/requirements.txt`。

```bash
python3.10 -m venv .venv-train
source .venv-train/bin/activate
pip install -r training/requirements.txt

# 从头训练；out 相对于 training/，也接受绝对路径
python training/train_rl.py --precision --centered \
  --steps 20000000 --envs 2048 --seed 7 --out runs/centered

# 基于现有权重继续训练，steps 为追加采样量
python training/train_rl.py --precision --centered \
  --resume piper_ik/assets/ppo_precision.pt \
  --steps 20000000 --envs 2048 --seed 7 --out runs/finetune
```

输出包括 `best.pt`、`latest.pt`、`train.jsonl`、`validation.jsonl`。实际步数向完整 batch 向上取整；不要将不同实验写入同一目录。GPU 内存比例限制仅覆盖 Torch，不覆盖 Warp。

回到 SDK 环境进行独立评估和使用：

```bash
source .venv/bin/activate
python training/evaluate.py --ik rl \
  --weights training/runs/finetune/best.pt --samples 256 \
  --out artifacts/rl.json
```

```python
ik = PiperIK("rl", rl=RLConfig(weights="training/runs/finetune/best.pt"))
```

## 权重与收敛记录

随包权重累计 615,448,576 transitions；历史固定 256 个随机样本、最多 200 次迭代下，3 mm / 1.5° 成功率为 **66.41%**。不是本接口 8 ms 预算下的保证，也不是通用收敛证明。

历史 centered precision 阶段选中 checkpoint 的训练计时区间约 **208.8–209.8 s**，从已有约 495M 步 checkpoint 继续训练，**不是从零训练时间**。计时不含全部初始化和尾部工作；20M 示例预算也不保证收敛。见 [权重元数据](../piper_ik/assets/metadata.json)、[历史计时](historical_training_timing.json)。
