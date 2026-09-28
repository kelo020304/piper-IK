# RL IK：训练、reward 与可调参数

[返回 README](../README.md) · [HTML 版](RL.html) · [Pinocchio](PINOCCHIO.md)

## 1. 当前 RL 在做什么

PPO 反复学习：给定当前关节和 tool0 目标位姿，输出下一次 **六关节角增量**，让 FK 接近目标。训练环境只运行 MuJoCo-Warp 正运动学，不积分电机动力学；无图像、无 FastUMI 示教、无 ROS，也没有 Pinocchio 教师。

运行时 CPU 执行 actor；每个控制周期最多迭代 20 次且预算约 8 ms，再交给 Ruckig 平滑。8 ms 在迭代边界检查，不是实时系统的硬时限。RL 不调用 Pinocchio 修正解；控制台仍加载 Pinocchio 以支持切换与读取模型信息。

源文件：[训练循环](../training/train_rl.py)、[环境和 reward](../training/warp_env.py)、[部署 actor](../piper_ik/policy.py)、[公共控制路径](../piper_ik/engine.py)。

## 2. 26 维观测与 6 维动作

以下为部署权重对应的 `--precision --centered` 模式；位姿均是 **基座中的绝对 tool0 位姿**，不是 TCP，TCP 偏移由 Engine 在输入前转换。

| Python 切片 | 内容 | 单位 / 约定 |
| --- | --- | --- |
| `[0:3]` | 当前位置 | m，基座坐标 |
| `[3:7]` | 当前姿态 | 四元数 **wxyz**，w ≥ 0 |
| `[7:10]` | 目标位置 | m，基座坐标 |
| `[10:14]` | 目标姿态 | 四元数 **wxyz**，w ≥ 0 |
| `[14:20]` | 当前六关节角 | rad，模型绝对关节配置 |
| `[20:23]` | `(p_goal − p_current) × 20` | 位置误差缩放 |
| `[23:26]` | `Log(R_goal @ R_current.T) × 3` | 基座表达的最短旋转向量，rad 后缩放 |

```text
zero_obs = 当前观测，但把目标 pose 设为当前 pose，误差置零
action = clip(actor(obs) − actor(zero_obs), −1, 1)
q_next = joint_limit_clip(q_current + 0.2 × action)
```

减去零误差输出使 **误差为零时确定性动作严格为零**；不能保证非零误差下无振荡。0.2 是每次内部 IK 迭代的最大增量尺度（rad），不是 SDK 速度，也不是每秒的转角。实时物理命令还经过轨迹器。

## 3. reward 的具体公式

入口：`training/warp_env.py → WarpIKEnv.step()`，精度模式：

```text
p = ||p_goal − p_current||                      # m
o = angle(R_goal @ R_current.T)                 # rad
C = log(1 + (p / 0.003)²) + 0.5 × log(1 + (o / radians(1.5))²)
r = (C_before − C_after) − 0.2 × C_after − 0.1 + 100 × success
success = (p < 0.003) AND (o < radians(1.5))
done = success OR episode_steps >= 200
```

| 修改位置 / 数值 | 作用 | 调大 / 调小的意义 |
| --- | --- | --- |
| `step()` 位置尺度 `.003` | 位置代价的归一化尺度 | 调小会更重罚同样位置误差；与 `__init__.position_tol` 是两处，需按目标一起检查 |
| `__init__.orientation_tol` | 成功阈值，同时用于姿态代价尺度 | 调小既提高验收精度也增加姿态代价；不要只改文档验收数字 |
| `step()` 姿态项系数 `.5` | 平移/姿态相对重要性 | 增大更关注朝向，可能牺牲位置收敛；`cost0` 和 `cost` 两处都要改 |
| `cost0 - cost` | 每一步取得进展的奖励 | 系数增大更鼓励快速减误差；当前系数为 1 |
| `-.2 * cost` | 未到位的持续惩罚 | 增大惩罚可减轻停滞，但与进展项、终点奖的比例一起变化 |
| `-.1` | 每步时间代价 | 绝对值增大更偏好少步完成；这不是物理执行时间惩罚 |
| `100 * success` | 到位终点奖励 | 调大突出成功，但可能掩盖途中质量；成功立即结束 episode，没有持续保持训练 |
| `age >= 200` | 训练 episode 上限 | 增大给困难姿态更多尝试，也增加训练成本；不是 200 秒 |

非精度 baseline 使用 `−2 atan(p) − atan(o) + exp(−p/.05) + exp(−o/.2) − .5 + 100 success`，成功阈值 5 cm / 0.2 rad。当前控制台权重加载器要求 `precision=True` 且 `centered=True`，baseline 权重不能直接部署。

**当前 reward 没有惩罚碰撞、关节速度、加速度、jerk 或动作变化率。**如需加入 `||action_t − action_previous||²`，须在环境保存并随 reset 清空上一动作；若要求马尔可夫观测，应把上一动作加入观测，同时修改训练和部署网络输入。只修改 reward 必须重新训练，修改后不会影响旧权重。

## 4. 其他可调位置

| 位置 | 默认 | 意义与依赖 |
| --- | --- | --- |
| `WarpIKEnv.reset()` | 随机关节产生可达目标；precision 下 50% 近目标初态 | `near_fraction` 增大强化局部精修，减少远距离探索；实际碰撞不检查 |
| `reset()` 的 sigma | 对数均匀 0.005–0.25 rad | 局部初始关节扰动范围，决定精修任务难度 |
| `obs()` 与 `Policy.propose()` | 误差尺度 20 / 3 | 训练与部署必须同步修改并重训，不能仅改推理端 |
| `propose()` 与 `Policy.propose()` | 动作尺度 0.2 rad | 调大可更快但可能震荡/越过解；两端同步，重训验证 |
| `ActorCritic` 与 `Policy` | 26→256→256→6，Tanh | 改网络宽度或输入顺序必须同时改两端，旧权重不兼容 |
| `ActorCritic.act()` | `logstd=-.5`，centered 探索缩放 0.02–1 | 标准差大则探索多、精修更难；部署使用确定性 actor，不采样 |
| `main()` 学习率 | precision 5e-4；线性降至约初值的 20% | 大学习率更快但易不稳定；`--resume` 会按新运行重新开始该调度 |
| `main()` horizon / envs | 32 / 2048 | 每次更新采样 65536 步；增加 envs 提高并行度但占显存 |
| `main()` epoch / minibatch | 4 / min(8192,batch) | 同批数据重复更新次数与每次优化批量；更多 epoch 易过拟合当前 rollout |
| GAE γ / λ | 0.99 / 0.95 | 更大强调远期回报；λ 越大方差通常更大 |
| PPO ratio clip | 0.8–1.2 | 更窄约束更新幅度，更稳但可能更慢 |
| critic loss 系数 | 0.5 | 调整价值估计学习权重；不直接表示位置/姿态重要性 |
| gradient norm / target KL | 0.5 / 0.03 | 限制梯度和提前结束当前更新，控制策略变化 |
| entropy | 计算但未加入 loss | 当前没有显式熵奖励；不能把打印出的 entropy 当成正则项 |
| `Policy.solve()` | 20 次 / 8 ms；停止阈值 0.15 mm / 0.15° | 增加迭代可改善部分残差、占用更多控制时间；这是推理停止阈值，不代表模型达到该精度 |

`WarpIKEnv.step()` 将时间上限和成功都视作终止，GAE 不对超时 bootstrap；这是当前实现的选择。修改成截断 bootstrap 时，需保存并使用返回的 `terminal_obs`。

## 5. 训练命令

在仓库根目录建立单独的训练环境，Python 3.10 为原训练版本：

```bash
python3.10 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt

# 从头训练可部署格式；输出路径相对于 training/，也可给绝对路径
env -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python training/train_rl.py --precision --centered \
  --steps 20000000 --envs 2048 --seed 7 --out runs/precision_centered

# 基于仓库现有 checkpoint 继续训练；steps 表示追加采样步数
env -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python training/train_rl.py --precision --centered \
  --resume assets/weights/ppo_precision.pt \
  --steps 20000000 --envs 2048 --seed 7 --out runs/finetune_centered
```

20M 是示例预算，不是收敛承诺。原程序用 `ceil(steps/batch)`，因此实际追加步数向完整 batch 向上取整。`train.jsonl` 记录吞吐、loss、KL；`validation.jsonl` 记录固定种子的 256 个随机目标，`best.pt` / `latest.pt` 保存参数、优化器和配置。不要复用已有输出目录混淆不同实验。

训练使用 MuJoCo-Warp/CUDA，PyTorch 内存比例 0.35 仅约束 PyTorch 分配，不约束 Warp 或整机显存。首次 CUDA 编译需要额外时间；训练时不运行实机控制台。

## 6. 验证与部署新权重

先回到运行环境，用新的独立随机种子在 CPU 上评估；两个方法使用相同目标分布：

```bash
source .venv/bin/activate
python training/evaluate.py --ik rl \
  --weights training/runs/finetune_centered/best.pt \
  --samples 256 --seed 94027 --out artifacts/new-policy.json
python training/evaluate.py --ik pinocchio \
  --samples 256 --seed 94027 --out artifacts/pin-policy.json

# 停止控制台后替换，先在仿真中验证；加载仅接受可信 checkpoint
cp assets/weights/ppo_precision.pt artifacts/ppo_precision.backup.pt
cp training/runs/finetune_centered/best.pt assets/weights/ppo_precision.pt
./run.sh --sim mujoco --ik rl
```

修改工具偏移只改变 Engine 对 TCP 的转换，不必改变 26 维观测。改变连杆、关节数/次序、关节轴或模型零点则会破坏原训练 FK 关系，需要同步模型与训练；当前实现固定六轴。

## 7. 现有权重与训练时间

发布 checkpoint 来自此前本地训练，累计 **615,448,576** 次环境采样；不是单次运行或控制周期数。其保存的随机验证集成功率为 **66.41%（3 mm 且 1.5°，最多 200 次迭代）**，不是实机成功率，也不是所有目标保证。完整参数见 [权重 metadata](../assets/weights/metadata.json)。

历史日志记录：baseline 第一段约 29.4 s，第二段约 779.5 s；从第二段选中 checkpoint 转 centered precision 后，选中权重对应的训练计时区间约 **208.8–209.8 s**。这些是日志内训练计时，不含完整启动、编译与末尾工作，也不构成可迁移的收敛时间保证。另一次 precision v3 实验不是 v4 的父 checkpoint。见 [历史原始记录](historical_training_timing.json)。

按你机器实测 `fps` 估算剩余时间：`剩余追加采样步数 / fps + 后续验证开销`。先看 held-out 成功率、误差分布及保持/连续目标表现，再决定是否停止训练。
