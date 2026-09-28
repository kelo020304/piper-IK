# Pinocchio IK：代价函数与可调参数

[返回 README](../README.md) · [HTML 版](PINOCCHIO.html) · [RL](RL.md)

## 1. 算法实际做什么

Pinocchio 提供正运动学、SE(3) 对数与 Jacobian；本仓库在其上实现带边界的阻尼最小二乘迭代。它不是 PPO，也没有训练过程。

源文件：[pin_solver.py](../piper_ik/pin_solver.py)，实时入口为 `Engine.tick() → PinIK.solve(goal_flange, previous_ik_solution, max_iter=40)`。使用前一拍 IK 解作为起点，避免每次从零开始或随机换分支。当前实时路径**不调用**文件中的 `track()` 或 `solve_point()`。

## 2. 坐标系和代价

输入目标 `G` 是 tool0 在基座中的**绝对位姿**；当前位姿为 `T(q)`。求解器内部：

```text
D = inverse(T(q)) @ G
e = Log_SE3(D) = [v_x, v_y, v_z, ω_x, ω_y, ω_z]
W = diag(1, 1, 1, 0.3, 0.3, 0.3)
pose_cost = ||W e||²
```

`e` 是在当前 tool0 局部坐标系中的相对 SE(3) 误差；前三维是 SE(3) 对数的线性部分，大角度下不应直接等同于简单 `p_goal-p_current`。单独报告的位置误差为两位置的欧氏距离，姿态误差为相对旋转角。

每次迭代使用 LOCAL 帧 Jacobian，并通过 `-Jlog6(D.inverse())` 修正成误差 Jacobian。令 `J_e` 为修正后的 Jacobian，`J=W J_e`：

```text
σ = smallest_singular_value(J)
λ = 1e-7 + max(0, 0.01 − σ)²
μ = 1e-8
Δq = argmin ||J Δq + W e||²
             + λ ||Δq||² + μ ||q + Δq − q_seed||²
```

约束是关节限位与内部迭代步长交集：`max(lower-q,-0.2) ≤ Δq ≤ min(upper-q,0.2)`。代码用 `scipy.optimize.lsq_linear(..., method='bvls')` 解这个子问题，然后从 `[1,.5,.25,.125,.0625]` 选择能降低**位姿代价**的步长。回溯验收没有再把正则项加进 cost；此处不是单一全局优化器一次求到最优解。

## 3. 可调整的位置与意义

| 代码位置 | 默认 | 调整影响 |
| --- | --- | --- |
| `PinIK.__init__.weight` | `[1,1,1,.3,.3,.3]` | 增大后三项更强调姿态；实际平方代价系数为 0.09，不是 0.3。m 与 rad 混合加权，没有通用最优比例 |
| `solve()` 阻尼底值 | `1e-7` | 增大使奇异附近更稳、步子更小，但收敛可能变慢 |
| `solve()` 奇异阈值 | `0.01` | 增大将更早增强阻尼；阈值随权重与 Jacobian 单位一起变化 |
| `solve()` seed 正则 | `1e-8` | 增大倾向少偏离本次 seed，可能降低精度；需要同步 A 和 rhs 中对应项 |
| `solve()` Δq 边界 | `±0.2 rad` | 限制每个内部迭代步，不是速度上限；增大会更激进并可能跨分支 |
| `Engine.tick()` `max_iter` | 40 | 实时迭代预算；增大可能减小残差但拖慢约 50 Hz 循环。`solve()` 默认 100 不代表控制台用了 100 |
| `solve()` 位置阈值 | `1e-4 m` | 算法停止条件；减小会多迭代，不能保证真机相同精度 |
| `solve()` 姿态阈值 | `1e-3 rad`（约 0.057°） | 同上，单位为 rad，不是度 |
| `solve()` BVLS `tol` | `1e-10` | 线性子问题收敛容差；过严可能浪费时间，过松可能降低方向质量 |
| `solve()` alpha 列表 | `1,.5,.25,.125,.0625` | 多加更小步长可能救回局部下降，也增加 FK 开销 |
| `solve_point()` `restarts` | 16 | 仅离线点查询的多起点搜索；不宜直接用于连续实机控制，可能换分支 |
| `track()` vmax / amax | 1.2 rad/s / 6 rad/s² | 备用独立跟踪 API 的参数；**当前控制台没有使用这一路径** |

改 Pinocchio 参数无需重训，但应重跑离线评估和连续目标测试。不要把速度/轨迹平滑调节写到误差权重里：前者在 Ruckig，后者在 IK。

## 4. 与公共平滑器的关系

```text
基座绝对 TCP 目标
  → PoseReference：三层位置 / SO(3) 参考滤波
  → 去除 TCP 偏移，得到基座绝对 tool0 目标
  → PinIK.solve：前一拍 IK 解 warm start
  → Ruckig：绝对关节目标、保留规划速度/加速度
  → SDK 绝对关节角，或仿真关节回放
```

`trajectory.py → PoseReference.step()` 每层 `alpha = 1-exp(-dt/tau)`；姿态使用 `Log(R_source @ R_stage.T)`，左乘最短旋转更新。三层都用 `tau=.06 s`，持续缓变输入的低频滞后大致是三层之和，不应把 tau 当成总延迟。该估算不含 IK、Ruckig、电机和可选速度限幅。

Ruckig 每次目标速度与加速度设为零，但新目标到来时**保留当前规划速度和加速度**，不会每拍强制把当前速度清零。输入目标若密集跳变仍可能导致反复制动；没有零超调保证。参数表在 [README](../README.md)。

## 5. 调整建议与验证

| 表现 | 优先检查 | 不应直接得出的结论 |
| --- | --- | --- |
| 接近目标但姿态残差较大 | 姿态权重、奇异值、是否到关节限位、目标是否可达 | 不一定是迭代次数太少 |
| 起步/换方向抖动 | 键盘目标序列、tau、Ruckig 加速度/jerk、SDK 执行模式 | 不应通过无限放大目标步长解决 |
| 指令已到位但实机跟不上 | 实测/指令差异、SDK speed_percent 与电机限制 | 不是 Pinocchio 代价自动失效 |
| 换姿态时跳分支 | seed 是否被重置、是否误用 solve_point、多解/奇异区 | 同一个末端目标不保证关节解唯一 |

```bash
source .venv/bin/activate
python -m pytest -q tests/test_controls.py tests/test_trajectory.py tests/test_coordinate_contract.py
python training/evaluate.py --ik pinocchio --samples 256 --seed 94027 \
  --out artifacts/pin-evaluation.json
./run.sh --sim mujoco --ik pinocchio
```

当前没有环境碰撞代价、自碰撞检测、力控或可达性全局证明。`IKResult.converged=False` 表示在预算/限位/局部极小等条件下未达到阈值；Engine 仍让轨迹器跟踪返回的最佳局部关节解，界面必须结合残差判断。
