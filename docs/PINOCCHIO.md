# Pinocchio 求解器

[README](../README.md) · [滤波器](FILTER.md) · [RL](RL.md)

## 使用

```python
from piper_ik import PiperIK, PinocchioConfig

ik = PiperIK("pinocchio", pinocchio=PinocchioConfig(
    weights=(1, 1, 1, .3, .3, .3),
    damping=1e-7,
    max_iterations=40,
))
result = ik.solve(target, seed=q)
```

输入为基座中的绝对 TCP 位姿。SDK 去除固定 TCP 偏移后，后端求解 tool0 位姿；输出为绝对关节角。默认使用包内 Piper 模型，也可传 `urdf="/path/model.urdf", frame="tool0"`，要求六个标量关节。自定义模型只适用于 Pinocchio 或自定义后端，现有 RL 权重不能直接复用。

## 代价函数

实现：[内部求解器](../piper_ik/_pinocchio.py)，配置：[SDK](../piper_ik/sdk.py)。

```text
D = inverse(T_current) @ T_goal
e = Log_SE3(D)
W = diag(1, 1, 1, 0.3, 0.3, 0.3)
J = W @ (-Jlog6(inverse(D)) @ J_LOCAL)
λ = damping + max(0, singularity_threshold − σ_min(J))²
μ = seed_weight

min_Δq ||J Δq + W e||² + λ ||Δq||² + μ ||q + Δq − q_seed||²
```

误差和 Jacobian 在当前 tool0 局部坐标系表达；这不改变输入目标的基座坐标约定。SE(3) 对数前三维在大角度下不等于简单的两位置相减。

增量受关节限位和 `±max_joint_step` 限制。每步用有界最小二乘求方向，再尝试 `1、0.5、0.25、0.125、0.0625` 的步长，只接受降低位姿代价的结果。

## 参数

以下均通过 `PinocchioConfig` 传入，无需修改源码或训练。

| 参数 | 默认 | 调整意义 |
| --- | --- | --- |
| `weights` | `(1,1,1,.3,.3,.3)` | 前三项为平移、后三项为旋转；调大后三项更重视朝向。平方代价中的姿态系数实际为 0.09 |
| `damping` | `1e-7` | 阻尼底值；调大减小迭代幅度，奇异附近更保守，也可能更慢 |
| `singularity_threshold` | `0.01` | 低于该奇异值时增强阻尼；调大更早介入，取决于权重和单位 |
| `seed_weight` | `1e-8` | 偏向初始 seed；调大减少远离 seed 的倾向，但可能牺牲精度 |
| `max_joint_step` | `0.2 rad` | 每次内部迭代增量上限，不是速度；调大更激进 |
| `max_iterations` | `40` | 增大给求解更多机会，也增加耗时 |
| `position_tolerance` | `1e-4 m` | 位置停止／验收阈值 |
| `orientation_tolerance` | `1e-3 rad` | 姿态停止／验收阈值，约 0.057° |

容差是求解标准，不是保证精度。目标不可达、关节限位或局部极小都可能返回 `converged=False`。连续输入使用上一拍关节解 warm start；不在库内隐藏保存该状态。

## 评估

```bash
python training/evaluate.py --ik pinocchio --samples 256 \
  --out artifacts/pin.json
```

评估为随机 FK 可达目标和随机初值，无多起点搜索。接近目标的连续跟踪与随机初值求解属于不同难度，不应混用成功率。
