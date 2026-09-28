# 三阶末端位姿滤波器

[README](../README.md) · [实现](../piper_ik/filters.py)

## 接口

```python
from piper_ik import ThirdOrderPoseFilter

f = ThirdOrderPoseFilter(initial_pose, tau=0.06)
reference = f.update(target_pose, dt=0.02)
current = f.value             # 返回副本
f.reset(new_initial_pose)     # 重置三层状态
```

输入、输出均为同一固定坐标系中的绝对 4×4 位姿，平移单位 m。`dt` 与 `tau` 单位秒，必须为有限正数。对象保留状态，每条数据流使用一个实例；不能每一拍重新创建。

该类可直接导入使用，不导入 Pinocchio 或 Torch。可接任意 IK 或单独处理轨迹。

## 算法

三个一阶低通滤波器串联，每次按层更新：第一层输入为目标，后两层输入为前一层本次输出。

```text
α = 1 − exp(−dt / tau)
p_i ← p_i + α (p_source − p_i)
r = Log_SO3(R_source @ R_i.T)
R_i ← Exp_SO3(α r) @ R_i
```

最终返回第三层。旋转沿相对旋转的最短路径过渡，不逐元素插值旋转矩阵，也不直接插值欧拉角。恰好 180° 时最短旋转轴存在歧义，不能保证跨该点的轴选择连续。

这是“三个一阶滤波器串联”的三阶低通，不是三次样条，也不等于 jerk 限制器。

## 参数与延迟

| 参数 | 默认 | 意义 |
| --- | --- | --- |
| `tau` | 0.06 s | 每层时间常数；越大越平滑、滞后越大 |
| `linear_speed` | None | 可选，每层位置更新的速度上限，m/s |
| `angular_speed` | None | 可选，每层旋转更新的角速度上限，rad/s |
| `dt` | 调用时提供 | 实际采样间隔；例如 50 Hz 时约 0.02 s |

无速度限幅时，平移的连续时间近似传递函数为 `1/(1+tau*s)^3`。低频滞后近似 `3*tau`，默认约 0.18 s；离散采样实现的实际滞后还取决于 dt。单轴阶跃约在 `6.3*tau` 达到 95%，这些是连续时间近似，不是硬延迟。

可选速度上限限制的是每层末端参考变化，不保证关节速度、加速度或执行器动态。要减少延迟，可降低 tau，再检查噪声和追踪误差；没有一个适用于所有输入频率的最优 tau。

## 与 IK 组合

完整可运行示例：[filter_target.py](../examples/filter_target.py)。

```python
reference = f.update(target, dt)
result = ik.solve(reference, seed=q_previous)
if result.converged:
    q_previous = result.q
```

滤波和 IK 独立：`PiperIK.solve()` 不隐式滤波。切换目标流、重置关节状态或重新开始时，调用 `reset(ik.fk(q))`，避免沿用旧参考。
