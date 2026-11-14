# Piper IK

Piper 六轴机械臂的 Python 逆运动学库。统一接口切换 **Pinocchio / RL（PPO）**，支持自定义求解器，并提供独立的 **三阶末端位姿滤波器**。

不依赖 ROS。只计算位姿和关节解，不包含网页、设备通信或运动执行。

## 安装

Ubuntu，Python 3.10–3.12；本次测试使用 Python 3.12。

```bash
git clone https://github.com/kelo020304/piper-IK.git
cd piper-IK
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .         # Pinocchio + 滤波器
# 需要 RL 时安装：
pip install -e '.[rl]'
```

模型和已有 RL 权重随包安装，不依赖当前工作目录。尚未发布到 PyPI，不要直接用 `pip install piper-ik` 查找同名包。

## 快速使用

```python
import numpy as np
from piper_ik import PiperIK

ik = PiperIK(backend="pinocchio")  # 换为 "rl" 即可切换
q = np.array([0., 1., -1.2, .2, .55, 0.])

target = ik.fk(q)                 # 当前 TCP 的 4×4 位姿
target[0, 3] += 0.005             # 沿基座 X 轴移动 5 mm
result = ik.solve(target, seed=q)

if result.converged:
    q_next = result.q             # 六个绝对关节角，rad
else:
    print(result.position_error, result.orientation_error)
```

`solve()` 不执行运动、不自动滤波，也不隐式记住上一次解。连续目标可用上一拍 `result.q` 作为下一拍 `seed`。实例内部有 FK 工作缓存，请勿跨线程共享。

可直接运行：

```bash
python examples/solve.py --backend pinocchio
python examples/solve.py --backend rl
python examples/filter_target.py
```

## 接口与坐标约定

| 接口 / 数据 | 定义 |
| --- | --- |
| `PiperIK(backend="pinocchio", tcp=None)` | 选择求解器；默认 TCP 为模型 `tool0`（与 link6 原点重合） |
| `fk(q, check_limits=True) → ndarray(4,4)` | TCP 在机器人基座中的绝对位姿；仅计算实测反馈时可设 `check_limits=False`，保留限位外原始反馈，不裁剪。IK 的关节限位校验不变。 |
| `solve(target, seed) → IKResult` | 目标为同一基座中的绝对 TCP 位姿；seed 为六个绝对关节角 |
| `result.q` | 六个绝对关节角，rad；**不是关节增量** |
| `result.converged` | TCP 位置和姿态残差均满足所选后端的容差 |
| `result.position_error` / `orientation_error` | 实际返回解的 TCP 误差，单位 m / rad |
| `result.seconds` | 本次求解及误差计算耗时，秒 |
| `joint_limits` | `(lower, upper)`，六轴限位，rad |
| `tcp` | 固定的 tool0 → TCP 变换；`T_base_tcp = T_base_tool0 @ T_tool0_tcp` |

位姿统一使用 `T = [[R, p], [0, 0, 0, 1]]`，其中 p 单位为 m，R 为旋转矩阵。输入无效会抛出异常；目标不可达或迭代未收敛则返回 `converged=False`，由调用者处理。

上游如果给的是增量，先转换成绝对目标：基座平移 `p_next = p_goal + Δp`；工具局部平移 `p_next = p_goal + R_goal @ Δp`。基座旋转左乘 `R_next = ΔR @ R_goal`，工具局部旋转右乘 `R_next = R_goal @ ΔR`。

## 三阶滤波

```python
from piper_ik import ThirdOrderPoseFilter

pose_filter = ThirdOrderPoseFilter(ik.fk(q), tau=0.06)

# 在调用者自己的循环中执行；dt 是本次时间间隔，单位秒
reference = pose_filter.update(target, dt=0.02)
result = ik.solve(reference, seed=q)

# 输入流重新开始时清空历史
pose_filter.reset(ik.fk(q))
```

三个一阶滤波器串联，平移线性更新，旋转沿 SO(3) 最短路径更新。`tau` 是**每层**时间常数：增大更平滑，也增加延迟。该类不依赖 IK 后端，不生成带关节加速度/jerk 约束的轨迹。详见 [滤波器](docs/FILTER.md)。

## 配置与扩展

```python
from piper_ik import PinocchioConfig, RLConfig

ik = PiperIK("pinocchio", pinocchio=PinocchioConfig(
    weights=(1, 1, 1, .3, .3, .3),
    max_iterations=40,
    position_tolerance=1e-4,       # m
    orientation_tolerance=1e-3,    # rad
))

ik_rl = PiperIK("rl", rl=RLConfig(
    max_iterations=20,
    time_budget_s=0.008,          # 软预算；None 表示仅限制迭代次数
))
```

自定义后端只需实现 `solve(target, seed) → q`：

```python
class MyBackend:
    def solve(self, target, seed):
        # target：基座中的绝对 tool0 位姿，SDK 已去除 TCP 偏移
        # 返回六个绝对关节角；替换为你自己的算法
        return seed.copy()

ik = PiperIK(backend=MyBackend())
```

SDK 校验返回关节并重新计算 TCP 残差；上例只是接口演示，不会求解新目标。自定义后端使用 `PinocchioConfig` 中的容差作为结果验收标准。

## 文档

- [Pinocchio：代价函数与配置](docs/PINOCCHIO.md)
- [RL：观测、reward、训练与权重](docs/RL.md)
- [三阶滤波：公式与参数](docs/FILTER.md)
- [HTML 阅读版](README.html)

## 目录

```text
piper_ik/    公共 SDK、滤波器、内部求解器、模型与权重
examples/    单目标求解、滤波与 IK 组合
training/    可选 PPO 训练和评估
docs/        参数文档（Markdown + HTML）
tests/       接口、坐标、滤波与模型一致性测试
```

基础调用仅需要 `piper_ik/` 安装包。RL 权重是运动学 IK 策略，不是视觉任务策略；没有碰撞检查、速度规划或动力学保证。

## 开发

```bash
pip install -e '.[rl,dev]'
pytest -q
python tools/build_docs.py
```

已验证：29 项测试、wheel 安装后从仓库外导入、两种 IK 示例、滤波组合示例及 CUDA 短训练。短训练仅检查链路，不代表收敛。

[MIT License](LICENSE) · [模型与权重来源](THIRD_PARTY.md)
