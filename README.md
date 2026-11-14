# piper_ik：从 URDF 出发的机械臂 IK 与 200 Hz 执行链路

`piper_ik` 把 VLA 输出的 FastUMI 末端动作变成机械臂关节指令。包里有三部分：

- **通用 IK（`ArmIK`）**：读任意固定基座串联臂的 URDF，自动找出臂关节、IK 坐标系和 TCP。求解器是线性化后的有界阻尼最小二乘，基于 Pinocchio（C++）。
- **执行链路**：每 5 ms 一个控制周期：按时间插值出目标（位置线性、姿态 Slerp），以上一周期的解为种子做 IK，然后把解原样发出。节奏用“追点”：跟不上就等。
- **实时控制进程（`RealtimeArm`）**：驱动、收发和上面的控制周期放在一个独立进程里。策略进程只取观测、送动作块。驱动可以替换（`Driver`）。

另有一个独立的三阶位姿滤波 `ThirdOrderPoseFilter`，给执行链路以外的位姿流用（第 10 节）。执行链路本身不滤波。

默认机型是 Piper + UMI 夹爪，模型随包附带（`PiperIK`）。Piper 的关节指令与作者的 data_viewer 真机回放控制台逐位相同。1.1.0 版加了通用层，Piper 的 IK 解和每个周期的指令与 1.0.0 逐位相同，有回归测试。从 piper-IK 0.2.0 升级见 1.1 节。

**换一台机械臂**：准备 URDF，写一行 `ArmIK('arm.urdf', tcp_frame=...)`。上真机时再写一个驱动类。见第 3 节。

先读 `AGENTS.md` 里的硬规则。

---

## 目录 (Table of Contents)

1. [安装 (Installation)](#1-安装-installation)
2. [Piper 快速上手 (Quick Start on the Piper)](#2-piper-快速上手-quick-start-on-the-piper)
3. [换一台机械臂 (Another Arm)](#3-换一台机械臂-another-arm)
4. [IK 求解器 (Solver)](#4-ik-求解器-solver)
5. [数据约定 (FastUMI Convention)](#5-数据约定-fastumi-convention)
6. [接口一览 (API)](#6-接口一览-api)
7. [200 Hz 控制进程 (Control Process)](#7-200-hz-控制进程-control-process)
8. [追点 (Pacing)](#8-追点-pacing)
9. [自检 (Self-Check)](#9-自检-self-check)
10. [可选工具：三阶位姿滤波 (Optional Pose Filter)](#10-可选工具三阶位姿滤波-optional-pose-filter)
11. [包里没有、也不要加的 (Not Included)](#11-包里没有也不要加的-not-included)
12. [文件结构 (Files)](#12-文件结构-files)

---

## 1. 安装 (Installation)

仓库的根目录就是包本身（`core.py`、`model/` 等直接在根目录）。

| 方式 | 做法 |
| :--- | :--- |
| 放文件夹 | `git clone https://github.com/kelo020304/piper-IK.git piper_ik`（文件夹必须叫 `piper_ik`），让它的上一级目录在 `sys.path` 里，然后 `import piper_ik` |
| pip | `pip install git+https://github.com/kelo020304/piper-IK.git`；本地文件夹用 `pip install /path/to/piper_ik`，开发时 `pip install -e /path/to/piper_ik` |

**依赖**：Python ≥ 3.9、numpy、scipy、`pin==3.7.0`（pip 版 Pinocchio，与控制台同一版本；pip 安装本包时自动装上）。Piper 真机另装 `piper_sdk==0.6.1`。已在 Python 3.12 上测过。

**验证**：

```bash
python -c "import piper_ik; print(piper_ik.__version__, piper_ik.PiperIK().joint_names)"
# 1.1.0 ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
```

> **注意**
> - **装了 ROS 的机器直接 `import piper_ik`。** ROS noetic 的 `libeigenpy.so` 排在 `LD_LIBRARY_PATH` 前面，会让 Pinocchio 导入失败（`undefined symbol: _Py_tracemalloc_config`）。本包导入 Pinocchio 之前，先按绝对路径加载 pip 版自带的 `libeigenpy.so`。不要自己设 `LD_LIBRARY_PATH` / `LD_PRELOAD`。在这类机器上直接 `import pinocchio` 仍会失败，先 `import piper_ik` 就没有问题。
> - **一个 Python 环境只装一个版本。** piper-IK 0.2.0 和本版的包名都是 `piper_ik`。
> - **不要 `pip install pinocchio`。** PyPI 上叫 `pinocchio` 的是一个无关的包；Pinocchio 的 pip 包名是 `pin`。

### 1.1 从 piper-IK 0.2.0 升级

0.2.0 的代码在 git 标签 `v0.2.0` 上：`git checkout v0.2.0`，或 `pip install git+https://github.com/kelo020304/piper-IK.git@v0.2.0`。

| 0.2.0 | 1.1 |
| :--- | :--- |
| `PiperIK('pinocchio')`、`PiperIK(config=PinocchioConfig(...))` | `PiperIK()`、`PiperIK(config=SolverConfig(...))`。求解器和默认参数相同。第一个位置参数现在是关节限位 |
| `PiperIK('rl')`、`RLConfig`、`training/`、`ppo_precision.pt` | 已删除。PPO 策略只在 Piper 上训练过，换 URDF 就不能用 |
| `PiperIK(backend=自定义 IKBackend)` | 已删除。换机械臂用 `ArmIK(urdf, ...)`（第 3 节） |
| `ik.fk(q)`、`ik.fk(q, check_limits=False)` | `ik.fk(q)`，不检查限位（实测关节可能略超限位） |
| `ik.solve(target, seed)` | 相同：基座系 TCP 目标 4×4，种子在限位内 |
| `ThirdOrderPoseFilter` | 相同，代码未改（第 10 节） |
| `piper_ik/assets/piper.urdf`、`piper.xml` | `model/piper_umi.urdf`：关节 1–6 的原点和轴与 0.2.0 的 `piper.urdf` 相同，限位换成真实限位（例如 J1 上限 124° → 150°）；MuJoCo 模型在 `model/piper_umi_models/xml/`（2.3 节） |

---

## 2. Piper 快速上手 (Quick Start on the Piper)

### 2.1 三件事

1. 按第 1 节装好。
2. 复制 `examples/vla_loop.py`，只把 `Policy.infer()` 和 `grab_images()` 换成你的模型和相机。
3. 跑第 9 节的自检，数字对上再上真机。

### 2.2 最小用法

完整版见 `examples/vla_loop.py`，结构不要改。

```python
from piper_ik import RealtimeArm

with RealtimeArm('can0') as arm:                     # 连接、使能（本机真实关节限位）、在当前姿态启动 = FastUMI 第一帧
    while running:
        images = grab_images()
        state, t_obs = arm.observation()             # 与相机同一时刻
        arm.push_actions(policy.infer(images, state), t_obs)       # (H, 7) 绝对动作 + 同一个 t_obs
        arm.wait_until(t_obs + REPLAN_STEPS / arm.fps)              # 执行 REPLAN_STEPS 帧后再推理
# 退出 with 后控制进程停止，机械臂保持最后一条指令，仍处于使能状态
```

`RealtimeArm('sim')` 用同一个控制进程驱动 `sim.py` 里的模拟 Piper。没有硬件时先用它跑通。

### 2.3 Piper 模型文件（`model/`）

| 文件 | 是什么 | 谁在用 |
| :--- | :--- | :--- |
| `model/piper_umi_models/` | 完整的 Piper + UMI 夹爪模型包，原样拷贝：URDF（`urdf/`）、MuJoCo（`xml/piper_umi.xml`、`scene.xml`）、USD（`usd/`），含全部显示 / 碰撞网格和许可证；`SHA256SUMS` 列出每个文件的校验和 | 控制台的仿真和 TCP 来源是其中的 MuJoCo 模型；USD 给 Isaac Sim |
| `model/piper_umi.urdf` | IK 版 URDF：在上面的 URDF 基础上，关节 1–6 换成控制台求 IK 的数值，限位换成真实限位，`tool0` 换成指尖 TCP；连杆、网格、手指不变，网格直接引用 `piper_umi_models/urdf/meshes/` | `PiperIK` 读它；规划器（Pinocchio、MoveIt、PyBullet、RViz）可直接用它做碰撞检测和显示 |
| `model/tcp.json` | 法兰（`link6`）→ 指尖 TCP 的 4×4 矩阵，控制台连接时由 `piper_umi.xml` 算出的原值 | IK 直接用；`piper_umi.urdf` 的 `tool0` 也由它生成 |

IK 版 URDF 里的数值：

| 内容 | 来源 |
| :--- | :--- |
| 关节 1–6 的几何 | 控制台求 IK 用的 piper-IK URDF（与控制器自己的 FK 一致到 0.1 mm）；与原模型包的 URDF 相差约 0.1 mm |
| 关节限位 | 规格表 ∩ piper_sdk 关节表：J1 ±150°、J2 0–179.9°、J3 −170–0°、J4 ±100°、J5 ±70°、J6 ±120°。使能后再与控制器固件读数取交集（`PiperArm.enable()` 返回） |
| `tool0` | UMI 指尖 TCP，在法兰前 217.1 mm；与原模型包 URDF 的 `tool0` 相差 0.03 mm |

TCP 以 `model/tcp.json` 的矩阵为准。URDF 只能写成 xyz / rpy，还原回矩阵差 8e-16，物理上等于 0；但 IK 和追点里的阈值判断偶尔会被这一点舍入翻转。所以从 URDF 算出的 TCP 与这个矩阵相差 < 1e-12 时，IK 用原矩阵，才能和控制台逐位一致。`model/` 下的文件不要改（`AGENTS.md` 第 8 条）。

---

## 3. 换一台机械臂 (Another Arm)

需要准备的东西：

| 要准备的 | 作用 | 只做 FK / IK | 离线跑 FastUMI 动作 | 上真机 |
| :--- | :--- | :---: | :---: | :---: |
| URDF（带 TCP 坐标系） | 运动学和关节限位 | 需要 | 需要 | 需要 |
| `tcp_axes`（3.3） | FastUMI 夹爪系与 TCP 坐标轴的对应 | — | 需要 | 需要 |
| 驱动类 `Driver`（3.6） | 读关节、发指令 | — | — | 需要 |

不需要改本包的任何文件。新机械臂的 URDF 放在你自己的目录里。

### 3.1 URDF 要满足的条件

| 项 | 要求 | 不满足时怎么办 |
| :--- | :--- | :--- |
| 基座 | 固定基座：URDF 的根连杆不动 | 移动底盘先单独拆出机械臂部分的 URDF |
| 臂关节 | `revolute` 或 `prismatic`，带 `<limit lower=... upper=...>`，lower < upper | `continuous` 关节改成 `revolute`，写真实限位（本包不支持 `continuous`、球关节和浮动关节，加载时报错） |
| TCP 坐标系 | 一个固定在最后一个臂连杆上的坐标系，默认名 `tool0` | 加一个 fixed 关节，见下面的例子；或用 `tcp=` 直接给 4×4 矩阵 |
| 其他关节 | 手指等不在“根 → TCP”链上的关节自动锁在 0 | — |
| 网格 | IK 不读网格，可以没有 | — |
| 单位 | 米、弧度（URDF 标准） | — |

没有 TCP 坐标系时，在 URDF 里加：

```xml
<link name="tool0"/>
<joint name="tool0_joint" type="fixed">
  <parent link="link6"/>                    <!-- 最后一个臂连杆（法兰） -->
  <child link="tool0"/>
  <origin xyz="0 0 0.217" rpy="0 0 0"/>     <!-- 法兰 → 指尖；建议 z 朝夹爪前方，y 沿手指开合方向 -->
</joint>
```

### 3.2 一行代码

```python
from piper_ik import ArmIK

ik = ArmIK('my_arm/robot.urdf', tcp_frame='tool0')
print(ik.dof, ik.joint_names, ik.ik_frame)      # 自动找到的臂关节和 IK 坐标系
T = ik.fk(q)                                     # 基座系 TCP 位姿，4×4，米
r = ik.solve(T_target, seed)                     # r.q、r.converged、r.position_error（m）、r.orientation_error（rad）
```

`ArmIK` 的执行链路接口（`start`、`observation`、`push_actions`、`tick`）与 `PiperIK` 完全相同，见第 6 节。

| 参数 | 默认 | 含义 |
| :--- | :--- | :--- |
| `urdf` | 必填 | URDF 路径 |
| `tcp_frame` | `'tool0'` | TCP 坐标系名。`fk()` 返回、`solve()` 接收的位姿都是它的 |
| `ik_frame` | 自动：带 TCP 的那个关节的子连杆 | 求解器内部对齐的坐标系（Piper：`link6`）。目标先右乘 TCP 的逆，再在这个坐标系上求解 |
| `joints` | 自动：从根到 `tcp_frame` 的链上的可动关节 | 指定臂关节名的列表；其余关节锁在 0 |
| `tcp` | 自动：URDF 里 `ik_frame` → `tcp_frame` 的变换 | 4×4 矩阵，代替 URDF 里的 TCP |
| `tcp_axes` | `('down', 'left', 'forward')`（Piper） | TCP 坐标轴在 FastUMI 夹爪系里的方向，见 3.3 |
| `limits` | URDF 里的限位 | `(lo, hi)`，弧度 / 米。真机用驱动读到的限位 |
| `gripper_zero_m` / `gripper_one_m` | 0 / 0.07 | 夹爪值 0 / 1 对应的开度（米） |
| `fps` | 20 | 策略 / 数据集帧率 |
| `follow` | `'wait'` | `'wait'` 为追点；`'strict'` 为原速 1× |
| `config` | `SolverConfig()` | 求解参数，不要改（第 4 节） |

关节顺序是 URDF 里的顺序（`ik.joint_names`）。`fk`、`solve`、`tick` 和驱动收发都按这个顺序。

### 3.3 `tcp_axes`：TCP 的坐标轴对应夹爪的哪个方向

FastUMI 动作写在“回合第一帧的夹爪系”里：x 朝前（夹爪朝向）、y 朝左、z 朝上。IK 需要知道 TCP 的 x、y、z 轴在这个夹爪系里各指向哪边。

用三个词写，依次是 TCP 的 x、y、z 轴，从 `forward` / `back` / `left` / `right` / `up` / `down` 里选，必须是右手系（x × y = z），否则报错。也可以给 3×3 矩阵，列是 TCP 的三个轴在夹爪系里的方向。

| TCP 的约定 | `tcp_axes` |
| :--- | :--- |
| z 朝夹爪前方，y 沿手指开合方向朝左（Piper `tool0`；ROS 里常见的 `tool0` 也是 z 朝前） | `('down', 'left', 'forward')`（默认） |
| x 朝前，z 朝上 | `('forward', 'left', 'up')` |

**确认方法**：让机械臂（或仿真）摆一个夹爪水平、指向基座 +x 的姿态，把这组关节角传给自检工具的 `--q0`（3.4）。工具打印 TCP 各轴在基座系里指向哪边，并给出对应的 `tcp_axes`。

`tcp_axes` 写错时，IK 本身照样收敛，但策略的“向前”会变成别的方向，夹爪姿态也会错。

### 3.4 新 URDF 先跑自检工具

```bash
# 在 piper_ik/ 的上一级目录运行
python -m piper_ik.check my_arm/robot.urdf --tcp-frame tool0 --q0 <夹爪水平朝前的关节角，逗号分隔>
```

例：包里的 7 关节测试臂（`tests/data/arm7.urdf`，关节名 a1–a7，带一个手指关节，TCP 名为 `tcp`）：

```text
URDF        piper_ik/tests/data/arm7.urdf
arm joints  7:
  a1                       revolute    -166.003 ..   166.003 deg
  ...
  a7                       revolute    -166.003 ..   166.003 deg
locked      ['finger_joint'] (held at 0)
IK frame    l7
TCP         tcp: 210.4 mm from l7 (xyz 0.0 0.0 210.4 mm)
at q0       q0 = [0.779, -0.733, -0.78, -2.615, -2.897, 2.585, 0.082]; TCP at [0.5, 0.0, 0.45] m; TCP axes in the base frame: x -z (0 deg off), y +y (0 deg off), z +x (0 deg off)
            tcp_axes = down,left,forward if the gripper at q0 is level and points along base +x (otherwise pick such a q0 with --q0)
tcp_axes    in use: TCP x down, y left, z forward on a level gripper (FastUMI convention)
IK near seeds (0.05 rad)  converged  98.0%  position p95 0.0672 mm, max 0.133 mm  rotation p95 0.0052 deg  0.20 ms/solve
IK far seeds (0.6 rad)    converged  84.0%  position p95 0.1492 mm, max 37.333 mm  rotation p95 0.0288 deg  1.04 ms/solve
OK
```

逐行检查：

| 行 | 检查什么 |
| :--- | :--- |
| `arm joints` | 关节数、名字、顺序（= 驱动收发的顺序）、限位都与真机一致 |
| `locked` | 只有手指等非臂关节 |
| `IK frame` / `TCP` | TCP 到法兰的距离与实物一致 |
| `at q0` / `tcp_axes` | 按 3.3 确认 `tcp_axes` |
| `IK near seeds` | 收敛 ≥ 95%。200 Hz 时种子是上一周期的解，离答案很近，这一行代表实际用法。低于 95% 时打印 `FAIL`，退出码 1 |
| `IK far seeds` | 只作参考。种子离答案 0.6 rad 时会落到别的解支或停在限位上，收敛率低于近种子属正常 |

Piper 的结果（`python -m piper_ik.check piper_ik/model/piper_umi.urdf`）：近种子收敛 96.5%，位置 p95 0.09 mm，每次 0.2 ms。没收敛的几个是线搜索不再下降时停下，残差略高于 0.1 mm（最大 0.19 mm），与控制台相同。

### 3.5 先在仿真里跑通

```python
from piper_ik import RealtimeArm, SimDriver

robot = dict(urdf='my_arm/robot.urdf', tcp_frame='tool0', tcp_axes=('down', 'left', 'forward'))
with RealtimeArm(robot=robot, driver=SimDriver(q_start)) as arm:      # q_start：起始关节角
    state, t_obs = arm.observation()
    arm.push_actions(actions, t_obs)                                    # 与 Piper 的策略循环完全相同
    arm.wait_until(t_obs + 10 / arm.fps)
```

- `robot` 是 `ArmIK` 的关键字参数，`urdf` 必填。控制进程和策略进程各用它建一个相同的 `ArmIK`。
- `SimDriver` 在控制进程里模拟机械臂：每个关节按加速度上限各自追目标（与 Piper 驱动器的行为相同），关节数等于 `q_start` 的长度。
- 不是 6 关节的机械臂必须传 `driver`。默认驱动是 Piper 的 CAN 驱动，不传时报错。

### 3.6 接真机：写一个驱动

控制进程只通过下面四个方法碰机械臂：

| 方法 | 何时调用 | 要做的事 |
| :--- | :--- | :--- |
| `open()` | 控制进程启动时一次 | 连接、使能（机械臂静止，不运动）。返回 `(limits, q0, width0)`：IK 用的关节限位 `(lo, hi)`（弧度；`None` 表示用 URDF 的）、当前关节角、当前夹爪开度（米） |
| `read()` | 每个周期 | 返回已收到的最新反馈 `(关节角, 夹爪开度 m)`，不等待 |
| `send(q, width)` | 每个周期 | 把这一周期的指令原样发出：不滤波、不限幅、不插值 |
| `close()` | 控制进程退出时 | 释放硬件。机械臂保持最后一条指令 |

```python
from piper_ik import Driver, RealtimeArm

class MyArmDriver(Driver):                         # 写在模块顶层：驱动对象要传给控制进程（spawn），必须能 pickle
    def __init__(self, port):
        self.port = port; self.bus = None          # 只存参数，硬件在 open() 里打开

    def open(self):
        self.bus = my_sdk.connect(self.port); self.bus.enable()
        return None, self.bus.joints(), self.bus.gripper_width()

    def read(self):
        return self.bus.joints(), self.bus.gripper_width()

    def send(self, q, width):
        self.bus.move_joints(q); self.bus.move_gripper(width)

    def close(self):
        if self.bus is not None:
            self.bus.close()

robot = dict(urdf='my_arm/robot.urdf', tcp_frame='tool0')
with RealtimeArm(robot=robot, driver=MyArmDriver('can1'), control_hz=200) as arm:
    ...                                            # 策略循环与 examples/vla_loop.py 相同
```

- 关节顺序与 `ik.joint_names` 相同。单位：转动关节弧度，移动关节米。
- `read()` 和 `send()` 每 5 ms 调一次，每次要在 1 ms 内返回。
- 可选属性 `dropped` / `received`（发送丢弃数、接收帧数）会进入 `RealtimeArm` 的统计。
- Piper 的驱动 `PiperDriver`（`realtime.py`）是一个完整的参考实现。

### 3.7 仍按 Piper 定的部分

| 项 | 现在的值 | 换机械臂时 |
| :--- | :--- | :--- |
| 控制频率 | 200 Hz（Piper 关节反馈的频率） | Piper 保持 200 Hz。其他机械臂用 `RealtimeArm(control_hz=...)` 设成它的反馈频率 |
| 追点阈值（`follow.py`） | 领先 0.25° 全速、1° 停；抓放附近 0.1° / 0.4°。在 Piper 驱动器上验证 | 先用 `follow='strict'`。要改阈值先问用户（`AGENTS.md` 第 9 条）。移动关节的领先量单位是米，阈值不适用 |
| 求解参数（`SolverConfig`） | TCP 处 0.1 mm / 0.001 rad，最多 40 次迭代，每步每个关节 ≤ 0.2 | 与机型无关，不改 |
| 夹爪 | 0–1 对应 0–70 mm，力矩 1000（UMI 夹爪） | `gripper_zero_m` / `gripper_one_m`；力矩由驱动自己处理 |
| Piper 专用 | `hardware_limits()`、`PiperArm`、`PiperCAN`、`PiperDriver`、`SimBus`、`RealtimeArm('can0')` / `RealtimeArm('sim')` | 用自己的 `Driver` 和 `SimDriver` |

---

## 4. IK 求解器 (Solver)

IK 是线性化后的有界阻尼最小二乘：每次迭代在当前关节角处把运动学线性化，然后解一个带关节边界的线性最小二乘问题，得到这一步的关节增量。代码是 piper-IK 0.2.0 Pinocchio 后端的原样拷贝（`kinematics.py` 的 `solve_flange()`），控制台用的就是它。

每次 `solve(T_target, seed)`：

1. TCP 目标右乘 TCP 的逆，得到 IK 坐标系的目标 T*。
2. 从 `seed` 开始迭代，最多 40 次：
   1. 在当前 q 处算 IK 坐标系的位姿 T(q) 和 6 维误差 e = log6(T(q)⁻¹·T*)。误差进入容差（0.1 mm、0.001 rad）就停。
   2. 算误差对关节的雅可比 J（Pinocchio `computeFrameJacobian` 与 `Jlog6`）。
   3. 解有界线性最小二乘，得步长 dq：

      min ‖ [W·J ; r·I]·dq − [−W·e ; −s·(q − q_seed)/r] ‖²，约束 max(lower − q, −0.2) ≤ dq ≤ min(upper − q, 0.2)

   4. 回溯线搜索：步长依次试 1、1/2、1/4、1/8、1/16，取第一个让加权误差下降的。都不下降就停。
3. 返回最后的 q，在 TCP 处重算残差，`converged` 按容差判断。

| 符号 | 值 | 作用 |
| :--- | :--- | :--- |
| W | diag(1, 1, 1, 0.3, 0.3, 0.3) | 位置权重 1，姿态权重 0.3 |
| r² | damping + s，damping = 1e-7 + max(0, 0.01 − σ_min)² | 阻尼；J 的最小奇异值 σ_min < 0.01（接近奇异）时加大 |
| s | 1e-8 | 把解拉向种子的权重 |
| 0.2 | rad（移动关节为 m） | 一次迭代里每个关节最多走的量 |
| I | n×n 单位阵 | n = 臂关节数（Piper 6，测试臂 7） |

- 有界线性最小二乘用 `bvls()`：scipy `lsq_linear(method='bvls')` 的精简拷贝，NumPy 调用顺序相同，结果逐位相同，速度约快 2 倍。
- 关节限位是每一步的硬约束，所以解永远在限位内。够不到的目标返回限位内的最近解，`converged=False`，残差在结果里。
- 冗余臂（7 关节及以上）同样可用：阻尼项 r²·‖dq‖² 让每一步在多个解里取关节运动最小的那个。
- 速度：种子离答案 0.05 rad 时，Piper 和 7 关节测试臂每次求解约 0.2 ms（3.4）。200 Hz 执行时种子是上一周期的解，离答案更近。

---

## 5. 数据约定 (FastUMI Convention)

与数据集一字不差，不要改。

| 项 | 约定 |
| :--- | :--- |
| 每行 7 个数 | `[x, y, z, rx, ry, rz, g]`，与 FastUMI-100K 的 `action` / `observation.state` 列相同 |
| 坐标系 | 本回合第一帧的夹爪系：x 前（夹爪朝向）、y 左、z 上；第一帧全 0。与 TCP 坐标轴的对应由 `tcp_axes` 给出（3.3） |
| x y z | 米；指尖 TCP 的位置，不是 T265，不加任何偏移 |
| rx ry rz | 弧度；scipy `Rotation.from_euler('xyz', ...)`，±π 包裹（与数据集相同） |
| g | 夹爪开度 0–1：0 = 合拢（`gripper_zero_m`，默认 0 mm），1 = 张开（`gripper_one_m`，默认 70 mm）。超出 0–1 时按端点发 |
| 绝对还是增量 | 绝对位姿（相对第一帧）。模型若按增量训练，先用它自己的 output transform 还原成绝对值，并且反归一化 |
| chunk 时间 | 一次推理输出 `(H, 7)`。第 k 个动作是观测时刻之后 `(k+1)/20` s 的位姿（数据 20 fps，`action[i] = state[i+1]`）；帧率不同就改 `fps` |
| 观测 state | 用 `arm.observation()` 给出，不要自己从关节换算 |

“FastUMI 第一帧”就是 `RealtimeArm` 启动时机械臂指尖的位姿。之后的动作都相对它。

---

## 6. 接口一览 (API)

| 调用 | 何时 | 说明 |
| :--- | :--- | :--- |
| `RealtimeArm(can='can0', follow='wait')` | Piper：每条手臂、每个回合一个 | `with` 进入时启动控制进程；`can='sim'` 为模拟 Piper；`follow='strict'` 为严格原速 1× |
| `RealtimeArm(robot=dict(urdf=...), driver=MyDriver(...))` | 其他机械臂 | `robot` 为 `ArmIK` 的关键字参数；`driver` 见 3.6；`control_hz` 设成该臂的反馈频率 |
| `arm.observation()` | 每次推理前 | 返回 `(state(7,), t_obs)`；`t_obs` 是这次观测的数据时间 |
| `arm.push_actions(actions, t_obs)` | 每次推理后 | `actions` 为 `(H, 7)`；从 `t_obs` 往后的旧计划全部被新 chunk 替换 |
| `arm.wait_until(t)` | 策略循环 | 阻塞到数据时间到 `t`（或已到最后一个动作点） |
| `arm.last_ik` / `arm.finished` / `arm.source_time` / `arm.command` / `arm.joints` | 任意时刻读 | 上一周期 IK 残差；是否已到最后一个动作点；当前数据时间；上一周期发出的指令；实测关节 |
| `arm.cycles` / `arm.late_cycles` | 任意时刻读 | 已跑的周期数；超过一个周期的周期数 |
| `ArmIK(urdf, limits, tcp_frame=..., ...)` | 底层 | 任意机械臂的模型、FK / IK 和执行链路（3.2） |
| `PiperIK(limits, follow='wait', fps=20)` | 底层 | `ArmIK` 加载随包的 Piper 模型；Piper 控制进程里用的就是它 |
| `ik.fk(q)` / `ik.solve(T, seed)` | 底层 | 单纯的正逆解，解始终在关节限位内 |
| `ik.start` / `observation` / `push_actions` / `tick(q, dt)` | 底层 | 单进程用法（`examples/dry_run.py`、与控制台的对比测试） |
| `Driver` / `PiperDriver` / `SimDriver` | 底层 | 控制进程的驱动接口；Piper 的 CAN 驱动；任意关节数的模拟臂 |
| `PiperArm` | 底层 | 与控制台相同的 piper_sdk 调用：`connect()`、`enable(speed_percent=100)`（返回 IK 用的真实限位）、`send_joints()`、`send_gripper()`、`disable()` |
| `python -m piper_ik.check <urdf>` | 换 URDF 后 | 打印从 URDF 读到的模型并测 IK（3.4） |

---

## 7. 200 Hz 控制进程 (Control Process)

`RealtimeArm` 启动一个独立进程（spawn 方式，和策略进程里的 CUDA、相机线程互不影响）。控制进程先建好 IK 模型，再调用驱动的 `open()`，然后按绝对时间表每个周期（200 Hz 时 5 ms）：

1. `driver.read()`：读已到的反馈（不阻塞）。
2. `ArmIK.tick(实测关节, 真实时间差)`。
3. `driver.send()`：把返回的关节和夹爪指令原样发出。

**Piper（`PiperDriver`）**：先用 piper_sdk 连接、使能、读固件限位，然后关掉 SDK（SDK 的读线程要在 Python 里解析总线上全部约 3400 帧/s），只留一个原始 SocketCAN 套接字，由内核过滤到关节和夹爪反馈（0x2A5–0x2A8，800 帧/s）。每个周期发 0x151（MOVE J，100%）、0x155–0x157（关节，0.001°）、0x159（夹爪，0.001 mm），与 `PiperArm.send_joints()` / `send_gripper()` 经 piper_sdk 发出的字节逐位相同（有测试逐字节比对）。

`push_actions()` 在策略进程里用一个完全相同的 `ArmIK`（同样的模型、限位、起点、动作块）拼出新轨迹，只把拼好的轨迹交给控制进程，下一周期生效。`observation()` 从共享内存读最新反馈和数据时间。

本机实测（模拟 Piper，整理餐具示教，策略每 0.5 s 推送一个 50 步动作块）：可达示教实际 200.0 Hz，周期最长 3.5 ms，0 个周期超时；部分超出工作空间的示教 0.57% 的周期超时（最长 9.1 ms，IK 在工作空间外多迭代几步）。原来的单进程循环同条件下是 195 / 194 Hz。CPU 默认省电调频时这颗核只跑约 1.1 GHz；设成 performance 调频后同样的周期短约 5 倍。

---

## 8. 追点 (Pacing)

Piper 驱动器对每个新的关节目标，各关节独立地按加速度上限去追。指令按原速发出时机械臂跟不上，各关节到位时间错开，指尖就离开原路线。控制台的做法是：路线上每一点都不变，只让数据时间按机械臂跟得上的速度前进。

- 上一条指令领先实测关节不超过 0.25° 时全速；领先达到 1° 时数据时间停下等。
- 抓、放时刻前后 0.3 s 内收紧到 0.1°–0.4°。
- 起点的静态下垂（重力）从领先量里扣掉。
- 机械臂停住、仍然追不上时，数据时间以 0.1× 慢慢前进。

控制台在拟合的驱动模型上验证过 5 条整理餐具轨迹（新臂）：原速时指尖最多离开路线 15–43 mm；追点时 95% 的时间在 1.0 mm 以内，最大 2.4 mm，抓放点约 1 mm，用时约原来的 4 倍。

用 VLA 时：数据时间变慢，chunk 就执行得慢，下一次推理的观测落在更晚的数据时间上。这是设计如此，不要补偿。chunk 切换时目标会跳一下，原样发出，驱动器自己去追，追点会自动放慢数据时间。

这些阈值按 Piper 驱动器定。其他机械臂见 3.7。

---

## 9. 自检 (Self-Check)

上真机前必须跑。在 `piper_ik/` 的上一级目录运行：

```bash
python -m pytest -q piper_ik/tests
python piper_ik/examples/dry_run.py
python piper_ik/examples/dry_run.py --follow strict
```

`tests` 共 57 个用例：

| 组 | 个数 | 检查什么 |
| :--- | :---: | :--- |
| 独立用例（`test_piper_ik.py`） | 15 | 正逆解、坐标转换、插值、chunk 时序、追点、线程、长时间运行（`push_actions` 只保留 `t_obs` 前 30 s 的轨迹，70 s 会话的指令与保留全部历史逐位相同） |
| 模型（`test_model.py`） | 3 | 模型包每个文件与 `SHA256SUMS` 一致；IK 版 URDF 引用的网格都在；`tcp.json` 与随包的 MuJoCo 模型算出的一致（没装 mujoco 时跳过） |
| `PiperArm`（`test_arm.py`，假 piper_sdk） | 6 | 调用顺序和单位与控制台相同：关 SDK 软件限位、MOVE J 100%、0.001°、夹爪先清错再使能、0.001 mm；已使能的机械臂不再收到使能或“恢复”指令 |
| 原始 CAN（`test_arm.py`） | 2 | 控制进程自己打包的 5 帧与 piper_sdk 发出的字节逐位相同（200 组随机指令）；反馈换算与 `PiperArm` 相同 |
| 模板实时运行（`test_vla_loop.py`） | 3 | 按真实时间跑 `examples/vla_loop.py` 的 `run()`（`RealtimeArm('sim')` + 随观测前进的假策略）：3 s 约 600 个周期、200.0 Hz、0 个超时、6 个 chunk，指尖沿策略方向走 8.8 cm 并合爪；策略抛错时循环结束、控制进程仍保持指令；控制进程启动失败时报错 |
| Piper 1.0.0 参考（`test_piper_reference.py`） | 3 | 120 次 IK 解、追点和原速各 1200 个周期的指令，与 1.0.0 录下的值逐位相同 |
| 通用机械臂（`test_generic_arm.py`） | 11 | 7 关节测试臂：自动找关节 / IK 坐标系 / TCP，IK 往返，手动指定参数，错误模型报错，`tcp_axes`，执行链路，`RealtimeArm` + `SimDriver` 200 Hz，自检工具；`ArmIK` 加载 Piper 模型与 `PiperIK` 逐位相同 |
| 位姿滤波（`test_filters.py`） | 11 | 三层级联、最短旋转弧、非法输入不改状态、速度上限；执行链路的模块都不导入 `filters.py` |
| 控制台对比（`test_console_equivalence.py`） | 3 | 在子进程里运行控制台本身（它的 piper-IK、`Runtime` 和拟合的驱动模型，整理餐具 ep247），逐周期比较：300 次 IK 差 0；追点回放 3190 个周期的关节指令、数据时间、夹爪指令差 0；原速回放 1255 个周期最大差 2.4e-15 rad（坐标转换的浮点舍入） |

预期结果：

- 有控制台和数据集的工作站（2026-10-15，当时 32 个用例）：全部通过。
- 没有控制台和数据集的机器：控制台对比的 3 个用例跳过；没装 mujoco 时 TCP 那条也跳过。2026-11-13 在开发机上：53 passed、4 skipped，约 15 s。
- `dry_run.py`（追点，200 Hz）：数据 8.0 s 用时 8.3 s（1.04×），IK 残差最大 0.096 mm，指尖离策略路线 p95 0.16 mm、最大 0.31 mm。
- `dry_run.py --follow strict`：用时 1.00×，p95 0.22 mm、最大 0.52 mm。

换了机械臂还要跑 `python -m piper_ik.check`（3.4），再用 `SimDriver` 跑通策略循环（3.5）。

---

## 10. 可选工具：三阶位姿滤波 (Optional Pose Filter)

`ThirdOrderPoseFilter` 是三个一阶低通滤波器串联，作用在 4×4 位姿上。代码与 piper-IK 0.2.0 相同。

**执行链路不用它。** `ArmIK.tick()`、`RealtimeArm` 和驱动都不调用它，有测试检查。真机回放链路去掉了滤波：旧链路的三阶滤波和 Ruckig 限速把轨迹抹平了。不要把它接进执行链路（`AGENTS.md` 第 2 条）。它适合执行链路以外、需要先去噪的位姿流，例如遥操作目标。

```python
from piper_ik import PiperIK, ThirdOrderPoseFilter

ik = PiperIK()
f = ThirdOrderPoseFilter(ik.fk(q), tau=0.06)       # 每条位姿流一个实例
reference = f.update(target_pose, dt=0.02)         # 绝对位姿 4×4，dt 是真实采样间隔（s）
result = ik.solve(reference, q)
q = result.q
f.reset(ik.fk(q))                                  # 换位姿流或重新开始时重置
```

每次 `update()`，三层依次更新，第一层的输入是目标，后两层的输入是前一层这次的输出：

```text
α = 1 − exp(−dt / tau)
p_i ← p_i + α·(p_source − p_i)
R_i ← Exp(α·Log(R_source·R_iᵀ))·R_i          # 沿最短旋转弧
```

| 参数 | 默认 | 含义 |
| :--- | :--- | :--- |
| `tau` | 0.06 s | 每层的时间常数。越大越平滑，滞后也越大 |
| `linear_speed` | 无 | 每层位置变化的速度上限（m/s） |
| `angular_speed` | 无 | 每层旋转变化的角速度上限（rad/s） |

- 滞后：低频约 3·tau（默认约 0.18 s）。单轴阶跃约 6.3·tau 到 95%。这些是连续时间近似。
- 它不保证关节速度、加速度，也不是 jerk 限制器。
- 恰好转 180° 时最短旋转轴不唯一。
- 完整示例：`python piper_ik/examples/filter_target.py`。

---

## 11. 包里没有、也不要加的 (Not Included)

用户明确要求不加：

- **平滑与限速**：控制台旧链路的三阶位姿滤波和 Ruckig 0.3 rad/s 把轨迹抹平了，已经去掉。
- **保护逻辑**：IK 跳变暂停、关节卡住就停、超时暂停、工作空间限制。
- **起点规划**：“移动到回放起点”由你们自己的复位完成：机械臂到位、静止后再启动 `RealtimeArm`。
- **结束**：`RealtimeArm` 停止后，机械臂保持最后一条指令，仍处于使能状态。失能前先让它停到安全姿态，再用 `PiperArm.disable()` 断开电机。

---

## 12. 文件结构 (Files)

```text
piper_ik/
├── core.py            # ArmIK：任意 URDF 的 fk / solve / start / observation / push_actions / tick；PiperIK = Piper 预设；hardware_limits()
├── kinematics.py      # KinematicModel（从 URDF 找臂关节、IK 坐标系、TCP）；有界阻尼最小二乘求解器 + BVLS；导入时处理 ROS 的 eigenpy 冲突
├── convention.py      # FastUMI ↔ 基座系 TCP 位姿（tcp_axes）；夹爪 0–1 ↔ 开度
├── trajectory.py      # 时间插值（位置线性 + 姿态 Slerp）和 chunk 拼接
├── follow.py          # 追点节奏和抓放时刻检测（控制台原样移植）
├── realtime.py        # RealtimeArm：控制进程和策略侧接口；Driver 接口；PiperDriver
├── arm.py             # PiperArm：与控制台相同的 piper_sdk 调用；PiperCAN：每周期的原始 CAN 收发
├── sim.py             # SimArm（任意关节数）；SimBus（模拟 Piper 的 CAN 套接字）；SimDriver（任意机械臂的模拟驱动）
├── check.py           # python -m piper_ik.check <urdf>：新 URDF 的自检
├── filters.py         # ThirdOrderPoseFilter：独立的三阶位姿滤波，执行链路不用（第 10 节）
├── model/             # Piper + UMI 夹爪的全部模型（2.3）；_build_model.py 是维护者重建 IK 版 URDF 用的，包本身不调用
├── examples/
│   ├── vla_loop.py    # 真机 VLA 模板（只填 Policy 和相机）；--can sim 跑模拟 Piper
│   ├── dry_run.py     # 无硬件演练：模拟驱动器 + 假策略
│   └── filter_target.py  # 滤波 + IK，不连设备
├── tests/             # 第 9 节的 57 个用例；data/ 里是 7 关节测试臂 arm7.urdf 和 Piper 1.0.0 参考值
├── AGENTS.md          # 改代码、接机械臂时必须遵守的规则
├── THIRD_PARTY.md     # 各部分的来源与许可
└── LICENSE            # MIT
```
