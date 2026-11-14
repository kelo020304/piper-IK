# 来源与许可

本包按 MIT 发布（`LICENSE`）。下面的部分来自别处，各自保留原许可。

| 内容 | 来源 | 许可 |
| :--- | :--- | :--- |
| IK 求解器（`kinematics.py` 的 `solve_flange()`）、三阶位姿滤波（`filters.py`） | piper-IK 0.2.0（本仓库的早期版本） | MIT（`LICENSE`） |
| 有界最小二乘 `bvls()` | SciPy `scipy/optimize/_lsq/lsq_linear.py` 与 `bvls.py` 的精简拷贝 | BSD-3-Clause（SciPy） |
| 执行链路：插值、追点、抓放检测、FastUMI 坐标约定 | 作者的 data_viewer 真机回放控制台，原样移植 | MIT（`LICENSE`） |
| Piper 关节几何（`model/piper_umi.urdf` 的关节 1–6） | piper-IK 0.2.0 的 `piper.urdf`，源自 [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros) | MIT（`model/piper_umi_models/urdf/PIPER_ROS_LICENSE.txt`） |
| Piper + UMI 夹爪模型包（`model/piper_umi_models/`） | 与 [kelo020304/piper-umi-asset](https://github.com/kelo020304/piper-umi-asset) 相同；其中的 AgileX 部分来自 [agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf) 和 piper_ros | 见 `model/piper_umi_models/THIRD_PARTY_NOTICES.md`；AgileX 部分 MIT，作者自制的 CAD / 网格保留来源权利 |
| 测试臂 `tests/data/arm7.urdf` | 本仓库编写，仅含关节（参照常见 7 轴协作臂的尺寸），不是任何真实机器人的模型 | MIT（`LICENSE`） |
| Python 依赖 | numpy、scipy、pin（Pinocchio）；真机另需 piper_sdk | 各项目自身的许可 |
