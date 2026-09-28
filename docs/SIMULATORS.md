# 仿真启动

两种仿真器使用同一控制后端。**关节位置直接写入场景，属于运动学可视化，不模拟驱动器跟踪动态。**

| 仿真器 | 用途 | 安装 |
| --- | --- | --- |
| MuJoCo | 原生窗口、离屏截图、六轴模型和目标坐标轴 | `requirements.txt` |
| Isaac Sim | 同样的关节/目标回放，独立进程 | 单独安装 Isaac Sim，指定其 `python.sh` |

## 命令

```bash
./run.sh --sim mujoco --ik pinocchio
./run.sh --sim mujoco --ik rl
./run.sh --sim mujoco --ik rl --headless
./run.sh --sim isaac --ik pinocchio --isaac-python /absolute/path/to/isaacsim/python.sh
```

每次先退出上一实例。`--port 8766` 适合另外启动纯仿真。Isaac 路径也可设置 `PIPER_ISAAC_PYTHON`。主进程 Python 由 `PIPER_IK_PYTHON` 或仓库 `.venv` 决定；Isaac 使用自己的 Python，两者不必相同。

## 画面含义

网页与独立窗口显示同一组关节。真机模式显示实测反馈，纯仿真显示软件轨迹输出。短实线是当前 TCP，长虚线是目标 TCP，RGB 分别为 XYZ。蓝色格子用于观察空间关系，不是相机图像。

`headless` 隐藏原生窗口，网页模型仍可交互；仿真截图需要 EGL/OpenGL 环境。运行服务器默认只监听本机，不应把真机控制端口公开到互联网。

## 验证范围

本次发布验证 MuJoCo headless 和网页，Isaac 分支保留但未重跑。训练用 MuJoCo-Warp FK，和控制台的 Isaac 分支没有依赖关系。

[返回 README](../README.md) · [真机](HARDWARE.md)
