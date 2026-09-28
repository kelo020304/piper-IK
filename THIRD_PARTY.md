# 来源与许可

| 内容 | 来源 | 说明 |
| --- | --- | --- |
| Piper 运动学模型 | [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros/tree/6cf947c58797775dba148861729a5e87c99564bf) | 保留关节、连杆和惯性参数；移除 visual/collision 与网格资源。保留 [模型 MIT 许可](piper_ik/assets/LICENSE)；运行不依赖 ROS |
| PPO checkpoint | 本项目已有 MuJoCo-Warp 运动学训练 | [元数据](piper_ik/assets/metadata.json)，随项目 MIT 发布；不是 FastUMI 或官方 Piper 策略 |
| Python 依赖 | pyproject.toml / training/requirements.txt | 各项目适用自身许可 |

加载 PyTorch checkpoint 只使用可信来源。模型的几何外观不参与 FK/IK；本包不提供碰撞模型。
