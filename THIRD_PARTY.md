# 第三方来源

| 内容 | 来源 | 许可与说明 |
| --- | --- | --- |
| Piper URDF / 网格 | [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros/tree/6cf947c58797775dba148861729a5e87c99564bf) | [模型 MIT 许可](assets/models/LICENSE)，保留版权；只使用模型资源，不依赖 ROS |
| Three.js、OrbitControls、STLLoader | [three.js](https://github.com/mrdoob/three.js) | [MIT 许可](web/vendor/THREE-LICENSE.txt)，前端资源已本地包含 |
| PPO checkpoint | 本项目既有 MuJoCo-Warp 运动学训练 | [权重元数据](assets/weights/metadata.json)，随项目 MIT 发布；不是 FastUMI / 官方 Piper 权重 |
| Python 依赖 | requirements 文件中的各项目 | 单独安装，各自保留上游许可，不由本仓库重新授权 |

模型历史核对记录见 [provenance.json](assets/models/provenance.json)。其中历史模型哈希和 FK 检查来自整理前的记录；不是本次对全部资产文件的重新审计。

权重使用 PyTorch checkpoint 格式，包含优化器状态。只加载可信来源的 checkpoint；替换权重时同步更新元数据。
