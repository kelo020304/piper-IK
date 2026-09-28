# 验证记录

发布整理日期：2026-09-28。测试在独立目录执行，没有连接 CAN 或操作真机。

## 本次检查

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| pytest | **108 passed**，7.01 s | fake SDK 状态机、使能/回零/失能、控制权、坐标转换、滤波、轨迹连续性、训练/推理一致性及 MuJoCo 子进程；一条第三方 hppfcl 弃用提示 |
| MuJoCo headless + 网页 | 通过 | 独立端口启动；RL 键盘输入后关节变化；切换 Pinocchio；原生离屏截图 HTTP 200；无浏览器脚本错误 |
| HTML | 通过 | README / RL / Pinocchio / 真机页，1440 px 与 390 px 宽度检查，无整页横向溢出 |
| CUDA 短训练 | 通过 | RTX 4090，32 环境 × 32 horizon，1,024 transitions；训练、验证和 checkpoint 保存完成；**不代表收敛** |
| MuJoCo-Warp / CPU FK | 通过 | 32 个姿态最大位置差 1.46e-7 m，四元数最小符号等价 L2 差 1.65e-7 |
| Isaac / 真机 | 本次未运行 | 不把 fake SDK、运动学回放当作电机动态验证 |

短训练最后固定验证成功率为 0%，只验证运行链路；不能将 rollout 的局部近目标成功率当成泛化成功率。原发布权重未被短训练替换。

## 新的独立小样本检查

两方法均为随机初始关节、FK 生成目标、种子 94027、32 样本、最多 200 次迭代，不做多初值搜索，不经过参考平滑。成功需同时满足 3 mm 和 1.5°。

| 方法 | 成功 | 位置误差中位数 | 姿态误差中位数 |
| --- | --- | --- | --- |
| RL 现有权重 | 20/32（62.5%） | 2.962 mm | 0.690° |
| Pinocchio 局部求解 | 16/32（50.0%） | 3.887 mm | 0.590° |

这是小样本流程检查，**不能据此断言 RL 优于 Pinocchio**；Pinocchio 易受随机初值与局部极小影响，实际连续控制使用前一拍解。原始每例记录见 [RL](evaluation_rl.json) / [Pinocchio](evaluation_pin.json)。

## 现有权重

权重累计训练计数为 615,448,576 transitions；这是从已有 checkpoint 继续训练后的总计数，不是本次重新训练量。历史固定 256 个随机验证样本在 3 mm / 1.5°、最多 200 次迭代下成功率 66.41%。该指标不是网页单拍 8 ms / 20 次迭代成功率，也不代表真机任务成功率。

[历史计时](historical_training_timing.json)、[运行环境](runtime_versions.json)、[权重说明](../assets/weights/metadata.json) 保留来源和范围。历史运行时间不等于从零训练收敛时间；没有可保证的通用收敛时长。

## 复测

```bash
python -m pytest -q
python training/evaluate.py --ik rl --samples 256 --out artifacts/rl-evaluation.json
python training/evaluate.py --ik pinocchio --samples 256 --out artifacts/pin-evaluation.json
```

独立评估使用相同随机目标与初始关节，但与历史验证集不同。目标由 FK 生成，因此运动学可达；局部 IK 仍可能失败。结果不包含碰撞、奇异性安全、夹爪、通信延迟或电机动态。

[返回 README](../README.md)
