"""piper_ik: IK for any fixed-base serial arm from its URDF, on Pinocchio (the data_viewer console's own solver), plus
the console's real-arm replay chain for FastUMI actions. The Piper + UMI gripper is bundled (model/piper_umi.urdf).

ArmIK(urdf) does FK / IK and runs the chain: 200 Hz timestamp interpolation -> IK seeded with the previous solution
-> command sent unchanged, paced by 追点 ("wait for the arm"). PiperIK is ArmIK on the bundled Piper. RealtimeArm runs
that loop for a policy in a control process of its own, through a Driver (PiperDriver: the console's piper_sdk / CAN
traffic). python -m piper_ik.check <urdf> checks a new arm. ThirdOrderPoseFilter is a standalone pose filter the chain
never uses. Read README.md and AGENTS.md.
"""
from .core import ArmIK, PiperIK, IKResult, hardware_limits, MODEL, CONTROL_HZ, CONTROL_DT, DATA_FPS
from .convention import fastumi_to_robot, robot_to_fastumi, gripper_width, gripper_value, episode_rows, tcp_axes_matrix, GRIPPER_ZERO_M, GRIPPER_ONE_M, GRIPPER_EFFORT_RAW
from .kinematics import KinematicModel, SolverConfig
from .arm import PiperArm, PiperCAN
from .realtime import RealtimeArm, Driver, PiperDriver
from .sim import SimDriver
from .filters import ThirdOrderPoseFilter

__version__ = '1.1.0'
__all__ = ['ArmIK', 'PiperIK', 'RealtimeArm', 'Driver', 'PiperDriver', 'SimDriver', 'ThirdOrderPoseFilter', 'PiperArm', 'PiperCAN', 'IKResult', 'KinematicModel', 'SolverConfig',
           'hardware_limits', 'fastumi_to_robot', 'robot_to_fastumi', 'gripper_width', 'gripper_value', 'episode_rows', 'tcp_axes_matrix', 'MODEL',
           'CONTROL_HZ', 'CONTROL_DT', 'DATA_FPS', 'GRIPPER_ZERO_M', 'GRIPPER_ONE_M', 'GRIPPER_EFFORT_RAW']
