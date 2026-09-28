# Piper IK SDK
- Keep the public API small: PiperIK, IKBackend, IKResult, solver configs, ThirdOrderPoseFilter.
- No ROS, device I/O, web servers or simulator launchers in this package.
- Targets and FK are absolute base-frame TCP poses; joints are absolute radians.
- Custom backends receive base-frame tool0 targets. Preserve the TCP conversion contract.
- Keep optional Torch imports lazy. Filtering does not import Pinocchio or Torch.
- Maintain concise Markdown and generated HTML docs together with tools/build_docs.py.
- Test FK/IK, filtering, wheel resources and the training/deployment observation contract.
