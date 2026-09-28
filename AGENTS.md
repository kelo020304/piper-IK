# Piper IK
- Standalone runtime and training; no ROS imports, services or sibling-directory dependencies.
- Never connect CAN or run a real SDK for development tests. Use injected fake SDKs.
- Startup remains disconnected; hardware access requires explicit CLI authorization and UI actions.
- Keep base-frame absolute TCP targets, base-axis keyboard increments, wxyz RL observations, and absolute SDK joint commands consistent. See README.md.
- Runtime simulators are kinematic visualization, not physical actuator validation.
- Maintain README.md / README.html and docs/*.md / *.html together using tools/build_docs.py.
- Test changed control paths with pytest; never restart a user's existing hardware console.
