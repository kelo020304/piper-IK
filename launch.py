import argparse,errno,os
from piper_ik.startup import check_port_available
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
parser=argparse.ArgumentParser(description='Piper IK console; hardware stays disconnected until explicit connect.')
parser.add_argument('--sim',choices=['mujoco','isaac'],default='mujoco',help='Native simulator (default: mujoco)')
parser.add_argument('--ik',choices=['rl','pinocchio','pin'],default='pinocchio',help='Initial IK method; may also switch in browser')
parser.add_argument('--headless',action='store_true',help='No native window; browser and simulator screenshot remain available')
parser.add_argument('--isaac-python',help='Isaac Sim python.sh; alternatively PIPER_ISAAC_PYTHON')
parser.add_argument('--port',type=int,default=8765);parser.add_argument('--config')
parser.add_argument('--allow-hardware',action='store_true',help='Authorize hardware operations for this run only; does not connect or enable')
parser.add_argument('--can', '--can-interface', dest='can_interface', help='CAN interface override, e.g. can0; no config file changes')
args=parser.parse_args()
if args.allow_hardware:
    if args.sim!='mujoco':parser.error('真机同步仅支持 --sim mujoco')
if not 1<=args.port<=65535:parser.error('Invalid port')
try:check_port_available(args.port)
except OSError as exc:
    if exc.errno!=errno.EADDRINUSE:parser.error(f'无法绑定端口 {args.port}：{exc}')
    hint=f"用 ss -ltnp 'sport = :{args.port}' 查询；在旧控制台终端 Ctrl+C 退出。"
    hint+='真机模式不要换端口并行连接同一 CAN。' if args.allow_hardware else f'也可指定 --port {args.port+1}。'
    parser.error(f'端口 {args.port} 无法绑定。'+hint)
try:os.nice(8)
except OSError:pass
allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-4:])
from piper_ik.simulator import SimulatorBridge
from piper_ik.server import create_app
import uvicorn
sim=None
try:
    print(f'Starting {args.sim}; Isaac first launch may take 30–120 s. No robot connection at startup.',flush=True)
    sim=SimulatorBridge(args.sim,args.headless,args.isaac_python)
    app=create_app(args.config,authorized=args.allow_hardware,can_interface=args.can_interface,simulator=sim,mode='rl' if args.ik=='rl' else 'pin')
    print(f'Piper IK: http://127.0.0.1:{args.port}/ | {args.sim} | {args.ik} | simulator PID {sim.process.pid}',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='info',access_log=False,ws='websockets',ws_ping_interval=1.,ws_ping_timeout=2.)
finally:
    if sim:sim.close()
