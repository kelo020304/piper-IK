"""Local simulator process bridge. No robot SDK or CAN access."""
import base64,json,os,signal,socket,subprocess,tempfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class SimulatorBridge:
    def __init__(self,backend,headless=False,isaac_python=None,timeout=120):
        if backend not in ('mujoco','isaac'):raise ValueError('Unknown simulator')
        self.backend=backend;self.headless=headless;self.lock=threading.Lock();self.closed=False
        self.temp=tempfile.TemporaryDirectory(prefix='piper-sim-');self.path=Path(self.temp.name)/'control.sock'
        self.log_path=ROOT/'artifacts'/f'{backend}-live-{os.getpid()}.log';self.log_path.parent.mkdir(exist_ok=True)
        env=os.environ.copy()
        for key in ['PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD']:env.pop(key,None)
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if backend=='mujoco':
            import sys
            runner=sys.executable
        else:
            runner=isaac_python or os.environ.get('PIPER_ISAAC_PYTHON')
            if not runner:
                self.temp.cleanup()
                raise ValueError('Isaac requires --isaac-python /absolute/path/to/isaacsim/python.sh or PIPER_ISAAC_PYTHON')
        if not Path(runner).is_file():self.temp.cleanup();raise FileNotFoundError(f'Simulator Python not found: {runner}')
        command=[runner,str(ROOT/'sim_worker.py'),'--sim',backend,'--socket',str(self.path)]
        if headless:command.append('--headless')
        self.log=self.log_path.open('w');self.process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        self.sock=None;self.reader=None
        try:
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError(f'{backend} exited; inspect {self.log_path}')
                if self.path.exists():
                    self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.settimeout(5)
                    self.sock.connect(str(self.path));self.reader=self.sock.makefile('rb');break
                time.sleep(.1)
            else:raise TimeoutError(f'{backend} startup timed out; inspect {self.log_path}')
            self.ready=self.request({'op':'ping'})
        except BaseException:self.close();raise
    def request(self,message):
        with self.lock:
            if self.closed or self.process.poll() is not None:raise RuntimeError('Simulator has closed')
            self.sock.sendall((json.dumps(message)+'\n').encode());line=self.reader.readline(4*1024*1024)
            if not line:raise ConnectionError('Simulator feedback disconnected')
            response=json.loads(line)
            if 'error' in response:raise RuntimeError(response['error'])
            return response
    def step(self,q,target,tcp):return self.request({'op':'step','q':q.tolist(),'target':target.tolist(),'tcp':tcp.tolist()})
    def capture(self):return base64.b64decode(self.request({'op':'capture'})['png'])
    def close(self):
        if self.closed:return
        self.closed=True
        if self.sock is not None:
            try:self.sock.sendall(b'{"op":"quit"}\n')
            except OSError:pass
            if self.reader:self.reader.close()
            self.sock.close()
        try:self.process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        # Some simulator helpers can outlive the wrapper. Only signal the
        # private process group created by this bridge, never other apps.
        try:os.killpg(self.process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        self.log.close();self.temp.cleanup()
