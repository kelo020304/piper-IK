import asyncio,json,time,uuid,os
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from fastapi import FastAPI,WebSocket,WebSocketDisconnect,HTTPException,Request
from fastapi.responses import FileResponse,HTMLResponse,Response,RedirectResponse
from fastapi.staticfiles import StaticFiles
from .engine import Engine
ROOT=Path(__file__).resolve().parents[1]

def create_app(config_path=None,authorized=False,sdk_factory=None,simulator=None,mode="pin",can_interface=None):
    config=json.loads(Path(config_path or ROOT/'config.json').read_text())
    # Process-local CLI overrides; never persist authorization to disk.
    config['hardware_enabled']=bool(authorized)
    if can_interface is not None:
        if not can_interface or len(can_interface)>15 or any(c.isspace() or c=='/' for c in can_interface):raise ValueError('Invalid CAN interface name')
        config['can_interface']=can_interface
    engine=Engine(config,authorized,sdk_factory)
    if mode not in ('pin','rl'):raise ValueError('Unknown IK mode')
    engine.mode=mode
    if simulator:engine.attach_simulator(simulator)
    clients={}
    async def loop():
        previous_tick=None
        while True:
            if simulator and engine.simulator_ok and simulator.process.poll() is not None:
                with engine.lock:
                    engine.simulator_ok=False;engine.stop('仿真器已关闭，请重启控制台');engine.fault='Simulator process exited'
            if not clients and engine.execution=='simulation' and engine.paused and max(abs(engine.limiter.v))<1e-10:
                previous_tick=None;await asyncio.sleep(.1);continue
            # An accepted target keeps running while the owner's transport is alive.
            # Uvicorn protocol ping/pong detects loss independently of browser timers/focus.
            with engine.lock:
                if engine.owner in clients:engine.last_heartbeat=time.monotonic()
            started=time.monotonic()
            elapsed=.02 if previous_tick is None else started-previous_tick;previous_tick=started
            # Account for scheduling jitter, but never catch up a long stall
            # with one large motion increment.
            await asyncio.to_thread(engine.tick,min(.04,max(.001,elapsed)))
            state=engine.snapshot()
            for client,ws in list(clients.items()):
                try:await asyncio.wait_for(ws.send_json({'type':'state',**state}),timeout=.05)
                except Exception:clients.pop(client,None)
            await asyncio.sleep(max(.001,.02-(time.monotonic()-started)))
    @asynccontextmanager
    async def lifespan(app):
        task=asyncio.create_task(loop())
        yield
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
        if engine.hardware.sdk is not None:engine.hardware.disconnect()
        if simulator:await asyncio.to_thread(simulator.close)
    app=FastAPI(lifespan=lifespan);app.state.engine=engine
    @app.get('/')
    async def index():return FileResponse(ROOT/'web/index.html')
    @app.get('/api/state')
    async def state():return engine.snapshot()
    @app.get('/api/simulator/frame')
    async def simulator_frame():
        if not simulator:raise HTTPException(404,'未连接独立仿真器')
        try:return Response(await asyncio.to_thread(simulator.capture),media_type='image/png',headers={'Cache-Control':'no-store'})
        except Exception as exc:raise HTTPException(503,str(exc))
    @app.get('/sim-guide/')
    async def sim_guide():return RedirectResponse('/docs/SIMULATORS.html')
    @app.get('/hardware-guide/')
    async def hardware_guide():return RedirectResponse('/docs/HARDWARE.html')
    @app.post('/api/hardware/{action}')
    async def hardware_block(action:str):
        raise HTTPException(403,'真机锁定；硬件操作只通过授权的同源控制会话执行')
    @app.websocket('/ws')
    async def socket(ws:WebSocket):
        origin=ws.headers.get('origin');expected=ws.headers.get('host')
        if not origin or urlparse(origin).netloc!=expected:
            await ws.close(code=1008,reason='Same-origin control only');return
        await ws.accept();client=uuid.uuid4().hex;await ws.send_json({'type':'hello','client':client});clients[client]=ws
        try:
            while True:
                msg=await ws.receive_json()
                try:
                    if not isinstance(msg,dict):raise ValueError('Expected command object')
                    if msg.get('type')=='hardware':await asyncio.to_thread(engine.hardware_action,client,msg.get('action'))
                    else:
                        msg['source']=ws.query_params.get('source','other')
                        await asyncio.to_thread(engine.control,client,msg)
                    await ws.send_json({'type':'ack','command':msg.get('type')})
                except Exception as exc:await ws.send_json({'type':'error','message':str(exc)})
        except (WebSocketDisconnect,RuntimeError):pass
        finally:
            clients.pop(client,None)
            with engine.lock:
                if engine.owner==client:engine.stop('控制页面已断开');engine.owner=None
    @app.get('/report/')
    async def report():return RedirectResponse('/docs/VALIDATION.html')
    @app.get('/benchmark-guide/')
    async def benchmark_guide():return RedirectResponse('/docs/RL.html')
    @app.get('/guide/')
    async def guide():return RedirectResponse('/README.html')
    @app.get('/README.html')
    async def readme():return FileResponse(ROOT/'README.html')
    @app.get('/THIRD_PARTY.html')
    async def third_party():return FileResponse(ROOT/'THIRD_PARTY.html')
    app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
    app.mount('/models',StaticFiles(directory=ROOT/'assets/models/meshes'),name='models')
    app.mount('/docs',StaticFiles(directory=ROOT/'docs'),name='docs')
    return app
