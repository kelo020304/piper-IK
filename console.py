"""Local WebSocket command console. SDK is owned only by the server."""
import argparse
import asyncio
import json
import sys
import urllib.request
from websockets import connect

HELP='connect → arm → home；status / pin / rl / w s a d up down q e z c r f / stop / clear / disable / disconnect / release / claim / quit'

def command(text):
    word=text.strip().lower()
    if word in ('connect','arm','home','stop','clear','disable','disconnect'):
        return {'type':'hardware','action':word}
    if word in ('pin','pinocchio','rl'):return {'type':'mode','mode':'rl' if word=='rl' else 'pin'}
    if word in ('claim','release','pause'):return {'type':word}
    if word in ('w','s','a','d','q','e','z','c','r','f','up','down'):
        return {'type':'nudge','key':{'up':'ArrowUp','down':'ArrowDown'}.get(word,word)}
    raise ValueError(HELP)

async def run(port):
    async with connect(f'ws://127.0.0.1:{port}/ws?source=terminal',origin=f'http://127.0.0.1:{port}') as ws:
        hello=json.loads(await ws.recv());client=hello['client'];state={};pending=asyncio.Queue()
        async def receive():
            nonlocal state
            async for raw in ws:
                msg=json.loads(raw)
                if msg['type']=='state':
                    if state.get('motion_busy') and not msg.get('motion_busy'):print('\n'+msg['message'],flush=True)
                    if msg.get('fault') and msg.get('fault')!=state.get('fault'):print('\n停止：'+msg['fault'],flush=True)
                    if state.get('owner')==client and msg.get('owner') not in (client,None):print('\n控制权已交给其他页面；本终端不再控制。',flush=True)
                    state=msg
                elif msg['type']=='error' or (msg['type']=='ack' and msg.get('command')!='heartbeat'):await pending.put(msg)
        async def heartbeat():
            while True:
                if state.get('owner')==client:await ws.send(json.dumps({'type':'heartbeat'}))
                await asyncio.sleep(.1)
        async def request(msg):
            await ws.send(json.dumps(msg));response=await asyncio.wait_for(pending.get(),5)
            if response['type']=='error':raise RuntimeError(response['message'])
        reader=asyncio.StreamReader();protocol=asyncio.StreamReaderProtocol(reader)
        transport,_=await asyncio.get_running_loop().connect_read_pipe(lambda:protocol,sys.stdin)
        tasks=[asyncio.create_task(receive()),asyncio.create_task(heartbeat())]
        print('已连接控制台；尚未连接机械臂。\n'+HELP)
        try:
            while True:
                print('piper> ',end='',flush=True)
                line_task=asyncio.create_task(reader.readline())
                done,_=await asyncio.wait([line_task,*tasks],return_when=asyncio.FIRST_COMPLETED)
                if line_task not in done:
                    line_task.cancel();raise RuntimeError('控制台连接中断')
                line=line_task.result()
                if not line or line.decode().strip().lower() in ('quit','exit'):break
                word=line.decode().strip().lower()
                if not word:continue
                if word=='help':print(HELP);continue
                if word=='status':
                    def fetch_state():
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/state',timeout=2) as response:return json.load(response)
                    state=await asyncio.to_thread(fetch_state)
                    print(json.dumps({k:state.get(k) for k in ('execution','mode','hardware_connected','hardware_armed','hardware_estopped','feedback_valid','feedback_age_ms','homing','motion_kind','safe_q','disable_remaining_s','q','message','fault')},ensure_ascii=False,indent=2));continue
                try:
                    msg=command(word)
                    if msg['type'] not in ('release','claim') and state.get('owner')!=client:await request({'type':'claim'})
                    await request(msg);print('已执行：'+word)
                except (ValueError,RuntimeError,asyncio.TimeoutError) as exc:print('未执行：'+str(exc))
        finally:
            transport.close()
            # Server WebSocket cleanup stops an owned hardware session on exit.
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Piper terminal controls; starts without connecting the robot')
    parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    try:asyncio.run(run(args.port))
    except KeyboardInterrupt:pass
