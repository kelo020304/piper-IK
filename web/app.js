import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {STLLoader} from './vendor/STLLoader.js';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const viewport=$('#viewport');let state=null,client=null,socket=null,hold=null,pendingNudge=null,lastKeyTime=0,lastErrorUntil=0;
const known=new Set(['w','a','s','d','q','e','z','c','r','f','ArrowUp','ArrowDown']);
function send(message){if(socket?.readyState===WebSocket.OPEN)socket.send(JSON.stringify(message));}
function notice(message,error=false){$('#message').textContent=message;$('#message').classList.toggle('error',error);if(error)lastErrorUntil=Date.now()+2500;}
function own(){return state&&state.owner===client;}
function otherOwner(){return Boolean(state?.owner&&!own());}
function ensureOwner(){if(otherOwner()){notice('请先点击“接管到本页”',true);return false;}if(!own())claim();return true;}
function claim(){send({type:'claim'});viewport.focus({preventScroll:true});}
function clearKeyboardInput(){clearInterval(hold);clearTimeout(pendingNudge);hold=null;pendingNudge=null;$$('[data-key]').forEach(b=>b.classList.remove('pressed'));}
function pause(release=false){clearKeyboardInput();if(own())send({type:release?'release':'pause'});}
function nudge(key){if(document.hidden||!document.hasFocus()||state?.motion_busy)return;if(!own()||state.paused){notice('请先接管键盘',true);return;}send({type:'nudge',key});}
function connect(){
 socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws?source=web`);
 socket.onopen=()=>{$('#connection').textContent='本地已连接';$('#connection').classList.remove('error');};
 socket.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.type==='hello'){client=msg.client;return;}if(msg.type==='error'){notice(msg.message,true);return;}if(msg.type==='state'){state=msg;renderState();}};
 socket.onclose=()=>{$('#connection').textContent='连接断开';$('#connection').classList.add('error');clearInterval(hold);state=null;client=null;notice('连接已断开，控制已停止',true);setTimeout(connect,1800);};
}
// Connection liveness is checked by server-side WebSocket ping/pong, not page focus.
$('#claim').onclick=()=>{if(otherOwner()){send({type:'takeover'});viewport.focus({preventScroll:true});}else claim();};$('#pause').onclick=()=>pause();$('#reset').onclick=()=>{if(!ensureOwner())return;send({type:'reset'});};
$$('[data-mode]').forEach(b=>b.onclick=()=>{if(!ensureOwner())return;send({type:'mode',mode:b.dataset.mode});});
for(const input of [$('#step-mm'),$('#step-deg')])input.onchange=()=>{if(!ensureOwner())return;send({type:'steps',mm:Number($('#step-mm').value),deg:Number($('#step-deg').value)});};
$$('[data-hardware]').forEach(b=>b.onclick=()=>{if(!ensureOwner())return;send({type:'hardware',action:b.dataset.hardware});});
$$('[data-key]').forEach(b=>{
 b.onpointerdown=e=>{e.preventDefault();if(state?.motion_busy||!ensureOwner())return;if(state.paused)claim();pendingNudge=setTimeout(()=>nudge(b.dataset.key),50);b.classList.add('pressed');clearInterval(hold);hold=setInterval(()=>nudge(b.dataset.key),130);b.setPointerCapture(e.pointerId);};
 const end=()=>clearKeyboardInput();b.onpointerup=end;b.onpointercancel=end;b.onlostpointercapture=end;
});
window.addEventListener('keydown',e=>{
 if(e.target.closest('input,select,textarea')||e.ctrlKey||e.metaKey||e.altKey)return;
 if(e.code==='Space'){e.preventDefault();pause();return;}
 const key=e.key.startsWith('Arrow')?e.key:e.key.toLowerCase();if(!known.has(key))return;
 e.preventDefault();if(Date.now()-lastKeyTime<120)return;lastKeyTime=Date.now();nudge(key);$$('[data-key]').filter(b=>b.dataset.key===key).forEach(b=>b.classList.add('pressed'));
});
window.addEventListener('keyup',e=>{$$('[data-key]').filter(b=>b.dataset.key===(e.key.startsWith('Arrow')?e.key:e.key.toLowerCase())).forEach(b=>b.classList.remove('pressed'));});
window.addEventListener('blur',clearKeyboardInput);document.addEventListener('visibilitychange',()=>{if(document.hidden)clearKeyboardInput();});window.addEventListener('beforeunload',()=>pause(true));
function renderState(){
 $$('#actual-pose span').forEach(e=>e.remove());$$('#target-pose span').forEach(e=>e.remove());
 for(const [id,p] of [['actual',state.actual],['target',state.target]])for(const value of [...p.position.map(x=>x*1000),...p.rpy]){const span=document.createElement('span');span.textContent=value.toFixed(2);$(`#${id}-pose`).appendChild(span);}
 $('#position-error').textContent=state.position_error_mm.toFixed(2);$('#rotation-error').textContent=state.orientation_error_deg.toFixed(2);$('#solver-ms').textContent=state.solver_ms.toFixed(1);
 $('#joints').textContent=state.q.map((q,i)=>`J${i+1} 实测 ${(q*180/Math.PI).toFixed(2)}° / 命令 ${(state.command_q[i]*180/Math.PI).toFixed(2)}°`).join('　');
 $('#tracking-details').textContent=state.command_position_error_mm===undefined?'':`命令位姿距目标 ${state.command_position_error_mm.toFixed(2)} mm / ${state.command_orientation_error_deg.toFixed(2)}°；命令关节近限位：${state.near_joint_limits.length?state.near_joint_limits.map(i=>'J'+i).join('、'):'无'}。仅供观察，不触发停止。`;
 $$('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode));
 $('#owner-status').textContent=otherOwner()?(state.owner_kind==='terminal'?'终端占用':state.owner_kind==='web'?'其他网页占用':'其他控制端占用'):own()?'本页控制':'未接管';
 $('#claim').textContent=otherOwner()?'接管到本页':state.motion_busy?'自动运动中':own()&&!state.paused?'键盘控制中':'接管键盘';
 $('#claim').disabled=own()&&state.motion_busy;
 $$('[data-key],[data-mode]').forEach(b=>b.disabled=otherOwner()||state.motion_busy);
$('#execution').textContent=state.execution==='simulation'?`${state.simulator==='isaac'?'Isaac Sim':'MuJoCo'} · ${state.simulator_connected?'实时反馈':'仿真未连接'} · 运动学`:state.feedback_valid?'真机实测 → MuJoCo':state.hardware_connected?'真机反馈失效 · 冻结':'真机未连接 · 占位姿态';
 $('#sim-frame').hidden=!state.simulator_connected;
 $('#reset').disabled=state.execution!=='simulation';
 $('#hardware-status').textContent=state.hardware_allowed?(state.hardware_estopped?(state.fault?'故障停止':'已停止'):state.hardware_armed?'已使能':state.hardware_connected?'已连接':'未连接'):'已锁定';
 $('.hardware').hidden=!state.hardware_allowed;
 $('#hardware-note').classList.toggle('error',Boolean(state.fault));
 $('#hardware-note').textContent=state.fault?`故障：${state.fault}。恢复：清除故障 → 使能 → 回零。`:state.hardware_estopped?'已停止：清除故障 → 使能 → 回零。':state.hardware_connected&&!state.feedback_valid?'等待有效关节反馈，暂不能使能。':state.hardware_connected&&!state.hardware_armed?'尚未使能：点击使能后才能回零。':state.motion_kind==='return_safe'?'返回首次使能位置…':state.motion_kind==='disable_wait'?`到位保持 · ${state.disable_remaining_s.toFixed(1)} 秒后失能`:state.safe_q?'失能：返回首次使能姿态 → 保持 3 秒 → 失能。':'使能时记录安全位置；请先摆在期望的安全姿态。';
 $('#safe-joints').textContent=state.safe_q?state.safe_q.map((q,i)=>`J${i+1} ${(q*180/Math.PI).toFixed(2)}°`).join('　'):'尚未记录';
 $$('[data-hardware]').forEach(b=>{
  const a=b.dataset.hardware;
  b.title=a==='clear'?'清除软件故障及停止锁存；不会使能或运动':a==='home'?(state.hardware_estopped?'停止锁存：先清除故障，再使能':!state.hardware_armed?'请先使能':state.motion_busy?'自动运动中':otherOwner()?'请先接管到本页':'缓速回到 SDK 零角度'):'';
  b.disabled=!state.hardware_allowed||otherOwner()||(a==='connect'&&state.hardware_connected)||(a!=='connect'&&!state.hardware_connected)||(a==='arm'&&(state.hardware_armed||state.hardware_estopped||!state.feedback_valid))||(a==='home'&&(!state.hardware_armed||state.motion_busy))||(a==='disable'&&(!state.hardware_armed||!state.safe_q||['return_safe','disable_wait'].includes(state.motion_kind)))||(a==='clear'&&state.hardware_armed);
 });
 if(Date.now()>lastErrorUntil)notice(state.fault||state.message,Boolean(state.fault));
 updateRobot();
}
let scene,camera,renderer,controls,robotMeshes={},goal,actualAxes;
function poseLabel(text,color){
 const canvas=document.createElement('canvas');canvas.width=384;canvas.height=80;
 const ctx=canvas.getContext('2d');ctx.fillStyle='rgba(15,32,43,.85)';ctx.fillRect(0,0,384,80);ctx.font='bold 36px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=color;ctx.fillText(text,192,40);
 const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),depthTest:false,depthWrite:false}));sprite.scale.set(.135,.028,1);sprite.renderOrder=12;return sprite;
}
function targetFrame(){
 const group=new THREE.Group();group.name='target-pose';
 const points=[],colors=[],palette=[[1,.15,.15],[.15,1,.3],[.15,.5,1]];
 for(let axis=0;axis<3;axis++){
  for(const [a,b] of [[0,.2],[.3,.5],[.6,.8],[.9,1]]){
   const start=[0,0,0],end=[0,0,0];start[axis]=a*.13;end[axis]=b*.13;points.push(...start,...end);colors.push(...palette[axis],...palette[axis]);
  }
 }
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
 const axes=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({vertexColors:true,depthTest:false,depthWrite:false}));axes.name='target-dashed-axes';axes.renderOrder=10;group.add(axes);
 const marker=new THREE.Mesh(new THREE.SphereGeometry(.006,16,12),new THREE.MeshBasicMaterial({color:0xffc36a,depthTest:false}));marker.renderOrder=11;group.add(marker);
 const label=poseLabel('目标 TARGET','#ffc36a');label.position.set(0,0,.17);group.add(label);return group;
}
async function initViewer(){
 scene=new THREE.Scene();scene.background=new THREE.Color('#d6e1e9');
 camera=new THREE.PerspectiveCamera(38,1,.01,20);camera.up.set(0,0,1);camera.position.set(.95,-1.15,.85);
 renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=false;viewport.prepend(renderer.domElement);
 controls=new OrbitControls(camera,renderer.domElement);controls.target.set(.17,0,.25);controls.enableDamping=true;controls.minDistance=.45;controls.maxDistance=2.5;controls.maxPolarAngle=Math.PI*.85;controls.update();
 scene.add(new THREE.HemisphereLight(0xffffff,0x899c91,2.5));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(1,-1,2);scene.add(light);
 const checker=document.createElement('canvas');checker.width=64;checker.height=64;
 const ctx=checker.getContext('2d');ctx.fillStyle='#1a334d';ctx.fillRect(0,0,64,64);ctx.fillStyle='#334d66';ctx.fillRect(32,0,32,32);ctx.fillRect(0,32,32,32);
 const floorTexture=new THREE.CanvasTexture(checker);floorTexture.colorSpace=THREE.SRGBColorSpace;floorTexture.wrapS=floorTexture.wrapT=THREE.RepeatWrapping;floorTexture.repeat.set(10,10);floorTexture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());
 const floor=new THREE.Mesh(new THREE.PlaneGeometry(4,4),new THREE.MeshBasicMaterial({map:floorTexture}));floor.name='blue-checker-floor';floor.position.z=-.01;scene.add(floor);
 scene.add(new THREE.AxesHelper(.12));
 goal=targetFrame();scene.add(goal);
 actualAxes=new THREE.Group();actualAxes.name='actual-pose';const axes=new THREE.AxesHelper(.075);axes.material.depthTest=false;axes.material.depthWrite=false;axes.renderOrder=9;actualAxes.add(axes);
 const actualLabel=poseLabel('当前 ACTUAL','#e5f4ff');actualLabel.position.set(0,0,-.055);actualAxes.add(actualLabel);scene.add(actualAxes);
 const loader=new STLLoader();await Promise.all(['base_link',...Array.from({length:6},(_,i)=>`link${i+1}`)].map(async name=>{const geometry=await loader.loadAsync(`/models/${name}.STL`);geometry.computeVertexNormals();const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xd8dfdf,roughness:.57,metalness:.2}));mesh.name=name;scene.add(mesh);robotMeshes[name]=mesh;}));
 $('#viewer-status').hidden=true;updateRobot();
 const resize=()=>{const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();};new ResizeObserver(resize).observe(viewport);resize();
 let previous=0;function animate(now){requestAnimationFrame(animate);if(document.hidden||now-previous<33)return;previous=now;controls.update();renderer.render(scene,camera);}requestAnimationFrame(animate);
 window.__piperViewer={scene,camera,renderer,robotMeshes,get state(){return state;}};
}
function updateRobot(){if(!state||!goal)return;for(const link of state.links){const m=robotMeshes[link.name];if(m){m.position.fromArray(link.position);m.quaternion.fromArray(link.quaternion);}}goal.position.fromArray(state.target.position);goal.quaternion.fromArray(state.target.quaternion);actualAxes.position.fromArray(state.actual.position);actualAxes.quaternion.fromArray(state.actual.quaternion);}
initViewer().catch(e=>{$('#viewer-status').textContent='三维视图加载失败：'+e.message;console.error(e);});connect();
