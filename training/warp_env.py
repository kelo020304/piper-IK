"""Batched MuJoCo-Warp kinematic IK environment (no dynamics integration).

Matches the reference's kinematic RL problem. Goal and current FK are computed
by MuJoCo's GPU backend; no analytic IK targets or demonstrations enter PPO.
"""
from pathlib import Path
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
import torch

ROOT = Path(__file__).resolve().parent


@torch.jit.script
def quat_error(current: torch.Tensor, target: torch.Tensor):
    # Quaternions in MuJoCo w,x,y,z order. Relative quaternion target*conj(current).
    w = target[:, :1]*current[:, :1] + (target[:, 1:]*current[:, 1:]).sum(-1,keepdim=True)
    v = -target[:, :1]*current[:, 1:] + current[:, :1]*target[:, 1:] - torch.cross(target[:, 1:],current[:, 1:],dim=-1)
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    w=w*sign;v=v*sign
    nv=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    angle=2*torch.atan2(nv,w.clamp_min(1e-8))
    return v*(angle/nv.clamp_min(1e-8)),angle.squeeze(-1)


class WarpIKEnv:
    def __init__(self,n=1024,seed=7,precision=False,near_fraction=.5,centered=False):
        wp.init();self.device='cuda:0';self.n=n
        self.precision=precision;self.near_fraction=near_fraction if precision else 0.
        self.centered=centered
        self.position_tol=.003 if precision else .05
        self.orientation_tol=float(np.deg2rad(1.5)) if precision else .2
        torch.manual_seed(seed)
        self.cpu=mujoco.MjModel.from_xml_path(str(ROOT.parent/'assets/models/piper.xml'))
        self.cpu.geom_contype[:]=0;self.cpu.geom_conaffinity[:]=0
        self.body=self.cpu.body('tool0').id
        with wp.ScopedDevice(self.device):
            self.model=mjw.put_model(self.cpu)
            self.data=mjw.make_data(self.cpu,nworld=n,nconmax=0,njmax=0)
            self.target_data=mjw.make_data(self.cpu,nworld=n,nconmax=0,njmax=0)
        self.q=wp.to_torch(self.data.qpos)
        self.tq=wp.to_torch(self.target_data.qpos)
        self.pos=wp.to_torch(self.data.xpos)[:,self.body]
        self.quat=wp.to_torch(self.data.xquat)[:,self.body]
        self.target_pos=wp.to_torch(self.target_data.xpos)[:,self.body]
        self.target_quat=wp.to_torch(self.target_data.xquat)[:,self.body]
        self.lo=torch.as_tensor(self.cpu.jnt_range[:,0],dtype=torch.float32,device=self.device)
        self.hi=torch.as_tensor(self.cpu.jnt_range[:,1],dtype=torch.float32,device=self.device)
        self.age=torch.zeros(n,dtype=torch.long,device=self.device)
        self.mask=torch.ones(n,dtype=torch.bool,device=self.device)
        self.reset(self.mask)

    def fk(self):
        # Explicit synchronization of Torch/Warp streams for shared tensor memory.
        stream=wp.stream_from_torch(torch.cuda.current_stream())
        with wp.ScopedStream(stream): mjw.kinematics(self.model,self.data)

    def reset(self,mask):
        rand_start=self.lo+.02+(self.hi-self.lo-.04)*torch.rand((self.n,6),device=self.device)
        rand_goal=self.lo+.02+(self.hi-self.lo-.04)*torch.rand((self.n,6),device=self.device)
        if self.near_fraction:
            near=torch.rand((self.n,1),device=self.device)<self.near_fraction
            sigma=torch.exp(torch.empty((self.n,1),device=self.device).uniform_(np.log(.005),np.log(.25)))
            local=torch.clamp(rand_goal+sigma*torch.randn((self.n,6),device=self.device),self.lo,self.hi)
            rand_start=torch.where(near,local,rand_start)
        self.q.copy_(torch.where(mask[:,None],rand_start,self.q))
        self.tq.copy_(torch.where(mask[:,None],rand_goal,self.tq))
        self.age.masked_fill_(mask,0)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            mjw.kinematics(self.model,self.target_data)
        self.fk()

    def obs(self):
        rv,oe=quat_error(self.quat,self.target_quat)
        dp=self.target_pos-self.pos
        # Canonical quaternion sign prevents equivalent poses changing observations.
        cq=torch.where(self.quat[:,:1]<0,-self.quat,self.quat)
        tq=torch.where(self.target_quat[:,:1]<0,-self.target_quat,self.target_quat)
        return torch.cat((self.pos,cq,self.target_pos,tq,self.q,dp*(20 if self.precision else 1),rv*(3 if self.precision else 1)),dim=-1),dp.norm(dim=-1),oe

    def propose(self,action):
        gain=.2
        if self.precision and not self.centered:
            _,pe,oe=self.obs()
            gain=(.2*torch.maximum(pe/.1,oe/.5).clamp(.01,1.))[:,None]
        return torch.maximum(self.lo,torch.minimum(self.hi,self.q+gain*action.clamp(-1,1)))

    def step(self,action):
        if self.precision:
            _,p0,o0=self.obs();cost0=torch.log1p((p0/.003)**2)+.5*torch.log1p((o0/self.orientation_tol)**2)
        self.q.copy_(self.propose(action))
        self.age+=1;self.fk()
        obs,pe,oe=self.obs()
        success=(pe<self.position_tol)&(oe<self.orientation_tol)
        # Explicit reconstruction of the described v5 principles, not an exact
        # recovered source implementation: independent fine rewards + terminal bonus.
        reward=-2*torch.atan(pe)-torch.atan(oe)+torch.exp(-pe/.05)+torch.exp(-oe/.2)-.5+100*success.float()
        if self.precision:
            cost=torch.log1p((pe/.003)**2)+.5*torch.log1p((oe/self.orientation_tol)**2)
            reward=(cost0-cost)-.2*cost-.1+100*success.float()
        done=success|(self.age>=200)
        terminal_obs=obs.clone()
        self.reset(done)
        return self.obs()[0],reward,done,success,terminal_obs

    def audit(self,count=32):
        q=self.q[:count].cpu().numpy();gp=self.pos[:count].cpu().numpy();gq=self.quat[:count].cpu().numpy()
        data=mujoco.MjData(self.cpu);ep=[];eq=[]
        for i,angles in enumerate(q):
            data.qpos[:]=angles;mujoco.mj_kinematics(self.cpu,data)
            ep.append(np.linalg.norm(data.xpos[self.body]-gp[i]))
            eq.append(min(np.linalg.norm(data.xquat[self.body]-gq[i]),np.linalg.norm(data.xquat[self.body]+gq[i])))
        return {'samples':count,'max_position_error_m':float(max(ep)),'max_quaternion_l2_error':float(max(eq))}
