"""The measured PPO actor; warm-start inference only during teleoperation."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation

class Policy:
    def __init__(self,weights,lo,hi):
        torch.set_num_threads(1)
        self.actor=nn.Sequential(nn.Linear(26,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,6))
        ck=torch.load(weights,map_location='cpu',weights_only=False)
        self.actor.load_state_dict({k.removeprefix('actor.'):v for k,v in ck['model'].items() if k.startswith('actor.')})
        self.actor.eval();self.steps=ck['steps'];self.lo=lo;self.hi=hi
        assert ck['config'].get('precision') and ck['config'].get('centered')
    @torch.inference_mode()
    def propose(self,q,T,G):
        cq=np.roll(Rotation.from_matrix(T[:3,:3]).as_quat(),1);cq*=1 if cq[0]>=0 else -1
        tq=np.roll(Rotation.from_matrix(G[:3,:3]).as_quat(),1);tq*=1 if tq[0]>=0 else -1
        dp=G[:3,3]-T[:3,3];rv=Rotation.from_matrix(G[:3,:3]@T[:3,:3].T).as_rotvec()
        x=torch.from_numpy(np.r_[T[:3,3],cq,G[:3,3],tq,q,dp*20,rv*3].astype('float32'))
        zero=x.clone();zero[7:14]=zero[:7];zero[20:]=0
        action=(self.actor(x)-self.actor(zero)).numpy().clip(-1,1)
        return np.clip(q+.2*action,self.lo,self.hi)
    def solve(self,q,target,fk,budget_s=.008,max_steps=20):
        started=time.perf_counter();q=q.copy()
        for _ in range(max_steps):
            T=fk(q);pe=np.linalg.norm(T[:3,3]-target[:3,3]);oe=Rotation.from_matrix(T[:3,:3].T@target[:3,:3]).magnitude()
            if (pe<.00015 and oe<np.deg2rad(.15)) or time.perf_counter()-started>budget_s:break
            q=self.propose(q,T,target)
            if not np.isfinite(q).all():raise RuntimeError('Policy produced non-finite joints')
        return q
