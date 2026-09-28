"""PPO on actual batched MuJoCo-Warp FK. No ROS, no IK imitation targets."""
import argparse,json,math,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn


ROOT=Path(__file__).resolve().parent


class ActorCritic(nn.Module):
    def __init__(self,centered=False):
        super().__init__()
        self.centered=centered
        def mlp(out): return nn.Sequential(nn.Linear(26,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,out))
        self.actor=mlp(6);self.critic=mlp(1);self.logstd=nn.Parameter(torch.full((6,),-.5))
        for module in self.modules():
            if isinstance(module,nn.Linear): nn.init.orthogonal_(module.weight,math.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)

    def mean(self,obs):
        out=self.actor(obs)
        if self.centered:
            zero=obs.clone();zero[:,7:14]=zero[:,:7];zero[:,20:]=0.
            out=out-self.actor(zero)
        return out

    def act(self,obs,action=None):
        scale=1.
        if self.centered:
            scale=torch.maximum(obs[:,20:23].norm(dim=-1)/2.,obs[:,23:26].norm(dim=-1)/1.5).clamp(.02,1.)[:,None]
        dist=torch.distributions.Normal(self.mean(obs),self.logstd.exp()*scale)
        action=dist.sample() if action is None else action
        return action,dist.log_prob(action).sum(-1),dist.entropy().sum(-1),self.critic(obs).squeeze(-1)


@torch.no_grad()
def validate(model,env,seed=8021):
    # Reproducible monitoring set; the final test set uses a different seed.
    gpu_state=torch.cuda.get_rng_state()
    torch.cuda.manual_seed(seed)
    env.reset(torch.ones(env.n,dtype=torch.bool,device='cuda'))
    done=torch.zeros(env.n,dtype=torch.bool,device='cuda');steps=torch.full((env.n,),200,device='cuda')
    bestp=torch.zeros(env.n,device='cuda');besto=bestp.clone()
    for i in range(200):
        obs,_,_=env.obs();action=model.mean(obs).clamp(-1,1)
        proposal=env.propose(action)
        env.q.copy_(torch.where(done[:,None],env.q,proposal));env.fk()
        _,pe,oe=env.obs();new=(pe<env.position_tol)&(oe<env.orientation_tol)&~done
        steps[new]=i+1;done|=new;bestp=pe;besto=oe
    torch.cuda.set_rng_state(gpu_state)
    return {'success':done.float().mean().item(),'success_5cm_0p2rad':((bestp<.05)&(besto<.2)).float().mean().item(),'success_3mm_1p5deg':((bestp<.003)&(besto<np.deg2rad(1.5))).float().mean().item(),'position_median_m':bestp.median().item(),'orientation_median_rad':besto.median().item(),'mean_iterations':steps.float().mean().item()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=20_000_000);ap.add_argument('--envs',type=int,default=2048);ap.add_argument('--out',default='runs/ppo_v1');ap.add_argument('--resume');ap.add_argument('--seed',type=int,default=7);ap.add_argument('--precision',action='store_true');ap.add_argument('--centered',action='store_true');args=ap.parse_args()
    from warp_env import WarpIKEnv
    if not torch.cuda.is_available():ap.error('Training requires an NVIDIA CUDA GPU; CPU inference is available through run.sh')
    if args.steps<=0 or args.envs<=0:ap.error('--steps and --envs must be positive')
    args.centered=args.precision and (args.centered or 'centered' in args.out)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(args.seed)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    env=WarpIKEnv(args.envs,args.seed,precision=args.precision,centered=args.centered);audit=env.audit()
    assert audit['max_position_error_m']<2e-6,audit
    model=ActorCritic(centered=args.centered).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-5)
    offset=0
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);offset=checkpoint['steps']
        if args.precision and not checkpoint['config'].get('precision',False):
            with torch.no_grad():
                for net in [model.actor,model.critic]:net[0].weight[:,20:23]/=20;net[0].weight[:,23:26]/=3
                model.logstd.fill_(-1.5);model.critic[-1].weight.zero_();model.critic[-1].bias.zero_()
            optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,eps=1e-5)
    horizon=32;batch=args.envs*horizon;mini=min(8192,batch);updates=math.ceil(args.steps/batch)
    config={'algorithm':'PPO','backend':'MuJoCo-Warp kinematics','source_reference':'reconstructed kinematic PPO; see docs/RL.md','exact_reference_source_available':False,'reward':'-2 atan(pos) - atan(ori) + exp(-pos/.05) + exp(-ori/.2) - .5 + 100 success','action':'q += .2 clip(action,-1,1), clipped to joint limits','observations':26,'envs':args.envs,'horizon':horizon,'epochs':4,'minibatch':mini,'seed':args.seed,'gamma':.99,'gae_lambda':.95,'target_kl':.03,'gpu_fraction_torch':.35,'fk_audit':audit,'initial_steps':offset,'requested_additional_steps':args.steps}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    config.update(precision=args.precision,centered=args.centered,position_tolerance=env.position_tol,orientation_tolerance=env.orientation_tol)
    if args.precision:config.update(reward='delta(log pose cost) - .2 cost - .1 + 100 success; cost=log1p((p/.003)^2)+.5 log1p((r/.02618)^2)',action='q += .2 clamp(max(p/.1,r/.5),.01,1) clip(action,-1,1)',observation_error_scales=[20,3],near_reset_fraction=.5,learning_rate_initial=5e-4)
    if args.centered:config.update(action='q += .2 clip(actor(obs)-actor(zero_error_obs),-1,1)',exploration='Gaussian std scaled by max(p/.1,r/.5), clamped .02..1')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    obs=env.obs()[0];shape=(horizon,args.envs)
    observations=torch.zeros((*shape,26),device='cuda');actions=torch.zeros((*shape,6),device='cuda')
    logprobs=torch.zeros(shape,device='cuda');rewards=torch.zeros(shape,device='cuda');dones=torch.zeros(shape,device='cuda');values=torch.zeros(shape,device='cuda');adv=torch.zeros(shape,device='cuda')
    valenv=WarpIKEnv(256,args.seed+111,precision=args.precision,near_fraction=0.,centered=args.centered)
    baseline=validate(model,valenv);print(json.dumps({'state':'baseline','validation':baseline,'audit':audit}),flush=True)
    best=-1;start=time.perf_counter();last_eval=offset-2_000_000
    def checkpoint(path,steps,validation=None):
        tmp=path.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'steps':steps,'config':config,'validation':validation},tmp);tmp.replace(path)
    for update in range(updates):
        rate=(5e-4 if args.precision else 1e-3)*(1-.8*update/max(updates,1));optimizer.param_groups[0]['lr']=rate
        success_count=torch.zeros((),device='cuda');end_count=success_count.clone()
        with torch.no_grad():
            for t in range(horizon):
                observations[t]=obs
                action,lp,_,value=model.act(obs)
                obs,r,done,success,_=env.step(action)
                actions[t]=action;logprobs[t]=lp;values[t]=value;rewards[t]=r;dones[t]=done
                success_count+=success.sum();end_count+=done.sum()
            last_value=model.critic(obs).squeeze(-1);last_gae=torch.zeros(args.envs,device='cuda')
            for t in reversed(range(horizon)):
                nv=last_value if t==horizon-1 else values[t+1]
                mask=1-dones[t];delta=rewards[t]+.99*nv*mask-values[t]
                last_gae=delta+.99*.95*mask*last_gae;adv[t]=last_gae
            returns=adv+values
        flat_obs=observations.reshape(-1,26);flat_actions=actions.reshape(-1,6);flat_lp=logprobs.flatten();flat_adv=adv.flatten();flat_ret=returns.flatten()
        flat_adv=(flat_adv-flat_adv.mean())/(flat_adv.std()+1e-8)
        stop=False
        for epoch in range(4):
            order=torch.randperm(batch,device='cuda')
            for idx in order.split(mini):
                _,lp,entropy,value=model.act(flat_obs[idx],flat_actions[idx]);logratio=lp-flat_lp[idx];ratio=logratio.exp()
                pg=torch.maximum(-flat_adv[idx]*ratio,-flat_adv[idx]*ratio.clamp(.8,1.2)).mean()
                loss=pg+.5*((value-flat_ret[idx])**2).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                kl=((ratio-1)-logratio).mean().item()
                if kl>.03: stop=True;break
            if stop: break
        steps=offset+(update+1)*batch
        if update%10==0:
            entry={'state':'training','steps':steps,'fps':(steps-offset)/(time.perf_counter()-start),'rollout_success':(success_count/end_count.clamp_min(1)).item(),'loss':loss.item(),'kl':kl,'gpu_allocated_mb':torch.cuda.memory_allocated()/2**20,'elapsed_s':time.perf_counter()-start}
            print(json.dumps(entry),flush=True)
            with (out/'train.jsonl').open('a') as file:file.write(json.dumps(entry)+'\n')
        if steps-last_eval>=2_000_000 or update==updates-1:
            result=validate(model,valenv);last_eval=steps
            record={'state':'validation','steps':steps,**result};print(json.dumps(record),flush=True)
            with (out/'validation.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
            checkpoint(out/'latest.pt',steps,result)
            if result['success']>best:best=result['success'];checkpoint(out/'best.pt',steps,result)
        # Yield the GPU regularly to the display compositor.
        time.sleep(.015)
    print(json.dumps({'state':'completed','steps':steps,'best_validation':best}),flush=True)

if __name__=='__main__':main()
