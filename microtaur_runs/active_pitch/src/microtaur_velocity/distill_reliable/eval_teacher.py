from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from microtaur_velocity.distill_reliable.mjlab_utils import flatten_done,get_initial_obs,make_env_and_teacher,read_velocity_metrics,step_env


def main():
    p=argparse.ArgumentParser(); p.add_argument("task_id"); p.add_argument("--checkpoint-file",required=True); p.add_argument("--steps",type=int,default=5000); p.add_argument("--num-envs",type=int,default=256); p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu"); p.add_argument("--log-every",type=int,default=500); p.add_argument("--out-json",default=None); p.add_argument("--clean-env",action="store_true"); p.add_argument("--no-terminations",action="store_true")
    args=p.parse_args(); robust=not args.clean_env
    env,_,teacher=make_env_and_teacher(args.task_id,args.checkpoint_file,args.num_envs,args.device,robust=robust,no_terminations=args.no_terminations); env_u=env.unwrapped; obs=get_initial_obs(env); sums={}; count=0; resets=0
    try:
        for step in range(args.steps):
            with torch.no_grad():
                action=torch.clamp(torch.nan_to_num(teacher(obs),nan=0.,posinf=1.,neginf=-1.),-1.,1.); obs,_,done,_=step_env(env,action); done=flatten_done(done); resets+=int(done.sum().item())
            m=read_velocity_metrics(env_u); m["action_abs_mean"]=float(action.abs().mean());
            for k,v in m.items(): sums[k]=sums.get(k,0.)+float(v)
            count+=1
            if args.log_every>0 and step%args.log_every==0: print(f"[eval teacher] robust={robust} {step}/{args.steps} err={m.get('forward_error_abs_mean',0):.3f} yaw_err={m.get('yaw_error_abs_mean',0):.3f}")
    finally: env.close()
    summary={k:v/max(count,1) for k,v in sums.items()}; summary.update({"steps":args.steps,"num_envs":args.num_envs,"reset_count":resets,"robust_env":robust}); print(json.dumps(summary,indent=2))
    if args.out_json:
        path=Path(args.out_json); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(summary,indent=2),encoding="utf-8")

if __name__=="__main__": main()
