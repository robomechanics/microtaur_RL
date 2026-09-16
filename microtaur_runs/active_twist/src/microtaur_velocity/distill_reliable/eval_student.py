from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from microtaur_velocity.distill_reliable.mjlab_utils import flatten_done,get_action_dim,get_actor_obs_tensor,get_initial_obs,make_env_only,read_velocity_metrics,step_env
from microtaur_velocity.distill_reliable.obs_prev_action import AlignedStudentObsBuilder
from microtaur_velocity.distill_reliable.policy import StudentPolicy


def load_student(checkpoint_path, device):
    ckpt=torch.load(checkpoint_path,map_location=device)
    student=StudentPolicy(obs_dim=int(ckpt.get("obs_dim",35)),action_dim=int(ckpt.get("action_dim",8)),hidden_dims=tuple(ckpt.get("hidden_dims",(64,64))),activation=str(ckpt.get("activation","elu")),output_tanh=True).to(device)
    student.load_state_dict(ckpt.get("model_state_dict",ckpt.get("state_dict",ckpt)))
    if "obs_mean" in ckpt and "obs_std" in ckpt: student.set_normalization(ckpt["obs_mean"].to(device),ckpt["obs_std"].to(device))
    return student.eval()


def main():
    p=argparse.ArgumentParser(); p.add_argument("task_id"); p.add_argument("--student-checkpoint",required=True); p.add_argument("--steps",type=int,default=5000); p.add_argument("--num-envs",type=int,default=256); p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu"); p.add_argument("--gait-freq-hz",type=float,default=1.45); p.add_argument("--log-every",type=int,default=500); p.add_argument("--out-json",default=None); p.add_argument("--clean-env",action="store_true"); p.add_argument("--no-terminations",action="store_true")
    args=p.parse_args(); robust=not args.clean_env; device=torch.device(args.device)
    env,_=make_env_only(args.task_id,args.num_envs,args.device,robust=robust,no_terminations=args.no_terminations); env_u=env.unwrapped
    action_dim=get_action_dim(env_u); student=load_student(args.student_checkpoint,device); builder=AlignedStudentObsBuilder(env_u.num_envs,env_u.device,action_dim,args.gait_freq_hz)
    if student.action_dim!=action_dim or student.obs_dim!=builder.obs_dim: raise RuntimeError(f"Checkpoint dims obs/action={student.obs_dim}/{student.action_dim}, env expects {builder.obs_dim}/{action_dim}")
    sums={}; count=0; resets=0; actor_obs=get_initial_obs(env)
    try:
        for step in range(args.steps):
            with torch.no_grad():
                s_obs=builder.build(get_actor_obs_tensor(actor_obs),env_u); action=torch.clamp(torch.nan_to_num(student(s_obs),nan=0.,posinf=1.,neginf=-1.),-1.,1.); actor_obs,_,done,_=step_env(env,action); done=flatten_done(done); resets+=int(done.sum().item())
            m=read_velocity_metrics(env_u); m["action_abs_mean"]=float(action.abs().mean());
            for k,v in m.items(): sums[k]=sums.get(k,0.)+float(v)
            count+=1
            if args.log_every>0 and step%args.log_every==0: print(f"[eval student] robust={robust} {step}/{args.steps} err={m.get('forward_error_abs_mean',0):.3f} yaw_err={m.get('yaw_error_abs_mean',0):.3f}")
    finally: env.close()
    summary={k:v/max(count,1) for k,v in sums.items()}; summary.update({"steps":args.steps,"num_envs":args.num_envs,"reset_count":resets,"robust_env":robust,"obs_dim":builder.obs_dim,"action_dim":action_dim})
    print(json.dumps(summary,indent=2))
    if args.out_json:
        path=Path(args.out_json); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(summary,indent=2),encoding="utf-8")

if __name__=="__main__": main()
