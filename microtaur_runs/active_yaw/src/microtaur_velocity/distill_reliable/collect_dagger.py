from __future__ import annotations
import argparse
from pathlib import Path
import torch
from microtaur_velocity.distill_reliable.dataset import DistillChunkWriter
from microtaur_velocity.distill_reliable.eval_student import load_student
from microtaur_velocity.distill_reliable.mjlab_utils import get_action_dim,get_actor_obs_tensor,get_initial_obs,make_env_and_teacher,read_velocity_metrics,step_env
from microtaur_velocity.distill_reliable.obs_prev_action import AlignedStudentObsBuilder


def main():
    p=argparse.ArgumentParser(); p.add_argument("task_id"); p.add_argument("--teacher-checkpoint",required=True); p.add_argument("--student-checkpoint",required=True); p.add_argument("--out-dir",default="logs/distill_reliable/dagger_robust")
    p.add_argument("--beta",type=float,default=.5); p.add_argument("--steps",type=int,default=50000); p.add_argument("--chunk-steps",type=int,default=500); p.add_argument("--num-envs",type=int,default=1024); p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu"); p.add_argument("--gait-freq-hz",type=float,default=1.45); p.add_argument("--student-action-noise-std",type=float,default=0.0); p.add_argument("--action-clip",type=float,default=1.0); p.add_argument("--log-every",type=int,default=1000); p.add_argument("--clean-env",action="store_true"); p.add_argument("--no-terminations",action="store_true")
    args=p.parse_args(); beta=max(0.,min(1.,args.beta)); robust=not args.clean_env; device=torch.device(args.device)
    env,_,teacher=make_env_and_teacher(args.task_id,args.teacher_checkpoint,args.num_envs,args.device,robust=robust,no_terminations=args.no_terminations); env_u=env.unwrapped
    student=load_student(args.student_checkpoint,device); action_dim=get_action_dim(env_u)
    if student.action_dim!=action_dim: raise RuntimeError(f"Student action_dim={student.action_dim}, env action_dim={action_dim}")
    builder=AlignedStudentObsBuilder(env_u.num_envs,env_u.device,action_dim,args.gait_freq_hz)
    if student.obs_dim!=builder.obs_dim: raise RuntimeError(f"Student obs_dim={student.obs_dim}, expected {builder.obs_dim}")
    writer=DistillChunkWriter(args.out_dir,f"dagger_beta{int(beta*1000):03d}",args.chunk_steps,builder.layout,metadata={"mode":"dagger_aligned_robust" if robust else "dagger_aligned_clean","task_id":args.task_id,"teacher_checkpoint":str(Path(args.teacher_checkpoint)),"student_checkpoint":str(Path(args.student_checkpoint)),"beta":beta,"robust_env":robust,"action_dim":action_dim,"obs_dim":builder.obs_dim},action_dim=action_dim)
    actor_obs=get_initial_obs(env)
    try:
        for step in range(args.steps):
            with torch.no_grad():
                s_obs=builder.build(get_actor_obs_tensor(actor_obs),env_u); s_act=student(s_obs)
                if args.student_action_noise_std>0: s_act=s_act+args.student_action_noise_std*torch.randn_like(s_act)
                s_act=torch.clamp(torch.nan_to_num(s_act,nan=0.,posinf=args.action_clip,neginf=-args.action_clip),-args.action_clip,args.action_clip)
                t_act=torch.clamp(torch.nan_to_num(teacher(actor_obs),nan=0.,posinf=args.action_clip,neginf=-args.action_clip),-args.action_clip,args.action_clip)
                executed=torch.clamp(beta*t_act+(1-beta)*s_act,-args.action_clip,args.action_clip)
                writer.record(s_obs,t_act,executed); actor_obs,_,_,_=step_env(env,executed)
            if args.log_every>0 and step%args.log_every==0:
                m=read_velocity_metrics(env_u); print(f"[dagger] {step}/{args.steps} beta={beta:.2f} err={m.get('forward_error_abs_mean',0):.3f} yaw_err={m.get('yaw_error_abs_mean',0):.3f}")
    finally:
        writer.close(); env.close()

if __name__=="__main__": main()
