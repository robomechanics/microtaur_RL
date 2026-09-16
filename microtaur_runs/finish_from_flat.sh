#!/usr/bin/env bash
# After the from_flat queue ends: record the yaw video, then zip all four finished
# *_rough_from_flat runs (every checkpoint, params, onnx, tensorboard) plus their videos.
set -uo pipefail
R=/home/naomio/Documents/spines_help/microtaur_runs
S=$R/followup_logs_20260914_163536_from_flat/status.txt
cd "$R"
until grep -q "ALL DONE" "$S" || ! pgrep -f "train_yaw_pitch_followup.sh from_flat" > /dev/null; do sleep 30; done
grep -q "END active_yaw_rough_from_flat" "$S" || { echo "yaw did not finish cleanly:"; tail -3 "$S"; }

yaw_run=$(ls -d active_yaw/logs/rsl_rl/microtaur_minitaur_velocity/*_active_yaw_rough_from_flat)
if [ -f "$yaw_run/model_4499.pt" ]; then
  env -u PYTHONPATH PYTHONPATH=$R/active_yaw/src MICROTAUR_VARIANT=active_yaw_microtaur MICROTAUR_CONTROL_SPINE=1 \
    MUJOCO_GL=egl MICROTAUR_USE_JOYSTICK_COMMANDS=1 CUDA_VISIBLE_DEVICES=0 timeout 900 \
    /home/naomio/anaconda3/envs/microtaur/bin/python record_checkpoint.py "$yaw_run/model_4499.pt" \
    videos/active_yaw_rough_from_flat_model_4499.mp4 --clip-actions 1.0 > "$(dirname "$S")/record_active_yaw.log" 2>&1
  grep -a "wrote\|Traceback" "$(dirname "$S")/record_active_yaw.log"
fi

runs=()
for v in rigid active_twist active_pitch active_yaw; do
  d=$(ls -d $v/logs/rsl_rl/microtaur_minitaur_velocity/*_${v}_rough_from_flat)
  [ -f "$d/model_4499.pt" ] && runs+=("$d") || echo "skipping $v: no model_4499.pt"
done
rm -f rough_from_flat_policies.zip
zip -qr rough_from_flat_policies.zip "${runs[@]}" videos/*_rough_from_flat_model_4499.mp4 \
  make_flat_warmstart.py record_checkpoint.py followup_logs_20260914_163536_from_flat/status.txt
ls -la rough_from_flat_policies.zip; unzip -l rough_from_flat_policies.zip | tail -1
echo FINISHED
