#!/usr/bin/env bash
# Stop the cable-tie BC run once the 50k checkpoint is on disk.
#
# The run was launched with num_train_steps=100000, but 50k is where we decided to
# stop. The LR is constant at 5e-5 after warmup (peak == decay), so stopping early
# leaves the model in the same state a 50k-configured run would have reached --
# there is no unfinished decay schedule to worry about.
#
# Orbax renames <step>.orbax-checkpoint-tmp-* to <step> only after the async save
# finalizes, so the final directory appearing means the checkpoint is complete. A
# grace period is still taken before signalling, and SIGTERM is preferred over
# SIGKILL so wandb can flush.
#
# Detached on purpose (setsid + nohup): this outlives the shell that started it.
set -u

PID="${1:?usage: stop_at_50k.sh <training-pid>}"
CKPT=/NHNHOME/WORKSPACE/gwanwoo/gwanwoo/ACRFT/checkpoints/pi05_yam_cable_tie/cable_tie_bc/50000
LOG=/NHNHOME/WORKSPACE/gwanwoo/gwanwoo/ACRFT/rounds/0817_cable_tie_bc/stop_at_50k.log

echo "$(date -Is) watching pid $PID for $CKPT" >>"$LOG"

while [ ! -d "$CKPT" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "$(date -Is) training pid $PID exited before 50k; nothing to stop" >>"$LOG"
    exit 0
  fi
  sleep 60
done

echo "$(date -Is) 50k checkpoint present, waiting 180s for finalize" >>"$LOG"
sleep 180

kill "$PID" 2>/dev/null
for _ in $(seq 1 30); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 2
done
if kill -0 "$PID" 2>/dev/null; then
  echo "$(date -Is) SIGTERM ignored, sending SIGKILL" >>"$LOG"
  kill -9 "$PID" 2>/dev/null
fi

echo "$(date -Is) stopped at 50k; checkpoint $(du -sh "$CKPT" | cut -f1)" >>"$LOG"
