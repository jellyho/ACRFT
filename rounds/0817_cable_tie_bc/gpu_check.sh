#!/usr/bin/env bash
# One-shot GPU health check. Distinguishes "driver alive but CUDA broken" (what we
# have) from "GPU truly gone" and from "fixed". Safe to run repeatedly.
echo "host      : $(hostname)"
echo "root      : $([ "$(id -u)" -eq 0 ] && echo yes || (sudo -n true 2>/dev/null && echo "via sudo" || echo no))"
echo "container : $(grep -qE 'docker|kubepods|containerd' /proc/1/cgroup 2>/dev/null && echo yes || echo "no / host")"
echo

echo -n "nvidia-smi        : "
nvidia-smi --query-gpu=name,memory.used,temperature.gpu --format=csv,noheader 2>&1 | head -1

echo -n "device nodes      : "
ls -d /dev/nvidia[0-9]* 2>/dev/null | tr '\n' ' '; echo

echo -n "GPUs in /proc     : "
ls /proc/driver/nvidia/gpus/ 2>/dev/null | wc -l

echo -n "CUDA cuInit()     : "
python3 - <<'PY' 2>/dev/null || echo "python3 unavailable"
import ctypes
try:
    cu = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print("cannot load libcuda:", e); raise SystemExit
r = cu.cuInit(0)
n = ctypes.c_int()
cu.cuDeviceGetCount(ctypes.byref(n))
meaning = {0: "OK", 3: "NOT_INITIALIZED (broken)", 100: "NO_DEVICE", 999: "UNKNOWN"}.get(r, "")
print(f"{r} {meaning}   visible devices: {n.value}")
PY

echo
if python3 -c "import ctypes,sys; sys.exit(0 if ctypes.CDLL('libcuda.so.1').cuInit(0)==0 else 1)" 2>/dev/null; then
  echo ">>> GPU IS USABLE — training and serving will work."
else
  echo ">>> CUDA IS BROKEN — nvidia-smi may still look fine. Needs a container restart"
  echo "    or host-side fix; nothing inside the container can repair this."
fi
