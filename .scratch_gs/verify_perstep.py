import gzip, json, pathlib
import numpy as np
p = pathlib.Path("slurm/probes/guidance_sweep.json.gz")
b = json.loads(gzip.decompress(p.read_bytes()))
ch = np.asarray(b["chunks"], np.float32)  # [S,A,H,AD]
sigma = float(b["sigma"])
alphas = b["alphas"]
S,A,H,AD = ch.shape
print("shape", ch.shape, "sigma", sigma, "alphas", alphas)
print()
for ai, al in enumerate(alphas):
    per = np.stack([[np.sqrt(np.mean((ch[si,ai,t]-ch[si,0,t])**2))/sigma for t in range(H)] for si in range(S)])
    m = per.mean(0)
    thirds = [m[0:10].mean(), m[10:20].mean(), m[20:30].mean()]
    print(f"alpha={al:<5} step0={m[0]:7.2f} peak={m.max():7.2f}@{m.argmax():2d} thirds={thirds[0]:7.2f}/{thirds[1]:7.2f}/{thirds[2]:7.2f} last={m[-1]:7.2f}")
print()
# full per-step for the deployed alphas
for al in (0.1,0.2,0.4,0.8,1.6):
    ai = alphas.index(al)
    per = np.stack([[np.sqrt(np.mean((ch[si,ai,t]-ch[si,0,t])**2))/sigma for t in range(H)] for si in range(S)])
    m = per.mean(0)
    print(f"alpha={al}:", " ".join(f"{v:.1f}" for v in m))
