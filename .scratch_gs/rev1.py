import gzip, json, numpy as np
b = json.loads(gzip.decompress(open('/home/rllab4/jellyho/ACRFT-sweep/slurm/probes/guidance_sweep.json.gz','rb').read()))
ch = np.asarray(b["chunks"], np.float32)
sigma = float(b["sigma"])
alphas = b["alphas"]
print("alphas", alphas, "chunks", ch.shape, "sigma", sigma, "pc", b["pc_sigma"])
S,A,H,AD = ch.shape
for ai,al in enumerate(alphas):
    per = np.stack([[np.sqrt(np.mean((ch[si,ai,t]-ch[si,0,t])**2))/sigma for t in range(H)] for si in range(S)])
    m = per.mean(0)
    thirds = [m[0:10].mean(), m[10:20].mean(), m[20:30].mean()]
    print(f"alpha={al:<5} step0={m[0]:7.2f} max={m.max():7.2f}@{m.argmax():2d} thirds={thirds[0]:6.2f}/{thirds[1]:6.2f}/{thirds[2]:6.2f}  overall={np.sqrt(np.mean((ch[:,ai]-ch[:,0])**2))/sigma:6.2f}")
    print("   per-step:", " ".join(f"{v:.2f}" for v in m))
