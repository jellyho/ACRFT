import gzip, json, numpy as np
b = json.loads(gzip.decompress(open('/home/rllab4/jellyho/ACRFT-sweep/slurm/probes/guidance_sweep.json.gz','rb').read()))
ch = np.asarray(b["chunks"], np.float32)
alphas = b["alphas"]; sigma=float(b["sigma"])
ref = ch[0]
JOINT_NAMES = [f"L{i}" for i in range(1,7)] + ["Lgrip"] + [f"R{i}" for i in range(1,7)] + ["Rgrip"]
joints=6
moved = sorted(np.argsort(-np.abs(ref[-1]-ref[0]).max(axis=0))[:joints])
print("selected joints:", moved, [JOINT_NAMES[j] if j < 14 else f"pad{j}" for j in moved])
print("alpha=1.6 range on those joints:", [(float(ref[-1,:,j].min()), float(ref[-1,:,j].max())) for j in moved])
for j in moved:
    lo = min(ref[:,:,j].min(), np.asarray(b["data"],np.float32)[:,j].min())
    hi = max(ref[:,:,j].max(), np.asarray(b["data"],np.float32)[:,j].max())
    print(f"\njoint {JOINT_NAMES[j] if j<14 else 'pad%d'%j} axis span ~ [{lo:.2f},{hi:.2f}] (span {hi-lo:.2f})")
    for ai,al in enumerate(alphas):
        d = ref[ai,:,j]-ref[0,:,j]
        frac = np.abs(d)/(hi-lo)
        print(f"  a={al:<5} |dev| step0={abs(d[0]):.3f} mean0-19={np.abs(d[:20]).mean():.3f} mean20-29={np.abs(d[20:]).mean():.3f} max={np.abs(d).max():.3f}@{np.abs(d).argmax()} | as frac of axis: head {frac[:20].mean():.3f} tail {frac[20:].mean():.3f}")
