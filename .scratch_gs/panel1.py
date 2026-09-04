import gzip, json, pathlib
import numpy as np
b = json.loads(gzip.decompress(pathlib.Path("slurm/probes/guidance_sweep.json.gz").read_bytes()))
ch = np.asarray(b["chunks"], np.float32)
data = np.asarray(b["data"], np.float32)
alphas = b["alphas"]
JOINT = [f"L{i}" for i in range(1,7)] + ["Lgrip"] + [f"R{i}" for i in range(1,7)] + ["Rgrip"]
ref = ch[0]  # [A,H,AD]  panel 1 uses noise 0
moved = sorted(np.argsort(-np.abs(ref[-1]-ref[0]).max(axis=0))[:6])
print("panel-1 joints:", [JOINT[j] if j < len(JOINT) else f"dim{j}" for j in moved], moved)
H = ref.shape[1]
for j in moved:
    y = ref[:, :, j]           # [A,H]
    span = max(y.max(), data[:,j].max()) - min(y.min(), data[:,j].min())
    spread = y.max(0)-y.min(0)  # across alphas, per step
    print(f"\n{JOINT[j] if j<len(JOINT) else 'dim'+str(j)}  yspan={span:.2f}")
    print("  all-alpha spread/span %:", " ".join(f"{100*v/span:4.0f}" for v in spread))
    # deployed alphas only (0, .1, .2)
    idx=[alphas.index(x) for x in (0.0,0.1,0.2)]
    s2 = y[idx].max(0)-y[idx].min(0)
    print("  a<=0.2  spread/span %:", " ".join(f"{100*v/span:4.0f}" for v in s2))
