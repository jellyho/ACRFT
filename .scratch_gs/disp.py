import gzip, json, pathlib
import numpy as np
b = json.loads(gzip.decompress(pathlib.Path("slurm/probes/guidance_sweep.json.gz").read_bytes()))
ch=np.asarray(b["chunks"],np.float32); s=float(b["sigma"])
for ai,al in enumerate(b["alphas"]):
    d=float(np.mean([np.sqrt(np.mean((ch[si,ai]-ch[si,0])**2))/s for si in range(ch.shape[0])]))
    print(f"alpha={al:<5} whole-chunk rms_sigma={d:.2f}")
