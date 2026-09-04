import gzip, json, numpy as np
b = json.loads(gzip.decompress(open('/home/rllab4/jellyho/ACRFT-sweep/slurm/probes/guidance_sweep.json.gz','rb').read()))
ch = np.asarray(b["chunks"], np.float32); bc = np.asarray(b["bc"], np.float32)
sigma=float(b["sigma"]); alphas=b["alphas"]
print("bc shape", bc.shape, "per-dim std:", np.round(bc.std(0).mean(0),4))
S,A,H,AD = ch.shape
# restrict to real 14 dims
ch14 = ch[...,:14]; sigma14 = float(bc[...,:14].std(0).mean())
print("sigma all32", sigma, "sigma14", sigma14)
for ai,al in enumerate(alphas):
    m = np.stack([[np.sqrt(np.mean((ch14[si,ai,t]-ch14[si,0,t])**2))/sigma14 for t in range(H)] for si in range(S)]).mean(0)
    print(f"a={al:<5} (14 dims) step0={m[0]:6.2f} max={m.max():6.2f}@{m.argmax():2d} thirds={m[:10].mean():6.2f}/{m[10:20].mean():6.2f}/{m[20:].mean():6.2f}")
# also _disp values
def disp(ai):
    return float(np.mean([np.sqrt(np.mean((ch[si,ai]-ch[si,0])**2))/sigma for si in range(S)]))
print("d01", disp(2), "d02", disp(3), "-> rounded", f"{disp(2):.0f}~{disp(3):.0f}")
