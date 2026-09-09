"""Annotate every YAM transition with Q(s, a_chunk), V(s), A = Q - V from a frozen patch critic.

Consumers and their provenance:
  AWR    -- weights w = exp(A/beta) on the flow-BC loss (Peng et al., arXiv 1910.00177, Eq. 8;
            the weighted-regression form; beta and weight clip set at train time).
  CFGRL  -- the hard optimality label O = 1{A > 0} exactly as the official value-based variant
            (kvfrans/cfgrl rlbase/algs_offline/iql_diffusion.py:157).
Chunk assembly mirrors scripts/score_critic_cached.py (clamp+hold past the episode end); the
critic preprocessing comes from its own input_spec via openpi.extraction.critic_q.
"""

# ruff: noqa: PLC0415  (jax imported after argparse for fast --help)

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--critic",
        type=pathlib.Path,
        default=pathlib.Path("/data5/jellyho/ACRFT/openpi/.scratch/patch_critic_yam_s347_fixed_tau9_min_200k"),
    )
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("/data1/jellyho/pc_cache/yam_s347"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", type=pathlib.Path, default=R / ".scratch/extraction/advantage_fixed_tau9min")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.extraction import critic_q

    critic = critic_q.load(a.critic)
    cache = critic_q.CacheView(a.cache)
    meta = cache.meta
    n = meta["N"]
    H = critic.config["horizon"]
    eps = sorted(((int(k), v["offset"], v["full_len"]) for k, v in meta["episodes"].items()))

    q_fn = jax.jit(critic.q_mean)
    v_fn = jax.jit(critic.v)

    Q = np.zeros(n, np.float32)
    V = np.zeros(n, np.float32)
    for e, off, ln in eps:
        # raw absolute chunks with clamp+hold at the episode end (score_critic_cached convention)
        acts = np.asarray(cache.actions[off : off + ln])
        states = np.asarray(cache.states[off : off + ln])
        pad = np.repeat(acts[-1:], H, axis=0)
        acts_ext = np.concatenate([acts, pad], axis=0)
        chunks = np.stack([acts_ext[t : t + H] for t in range(ln)])  # [ln, H, ad] raw absolute
        chunk_norm = critic.pre.actions(chunks, states)  # -> pi05-normalized joint delta (preproc.py:92-104)
        for s in range(0, ln, a.batch):
            sl = slice(off + s, off + min(s + a.batch, ln))
            f, _st, pr = cache.rows(np.arange(sl.start, sl.stop), critic)
            Q[sl] = np.asarray(q_fn(jnp.asarray(f), jnp.asarray(chunk_norm[s : s + a.batch]), jnp.asarray(pr)))
            V[sl] = np.asarray(v_fn(jnp.asarray(f), jnp.asarray(pr)))
        if e % 40 == 0:
            print(f"ep{e} done", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "q_data.npy", Q)
    np.save(a.out / "v_data.npy", V)
    adv = Q - V
    (a.out / "meta.json").write_text(
        json.dumps(
            {
                "critic": str(a.critic),
                "cache": str(a.cache),
                "n": n,
                "adv_mean": float(adv.mean()),
                "adv_std": float(adv.std()),
                "frac_positive": float((adv > 0).mean()),
            },
            indent=1,
        )
    )
    print(f"A: mean {adv.mean():.2f} std {adv.std():.2f} frac>0 {(adv > 0).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
