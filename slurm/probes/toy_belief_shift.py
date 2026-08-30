"""Belief-shift measurement: the chunk-regression leak, and that synthetic backup pins the belief.

Direct numerical test of docs/reactive_commitment_value.md Lemma 1 + Theorem 4. A one-step occluded
POMDP where the closed-loop demo's chunk reveals the latent, so conditioning a value on the chunk
shifts the latent posterior (the leak). We compute, at the occluded observation:

  * the true belief             b(s | o)                       -- what a value should integrate
  * the chunk-shifted belief    b(s | o, chunk=a) ∝ b(s|o) β(a|s)
  * Q_reg  = E_data[G | o, chunk]      (chunk-return regression)  -> uses b(s|o,chunk)  [LEAK]
  * Q_syn  = 1-step synthetic backup   (marginalize o' over o,a1) -> uses b(s|o)         [HONEST]
  * V_exec(s, chunk) closed form, so the two estimators can be checked against the belief identity.

Prediction: Q_reg = E_{b(s|o,chunk)}[V_exec] (optimistic by I(s;chunk|o)); Q_syn = E_{b(s|o)}[V_exec];
their gap equals the belief-shift bias, and it is exactly the measured leak. No RL, no seeds needed for
the identities -- but we add sampled seeds to show the estimators converge to them from finite data.
"""

# ruff: noqa: N802, N806  (V_exec, S, R are math symbols)
import argparse
import json
import pathlib

import numpy as np

# One-step occluded POMDP.
#   latent s in {A, B}, prior 1/2 each. Observation o = 'dark' for BOTH (occlusion).
#   chunk = a single action a in {0, 1}.
#   correct action: A->0, B->1. reward 1 if a == correct(s) else 0. (then terminal)
#   demo (closed-loop) knows s: beta(a|s) = 1-eps on the correct action, eps on the wrong one.
EPS = 0.1  # demo error


def correct(s):
    return 0 if s == "A" else 1


def V_exec(s, a):
    return 1.0 if a == correct(s) else 0.0


def beta(a, s):
    return (1 - EPS) if a == correct(s) else EPS


def true_belief():
    return {"A": 0.5, "B": 0.5}


def chunk_belief(a):
    """b(s | o, chunk=a) ∝ b(s|o) β(a|s)."""
    w = {s: 0.5 * beta(a, s) for s in ("A", "B")}
    z = sum(w.values())
    return {s: w[s] / z for s in w}


def analytic():
    """The identities, in closed form."""
    b = true_belief()
    out = {}
    for a in (0, 1):
        bc = chunk_belief(a)
        q_reg = sum(bc[s] * V_exec(s, a) for s in ("A", "B"))  # E_{b(s|o,chunk)}[V_exec]
        q_syn = sum(b[s] * V_exec(s, a) for s in ("A", "B"))  # E_{b(s|o)}[V_exec]
        out[a] = {
            "b_true": b,
            "b_chunk": bc,
            "Q_reg": q_reg,
            "Q_syn": q_syn,
            "leak": q_reg - q_syn,
            "I_bias": q_reg - q_syn,
        }
    return out


def sampled(rng, n):
    """Finite-data estimators, to show convergence to the identities.

    Data = demo episodes: draw s, draw a ~ beta(.|s), reward = V_exec(s,a). Observation is 'dark'.
    Q_reg(a) = mean reward among demo episodes whose chunk == a  (chunk-return regression).
    Q_syn(a) = E over the belief of V_exec: since it is one step, the synthetic backup at 'dark' with a
               FIXED a marginalizes s over b(s|dark) = the data's latent frequency at 'dark' (uniform),
               i.e. Q_syn(a) = mean over ALL demo latents of V_exec(s, a), NOT conditioned on chunk==a.
    """
    S = rng.choice(["A", "B"], size=n)
    Achosen = np.array([0 if (correct(s) == 0) == (rng.random() > EPS) else 1 for s in S])
    # equivalently: a = correct(s) w.p. 1-eps
    Achosen = np.array([correct(s) if rng.random() > EPS else 1 - correct(s) for s in S])
    R = np.array([V_exec(s, a) for s, a in zip(S, Achosen, strict=True)])
    out = {}
    for a in (0, 1):
        mask = Achosen == a
        q_reg = float(R[mask].mean()) if mask.any() else float("nan")  # conditions on chunk==a  [LEAK]
        # synthetic: value of committing action a, belief = all latents at 'dark' (marginal), NOT chunk-cond
        q_syn = float(np.mean([V_exec(s, a) for s in S]))
        out[a] = {"Q_reg": q_reg, "Q_syn": q_syn, "leak": q_reg - q_syn}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".scratch/toy_belief_shift.json"))
    a = ap.parse_args()

    an = analytic()
    print(f"belief-shift toy | occluded 1-step POMDP | demo error eps={EPS}\n")
    print("ANALYTIC identities (Lemma 1 + Theorem 4):")
    for act in (0, 1):
        r = an[act]
        print(
            f"  commit a={act}:  b(s|o)={{{r['b_true']['A']:.2f},{r['b_true']['B']:.2f}}}  "
            f"b(s|o,chunk)={{{r['b_chunk']['A']:.2f},{r['b_chunk']['B']:.2f}}}  "
            f"Q_reg={r['Q_reg']:.3f}  Q_syn={r['Q_syn']:.3f}  leak={r['leak']:+.3f}"
        )
    # the leak is the belief-shift bias = I(s;chunk|o) cashed out in value units
    print("\n  => Q_reg conditions on chunk -> posterior shifts to the state where the chunk is correct")
    print("     -> optimistic (leak > 0). Q_syn keeps b(s|o)=uniform -> honest 0.5. This IS the V(s0) gap.")

    per_seed = [sampled(np.random.default_rng(s), a.n) for s in range(a.seeds)]

    def agg(act, field):
        v = np.array([ps[act][field] for ps in per_seed])
        return v.mean(), v.std()

    print(f"\nSAMPLED estimators ({a.seeds} seeds, n={a.n}) converge to the identities:")
    for act in (0, 1):
        rm, rs = agg(act, "Q_reg")
        sm, ss = agg(act, "Q_syn")
        lm, ls = agg(act, "leak")
        print(
            f"  a={act}:  Q_reg={rm:.3f}±{rs:.3f}  Q_syn={sm:.3f}±{ss:.3f}  leak={lm:+.3f}±{ls:.3f}  (analytic leak {an[act]['leak']:+.3f})"
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"analytic": an, "per_seed": per_seed, "eps": EPS}, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
