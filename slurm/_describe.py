"""Turn a run's config.json into a sentence a human can read.

Run names in the sweeps are abbreviations invented for the manifest — `mcf_hlg_upc`, `tn03`, `bc8`,
`g999`. They are fine as directory names and useless as documentation, and a hand-written table of
what they mean drifts from the configs the moment a flag changes. So the description is DERIVED from
config.json: it cannot go stale, and a run whose flags were set some other way still describes itself.

Framed as a diff against the baseline, because every sweep here is one-factor-at-a-time — what a
reader needs is "what is different about this one", not a dump of forty settings that are all shared.
"""

# (config key, baseline value, how to say it when it differs)
_AXES = [
    ("kind", "arq", lambda v: "QC — chunk 전체에 값 하나, prefix head 없음" if v == "qc" else f"kind={v}"),
    ("num_atoms", 51, lambda v: "스칼라 회귀 critic — 기본값인 HL-Gauss 히스토그램 대신 값 하나를 직접 회귀"
     if v == 1 else f"HL-Gauss atoms={v}"),
    ("v_agg", "max", lambda v: {"topm": "후보 집계를 상위 m개 평균으로 (hard max 대신)",
                                "soft": "후보 집계를 softmax 가중평균으로 (hard max 대신)"}.get(v, f"v_agg={v}")),
    ("ens_agg", "min", lambda v: "ensemble 집계를 mean−β·std(LCB)로 (hard min 대신)" if v == "lcb" else f"ens_agg={v}"),
    ("bootstrap_candidates", 0, lambda v: f"부트스트랩 max를 후보 {v}개로 좁힘 (매 스텝 재추출)"),
    ("target_noise", 0.0, lambda v: f"타깃 후보에 시간적으로 일관된 노이즈 {v}σ 주입"),
    ("macro_group_size", 2, lambda v: f"macro token {v}스텝 묶음 → prefix {16 // v}개 (기본 8개)"),
    ("mc_lower_bound", True, lambda v: "타깃 하한 없음 — 기본값인 max(TD, mc_return) 대신 순수 TD"
     if not v else None),
    ("proprio_mode", "concat", lambda v: "proprio를 전용 토큰으로 (토큰에 이어붙이지 않음)" if v == "token" else None),
]


def describe(cfg: dict) -> list[str]:
    """Bullet points for how this run differs from the sweep baseline. Empty list = it IS the baseline."""
    out = []
    for key, base, say in _AXES:
        v = cfg.get(key, base)
        if v == base:
            continue
        s = say(v)
        if s:
            out.append(s)
    # gamma lives in the dataset, not the flags, so it is read from the config's recorded discount.
    g = cfg.get("discount")
    if g is not None and abs(g - 0.99) > 1e-9:
        out.append(f"γ={g} 데이터셋 (기본 0.99) — mc_return과 value support가 함께 바뀜")
    return out


BASELINE_NOTE = (
    "ARQ 트랜스포머 critic · 후보 16개 전부에 hard max · ensemble min · 타깃 노이즈 없음 · "
    "HL-Gauss 분포형 Q (51 atoms) · 타깃 하한 max(TD, mc_return) · macro group 2 (prefix 8개) · "
    "proprio 포함(항상) · γ=0.99"
)
# v3까지는 스칼라 Q + 하한 없음이 baseline이었다. 두 축을 기본값으로 올린 것은 v3에서 이겼기
# 때문이 아니라 — v3는 모든 런이 action_sensitivity ~0.0003에 붙어 서열을 매길 수 없었다 —
# 더 방어 가능한 기본값이라서다. v3 런의 config.json은 그대로 두 축이 기록되어 있으므로
# 이 표를 v3에 적용하면 "스칼라 회귀"와 "하한 없음"이 차이점으로 표시되는데, 그게 사실이다.
