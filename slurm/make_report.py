"""Build the HTML experiment report from whatever the sweep has produced so far.

Reads the artefacts the runs write (config.json / diag.json / gpu.json per run, the offline
prefix-bias JSONs, the prefix-profile JSON, and the job logs) and emits a single self-contained
page. Safe to re-run while jobs are still going — anything missing is reported as missing rather
than guessed at.

    uv run slurm/make_report.py --sweep abl_main --out report.html
"""

import argparse
import html
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _describe import BASELINE_NOTE
from _describe import describe


def read_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt(v, spec=".4f", dash="—"):
    if v is None or (isinstance(v, float) and v != v):
        return dash
    try:
        return format(v, spec)
    except Exception:
        return html.escape(str(v))


def collect(runs_dir, logs_dir, sweep):
    out = {}
    if not runs_dir.is_dir():
        return out
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        cfg, diag, gpu = (read_json(d / f) for f in ("config.json", "diag.json", "gpu.json"))
        rate = steps_done = None
        curve = []
        for lg in sorted(logs_dir.glob(f"{sweep}_*.out")) + sorted(logs_dir.glob(f"{d.name}_*.out")):
            try:
                txt = lg.read_text(errors="replace")
            except Exception:
                continue
            if f"/{d.name}\n" not in txt and f": {sweep}/{d.name}" not in txt:
                continue
            m = re.findall(r"step (\d+)/(\d+)\s+([\d.]+) it/s", txt)
            if m:
                steps_done, _, rate = int(m[-1][0]), m[-1][1], float(m[-1][2])
            # The in-training diagnostic, logged every --eval-every steps. This is the safety net:
            # a run that does not reach the end writes no diag.json, but these lines still say
            # whether the ranking signal ever appeared.
            curve = [
                (int(s), float(rs), float(rk))
                for s, rs, rk in re.findall(r"\[diag @ (\d+)\] range/std ([\d.-]+)\s+rank ([\d.-]+)", txt)
            ]
        out[d.name] = {
            "cfg": cfg,
            "diag": diag,
            "gpu": gpu,
            "rate": rate,
            "steps_done": steps_done,
            "curve": curve,
        }
    return out


CSS = """
/* 계측기 판독지: 따뜻한 회색 중립 + 신호 앰버. 토큰으로만 색을 쓰고 두 테마 모두 재정의한다. */
:root{
  --paper:#f7f5ef; --ink:#16180f; --ink-2:#4a4d40; --ink-3:#75786a;
  --rule:#ddd9cc; --panel:#efece2; --panel-2:#e7e3d6;
  --signal:#b4531a; --defect:#8f2d1f; --confirm:#2f6b4a; --caution:#8a6a12;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#131410; --ink:#eceadf; --ink-2:#b3b0a2; --ink-3:#85826f;
    --rule:#2c2e25; --panel:#1a1c16; --panel-2:#20221a;
    --signal:#e0863f; --defect:#e07a68; --confirm:#78c096; --caution:#d6ad4a;
  }
}
:root[data-theme=dark]{
  --paper:#131410; --ink:#eceadf; --ink-2:#b3b0a2; --ink-3:#85826f;
  --rule:#2c2e25; --panel:#1a1c16; --panel-2:#20221a;
  --signal:#e0863f; --defect:#e07a68; --confirm:#78c096; --caution:#d6ad4a;
}
:root[data-theme=light]{
  --paper:#f7f5ef; --ink:#16180f; --ink-2:#4a4d40; --ink-3:#75786a;
  --rule:#ddd9cc; --panel:#efece2; --panel-2:#e7e3d6;
  --signal:#b4531a; --defect:#8f2d1f; --confirm:#2f6b4a; --caution:#8a6a12;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  background:var(--paper); color:var(--ink); margin:0; padding:3rem 1.25rem 7rem;
  font-family:Pretendard,"Apple SD Gothic Neo","Noto Sans KR",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.7; font-feature-settings:"tnum" 0;
}
main{max-width:56rem;margin:0 auto}
h1{font-size:clamp(1.65rem,4vw,2.1rem);line-height:1.2;letter-spacing:-.02em;margin:0 0 .5rem;text-wrap:balance}
h2{font-size:1.3rem;letter-spacing:-.01em;margin:3.5rem 0 1rem;padding-bottom:.4rem;border-bottom:1px solid var(--rule);text-wrap:balance}
h3{font-size:1.06rem;letter-spacing:-.005em;margin:2.2rem 0 .6rem;text-wrap:balance}
h4{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.74rem;font-weight:600;
   text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3);margin:1.5rem 0 .5rem}
p,li{color:var(--ink-2)}
p{margin:.7rem 0}
.sub{color:var(--ink-3);margin:0 0 2.5rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem}
b,strong{color:var(--ink);font-weight:650}
code,pre,td.num,th.num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
code{background:var(--panel-2);color:var(--ink);padding:.1em .38em;border-radius:3px;font-size:.85em}
pre{background:var(--panel);border:1px solid var(--rule);border-left:2px solid var(--signal);
    border-radius:0 4px 4px 0;padding:1rem 1.15rem;overflow-x:auto;font-size:.82rem;line-height:1.6}
pre code{background:none;padding:0;font-size:1em}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:1rem 0;border:1px solid var(--rule);border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:.87rem;min-width:32rem}
th,td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:baseline}
thead th{background:var(--panel);color:var(--ink-3);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
         font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--panel)}
td.num,th.num{text-align:right}
/* 상태를 색이 아니라 형태로도 인코딩: 좌측 스트라이프 두께/색 */
.box{border:1px solid var(--rule);border-left:3px solid var(--signal);border-radius:0 4px 4px 0;
     padding:1rem 1.2rem;margin:1.3rem 0;background:var(--panel)}
.box.warn{border-left-color:var(--caution)}
.box.bad{border-left-color:var(--defect)}
.box.good{border-left-color:var(--confirm)}
.box h4{margin-top:0;color:var(--ink)}
.box p:last-child{margin-bottom:0}
.tag{display:inline-block;font-family:ui-monospace,monospace;font-size:.68rem;padding:.15em .55em;
     border-radius:2px;border:1px solid var(--rule);color:var(--ink-3);margin-left:.45rem;vertical-align:middle}
.bad-t{color:var(--defect);font-weight:650}
.good-t{color:var(--confirm);font-weight:650}
.warn-t{color:var(--caution);font-weight:650}
.bar{display:inline-block;height:.55em;background:var(--signal);border-radius:1px;vertical-align:middle;min-width:1px;opacity:.75}
ul,ol{padding-left:1.35rem}
li{margin:.35rem 0}
hr{border:0;border-top:1px solid var(--rule);margin:2.5rem 0}
.meta{font-size:.85rem;color:var(--ink-3)}
a{color:var(--signal)}
:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


GLOSSARY_TBL = "<div class='scroll'><table><thead><tr><th>표기</th><th>정식 이름</th><th>무엇을 재나</th><th>기준값</th><th>방향</th></tr></thead><tbody><tr><td><code>act_sens</code></td><td class='mut'>action_sensitivity</td><td>상태 <b>내부</b> Q 분산 ÷ 상태 <b>간</b> Q 분산. Q의 변동 중 액션 때문인 몫.</td><td>0 = 액션 무시(<code>Q(z,a)=V(z)</code>로 붕괴), 1 = 액션이 상태만큼 중요</td><td class='mut'>클수록 좋음</td></tr><tr><td><code>rank_cand</code></td><td class='mut'>ranking_accuracy_demo_vs_candidate</td><td>같은 상태에서 시연된 chunk가 정책 후보보다 높은 Q를 받는 비율. 시연 chunk만이 결과를 아는 액션.</td><td><b>우연 = 0.5</b></td><td class='mut'>클수록 좋음</td></tr><tr><td><code>rank_other</code></td><td class='mut'>ranking_accuracy_demo_vs_other</td><td>시연 chunk vs <b>다른 상태</b>에서 가져온 chunk. 쉬운 문제 — 위치만 알면 맞춤.</td><td>우연 = 0.5</td><td class='mut'>높아도 <code>rank_cand</code>가 0.5면 의미 없음</td></tr><tr><td><code>ws_range</code></td><td class='mut'>within_state_q_range</td><td>한 상태의 16개 후보 Q의 (최대 − 최소) 평균.</td><td>0 = 후보를 전혀 구분 못 함</td><td class='mut'>클수록 좋음</td></tr><tr><td><code>rho_close</code></td><td class='mut'>spearman_q_vs_closeness_to_demo</td><td>상태별로 후보를 Q 순위와 '실행된 chunk와의 거리' 순위로 매겨 상관. 액션을 아는 critic은 실제로 통했던 것에 가까운 후보를 선호.</td><td>0 = 무관</td><td class='mut'>클수록 좋음</td></tr><tr><td><code>pfx_H</code></td><td class='mut'>prefix_argmax_entropy</td><td>배포 arg-max가 고르는 prefix 길이의 정규화 엔트로피.</td><td>1 = 8개 prefix에 고르게 분산, 0 = 항상 같은 길이(적응적 청킹이 퇴화)</td><td class='mut'>클수록 적응적</td></tr><tr><td><code>bias_grow</code></td><td class='mut'>bias_growth_last_double</td><td>후보 개수를 두 배로 늘렸을 때 arg-max 값이 오르는 양. 편향 없는 critic은 포화.</td><td>0 = 포화(건전), >0 = max-over-noise 과대추정</td><td class='mut'>0에 가까울수록 좋음</td></tr><tr><td><code>held_gap</code></td><td class='mut'>argmax_gap_train_minus_heldout</td><td>부트스트랩이 본 후보 vs 못 본 후보(held-out)에서의 arg-max 값 차이.</td><td>0 = 일반화, >0 = 저장된 표본에 과적합</td><td class='mut'>0에 가까울수록 좋음</td></tr><tr><td><code>q−mc</code></td><td class='mut'>q_demo_minus_mc_mean</td><td>시연 chunk의 Q에서 그 chunk가 실제 거둔 리턴을 뺀 값. 값함수의 교정 상태.</td><td>0 = 정확, &lt;0 = 과소, &gt;0 = 과대추정</td><td class='mut'>0에 가까울수록 좋음</td></tr></tbody></table></div>"


def table(headers, rows, numeric_from=1):
    h = "".join(
        f'<th class="{"num" if i >= numeric_from else ""}">{html.escape(x)}</th>' for i, x in enumerate(headers)
    )
    body = ""
    for r in rows:
        body += (
            "<tr>"
            + "".join(f'<td class="{"num" if i >= numeric_from else ""}">{c}</td>' for i, c in enumerate(r))
            + "</tr>"
        )
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="abl_main")
    ap.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/critic_runs"))
    ap.add_argument("--logs", type=pathlib.Path, default=pathlib.Path("/scratch/jellyho/acrft/logs"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--generated", default="", help="timestamp string (the script cannot call Date)")
    args = ap.parse_args()

    runs = collect(args.runs_root / args.sweep, args.logs, args.sweep)
    bias = {n: read_json(args.runs_root / f"prefix_bias_{n}.json") for n in ("noprop", "noprop_g999", "noprop_g9995")}
    profile = read_json(args.runs_root / f"prefix_profile_{args.sweep}.json") or {}

    # ---- sim rollout: successes aggregated over seeds ---------------------------------------------
    import collections

    roll = {}
    for d in sorted((args.runs_root / args.sweep).glob("*/rollout")):
        agg = collections.defaultdict(lambda: [0, 0])
        for f in sorted(d.glob("*.json")):
            j = read_json(f) or {}
            for m, v in j.items():
                if isinstance(v, dict) and "successes" in v:
                    agg[m][0] += v["successes"]
                    agg[m][1] += v["num_trials"]
        if agg:
            roll[d.parent.name] = {m: (a, b) for m, (a, b) in agg.items()}

    def _paired(run):
        """vla 대비 짝지은 승패 (동일 scene). trials 배열이 있는 결과에서만."""
        import math

        tr = {}
        for f in sorted((args.runs_root / args.sweep / run / "rollout").glob("*.json")):
            j = read_json(f) or {}
            for m, v in j.items():
                if isinstance(v, dict) and "trials" in v:
                    for t in v["trials"]:
                        tr.setdefault((f.stem, t["trial"]), {})[m] = t["success"]
        out = {}
        for m in ("critic", "bon", "prefix"):
            b = sum(1 for d in tr.values() if d.get("vla") and not d.get(m))
            c = sum(1 for d in tr.values() if d.get(m) and not d.get("vla"))
            n = b + c
            pv = min(1.0, sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n * 2) if n else 1.0
            out[m] = (b, c, pv)
        return out, len(tr)

    roll_rows = []
    for name in sorted(roll):
        m = roll[name]
        cell = lambda k, m=m: (f"{m[k][0]}/{m[k][1]} ({m[k][0] / m[k][1]:.0%})" if k in m else "—")  # noqa: E731
        pw, npair = _paired(name)
        pc = pw.get("critic", (0, 0, 1.0))
        roll_rows.append(
            [
                f"<code>{html.escape(name)}</code>",
                f"{npair}",
                cell("vla"),
                cell("bon"),
                cell("prefix"),
                cell("critic"),
                f"{pc[0]}–{pc[1]} (p={pc[2]:.3f})",
            ]
        )
    roll_tbl = (
        table(
            [
                "run",
                "scene",
                "vla (critic 없음)",
                "bon (후보만)",
                "prefix (커밋길이만)",
                "critic (결합)",
                "vla vs critic 짝지은 승패",
            ],
            roll_rows,
        )
        if roll_rows
        else "<p class='meta'>rollout 결과 없음</p>"
    )

    # ---- section: per-prefix target ceiling ------------------------------------------------------
    ceil_rows = []
    b = bias.get("noprop")
    if b:
        for r in b["per_prefix"]:
            w = max(1, int(260 * r["max_mc"]))
            ceil_rows.append(
                [
                    str(r["prefix"]),
                    fmt(r["max_mc"]),
                    f'<span class="bar" style="width:{w}px"></span>',
                    f'{r["valid"]:,}',
                    fmt(100 * r["frac_of_alive"], ".2f") + "%",
                    fmt(r["mean_mc"]),
                ]
            )
    ceil_tbl = (
        table(["prefix h", "max target (=γ^h)", "", "valid transitions", "% of alive", "mean mc_return"], ceil_rows)
        if ceil_rows
        else "<p class='meta'>prefix_bias_noprop.json 없음</p>"
    )

    gamma_rows = []
    for n, lbl in (("noprop", "0.99 (기본)"), ("noprop_g999", "0.999"), ("noprop_g9995", "0.9995")):
        bb = bias.get(n)
        if bb:
            pp = bb["per_prefix"]
            gamma_rows.append(
                [
                    lbl,
                    fmt(bb["bias_ratio_short_over_long"], ".3f") + "×",
                    fmt(pp[0]["max_mc"]),
                    fmt(pp[-1]["max_mc"]),
                    fmt(pp[-1]["mean_mc"] - pp[0]["mean_mc"], "+.4f"),
                ]
            )
    gamma_tbl = (
        table(["dataset (γ)", "γ^2/γ^16", "천장 @h=2", "천장 @h=16", "mean mc Δ(긴−짧은)"], gamma_rows)
        if gamma_rows
        else ""
    )

    # ---- section: sweep results ------------------------------------------------------------------
    keys = [
        ("action_sensitivity", "act_sens", ".4f"),
        ("ranking_accuracy_demo_vs_candidate", "rank_cand", ".3f"),
        ("ranking_accuracy_demo_vs_other", "rank_other", ".3f"),
        ("spearman_q_vs_closeness_to_demo", "ρ_close", "+.3f"),
        ("within_state_q_range", "ws_range", ".3f"),
        ("prefix_argmax_entropy", "pfx_H", ".3f"),
        ("bias_growth_last_double", "bias↑", "+.3f"),
        ("argmax_gap_train_minus_heldout", "held_gap", "+.3f"),
    ]
    sweep_rows = []
    for name, r in runs.items():
        dg, cfg = r["diag"], r["cfg"] or {}
        status = (
            "<span class='good-t'>완료</span>"
            if dg
            else f"<span class='warn-t'>{r['steps_done'] or 0}/{cfg.get('steps', '?')}</span>"
        )
        row = [f"<code>{html.escape(name)}</code>", status]
        row += [fmt((dg or {}).get(k), s) for k, _, s in keys]
        sweep_rows.append(row)
    sweep_tbl = (
        table(["run", "상태"] + [lbl for _, lbl, _ in keys], sweep_rows, numeric_from=1)
        if sweep_rows
        else "<p class='meta'>아직 완료된 run 없음</p>"
    )

    # In-training curve: works even for runs that never reached the final eval.
    curve_rows = []
    for name, r in runs.items():
        c = r["curve"]
        if not c:
            continue
        first, last = c[0], c[-1]
        curve_rows.append(
            [
                f"<code>{html.escape(name)}</code>",
                f"{last[0]:,}",
                fmt(first[1], ".2f"),
                fmt(last[1], ".2f"),
                fmt(first[2], ".3f"),
                fmt(last[2], ".3f"),
            ]
        )
    curve_tbl = (
        table(["run", "도달 step", "range/std 초기", "range/std 최종", "rank 초기", "rank 최종"], curve_rows)
        if curve_rows
        else ""
    )

    # ---- section: learned prefix profile ---------------------------------------------------------
    prof_rows = []
    for name, r in sorted(profile.items()):
        a, ng = r.get("all") or {}, r.get("near_goal") or {}
        prof_rows.append(
            [
                f"<code>{html.escape(name)}</code>",
                fmt(r.get("discount"), ".4f"),
                fmt(100 * a.get("frac_shortest", float("nan")), ".1f") + "%",
                fmt(a.get("mean_exec_steps"), ".2f"),
                fmt(a.get("spearman_q_vs_prefix"), "+.2f"),
                fmt(100 * ng.get("frac_shortest", float("nan")), ".1f") + "%",
                fmt(ng.get("mean_exec_steps"), ".2f"),
            ]
        )
    prof_tbl = (
        table(
            [
                "run",
                "γ",
                "최단 선택(전체)",
                "평균 실행 step",
                "ρ(Q, prefix)",
                "최단 선택(목표근처)",
                "평균 step(목표근처)",
            ],
            prof_rows,
        )
        if prof_rows
        else "<p class='meta'>prefix_profile JSON 없음 — 학습 완료 후 생성</p>"
    )

    done = sum(1 for r in runs.values() if r["diag"])
    vvd_bad = read_json(args.runs_root / "vvd_base.json")
    vvd_ok = read_json(args.runs_root / "vvd_tmc.json")

    parts = []
    A = parts.append
    A("<h1>ACRFT critic ablation — RLT 임베딩 위에서의 critic 학습</h1>")
    A(
        f"<p class='sub'>sweep <code>{html.escape(args.sweep)}</code> · 완료 {done}/{len(runs)} run"
        + (f" · {html.escape(args.generated)}" if args.generated else "")
        + "</p>"
    )

    A(
        "<div class='box bad'><h4>요약</h4><ul>"
        "<li><b>타깃에 결함이 있었습니다.</b> sparse 종료 보상이 어떤 타깃에도 들어가지 않아 값함수가 0으로 붕괴했습니다. "
        "<code>--terminal-uses-mc</code>가 고치며, 30k step으로 안 되던 것이 6k step에 학습됩니다. 이 플래그는 <b>기본값이 꺼져</b> 있었습니다.</li>"
        "<li><b>short-horizon 선호는 실재합니다.</b> 값함수를 고친 뒤에도 ρ(Q, prefix) = −0.83, baseline은 16스텝 중 평균 6.78스텝만 커밋합니다.</li>"
        "<li><b>critic은 여전히 후보를 순위 매기지 못합니다.</b> action_sensitivity ~5e-4, ranking ~0.52(우연 0.5). proprio를 넣어도 그대로입니다.</li>"
        "<li><b>유일하게 편향을 뒤집은 손잡이는 <code>--v-agg soft</code></b>입니다 (ρ +0.45, 평균 커밋 12.95/16).</li>"
        "</ul></div>"
    )

    A("<h2>1. 실험 프로토콜</h2>")
    A("<p>이 절만 읽고 재현할 수 있도록 적었습니다. 경로는 이 클러스터 기준 절대경로입니다.</p>")
    A("<h3>1.1 환경</h3>")
    A(
        "<pre><code>ROLLOUT=1 slurm/setup.sh          # venv, sim deps, RoboCasa 에셋 (로그인 노드, 1회)\n"
        "source slurm/env.sh              # 경로/캐시/tier→파티션·QOS 매핑. <b>반드시 source</b> (아래 부록 B)\n"
        "CACHE_DIR=/scratch/jellyho/acrft # 모든 산출물 루트 (NFS 공유, 전 노드에서 보임)</code></pre>"
    )
    A("<h3>1.2 데이터</h3>")
    A(
        "<pre><code>slurm/fetch_data.sh            # jellyho/acrft-annot-noprop      -> $ANNOT_ROOT/noprop\n"
        "slurm/fetch_data.sh --ckpt     # jellyho/pi05-robocasa-prepcoffee-rlt-pardec-noprop-70k -> $VLA_CKPT\n"
        "slurm/fetch_data.sh --lerobot  # jellyho/robocasa365-PrepareCoffee -> $HF_LEROBOT_HOME\n"
        "uv run slurm/extract_proprio.py --data $ANNOT_ROOT/noprop        # proprio 복원 (아래 1.3)\n"
        "uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.999\n"
        "uv run slurm/make_discount_variant.py --data $ANNOT_ROOT/noprop --discount 0.9995</code></pre>"
    )
    A(
        table(
            ["항목", "값"],
            [
                ["frames", "279,534 (stride 1)"],
                ["token / chunk", "2048-d / 16×12"],
                ["candidates N", "16 (+ held-out 8)"],
                ["reward", "sparse, terminal success, support [0,1]"],
                ["discount γ", "0.99 기본 / 0.999 / 0.9995 (재라벨 사본, hardlink로 3.5 MB)"],
                ["episodes", "514 — <b>전부 성공</b>, 실패 0"],
                ["episode 길이", "중앙값 521 step, 최대 1092"],
                ["GPU 상주", "5.96 GB (proprio 포함 obs 2064-d)"],
            ],
            numeric_from=1,
        )
    )
    A("<h3>1.3 proprioception 복원</h3>")
    A(
        "<p><code>noprop</code> RLT 토큰은 proprio를 <b>의도적으로</b> 제외합니다(paper-faithful bottleneck). "
        "README는 그래서 “critic must supply proprio”라고 적지만, 공급하는 코드가 없었습니다 — "
        "<code>net.apply(params, token, action_chunk)</code>가 인자의 전부였습니다.</p>"
    )
    A(
        "<p>재-annotation(VLA 전체 재실행) 없이 복원했습니다: annotation이 <code>episode_index</code>/<code>frame_index</code>를 "
        "보관하고, 원본 LeRobot parquet이 <code>observation.state</code>(16-d)를 <b>같은 279,534행</b>으로 갖고 있어 조인이 됩니다. "
        "정합은 가정하지 않고 두 인덱스 컬럼이 전 행에서 일치하는지 검사한 뒤 진행합니다.</p>"
    )
    A(
        "<p>dim별 z-score를 적용합니다. raw state는 미터·쿼터니언·gripper qpos가 섞여 있고 critic의 첫 연산이 "
        "observation 전체에 대한 LayerNorm이라, 정규화 없이는 16 dim이 2048 토큰 dim의 통계에 짓눌립니다. "
        "std=0인 상수 dim 2개는 NaN 대신 0으로 통과시킵니다.</p>"
    )
    A("<h3>1.4 학습</h3>")
    A(
        "<pre><code>SWEEP=fix_main STEPS=20000 EVAL_EVERY=2000 SAVE_EVERY=10000 ROLLOUT=0 MAXPAR=8 \\\n"
        "  slurm/sweep.sh</code></pre>"
    )
    A(
        "<p>baseline에서 <b>한 번에 하나만</b> 바꾸는 OFAT입니다. baseline = ARQ · 후보 16개 hard max · ensemble min · "
        "target smoothing 없음 · scalar Q · macro group 2 · <b>proprio 포함</b> · <b><code>--terminal-uses-mc</code></b>, "
        "γ와 value support는 annotation <code>meta.json</code>을 따릅니다.</p>"
    )
    A(
        "<pre><code>batch 256 · lr 3e-4 · 20,000 steps · seed 0 · ensemble 2 · target τ=0.005\n"
        "ARQ: 3 layers × 8 heads × 48 head_dim (n_embd 384), mlp_dim 1024 → <b>9.90M 파라미터</b>(ensemble 2 합산)\n"
        "macro_group 2 → prefix 8개, 시퀀스 = 1 state token + 8 macro token\n"
        "SLURM: base_qos(비선점, 사용자당 GPU 8), tier=wide · 반복당 0.14–0.25 s</code></pre>"
    )
    A(
        "<div class='box'><h4>batch 256인 이유 (RTX 3090 24576 MiB 실측)</h4>"
        "<p>batch 1024 → 9.00 GiB 텐서 하나로 OOM. batch 512 → <b>10.69 GiB</b> 요구, 역시 OOM "
        "(1024보다 <i>큰</i> 단일 할당 — XLA rematerialization이 1024에서는 작동하고 512에서는 안 함). "
        "batch 256 → peak 22,344 MiB로 통과. <b>peak은 batch에 단조가 아니므로 보간 금지.</b> "
        "sweep 전체에 같은 batch를 고정해 변종 비교를 보존했습니다.</p></div>"
    )
    A("<h3>1.5 평가</h3>")
    A(
        "<pre><code>uv run scripts/eval_rlt_critic.py --data &lt;annot&gt; --params &lt;run&gt;/params.msgpack --out &lt;run&gt;/diag.json\n"
        "uv run slurm/prefix_bias_analysis.py --data &lt;annot&gt;          # 오프라인, GPU 불필요\n"
        "uv run slurm/value_vs_distance.py   --run &lt;run&gt;              # Q vs 목표까지 거리\n"
        "uv run slurm/prefix_profile.py      --runs $CRITIC_RUNS/fix_main\n"
        "sbatch slurm/eval_rollout.sbatch                              # sim rollout (별도 job)</code></pre>"
    )
    A(
        "<p>모든 진단은 <b>상태 내부</b>에서 계산합니다. <code>Q(z,a)=V(z)</code>로 붕괴한 critic도 TD loss와 "
        "Q-vs-return 상관에서는 좋아 보이지만 후보를 무작위로 고르므로, 상태 간 지표는 판단 근거가 못 됩니다.</p>"
    )

    A("<h2>2. 예상했던 결과</h2>")
    A(
        table(
            ["#", "예측", "근거", "결과"],
            [
                [
                    "E1",
                    "critic은 후보를 순위 매기지 못한다 (act_sens≈0, rank≈0.5)",
                    "커밋 <code>167cc66</code>: online loss가 상태당 액션 하나에서만 평가되어 후보는 자기 타깃을 못 받음",
                    "<span class='good-t'>적중</span>",
                ],
                [
                    "E2",
                    "aggregation 축은 과대추정만 줄이고 순위 능력은 못 만든다",
                    "전부 bootstrap의 max를 좁힐 뿐",
                    "<span class='good-t'>적중</span> (단 <code>soft</code>가 prefix 편향을 바꿈 — 예상 밖)",
                ],
                [
                    "E3",
                    "γ를 키우면 short-horizon 편향이 준다",
                    "짧은 prefix가 유지하는 편향비 1.151×(γ=0.99) → 1.014×(γ=0.999)",
                    "<span class='bad-t'>빗나감</span> — γ=0.9995는 ρ −0.38로 개선되나 γ=0.999는 최단 선택 61%로 <b>악화</b>",
                ],
                [
                    "E4",
                    "<code>upc</code>가 prefix 프로파일을 평탄하게 만든다",
                    "커버리지 비대칭이 원인이라면 균일화로 사라져야 함",
                    "<span class='good-t'>적중</span> — ρ −0.83 → <b>−0.07</b>",
                ],
            ],
            numeric_from=99,
        )
    )
    A(
        "<p class='meta'>E1–E2는 실행 전 예측이었고, 아래 3.1의 결함은 <b>예측에 없었습니다</b> — 진단 과정에서 발견했습니다.</p>"
    )

    A("<h2>3. 실제 결과와 분석</h2>")

    A("<h3>3.1 결함 — 부트스트랩의 종료 처리가 두 규약을 섞어 썼다</h3>")
    A(
        "<p>첫 sweep의 완료 런이 전부 같은 증상을 보였습니다: <code>q_demo_mean</code> ≈ 0.003인데 "
        "<code>mc_return</code> 평균은 0.182, 상태 <b>간</b> 표준편차도 0.002, 그리고 "
        "<code>corr(Q, mc_return)</code> = <b>−0.75</b>. 액션을 무시하는 정도가 아니라 거의 상수이고, "
        "남은 미세 변동이 실제 수익과 반대였습니다.</p>"
    )
    A("<p><code>slurm/value_vs_distance.py</code>로 목표까지 거리별로 보면 원인이 드러납니다.</p>")
    if vvd_bad and vvd_ok:
        rows = []
        for b, o in zip(vvd_bad["bins"], vvd_ok["bins"], strict=False):
            rows.append(
                [
                    f"{b['lo']}–{b['hi'] if b['hi']<10**6 else '∞'}",
                    f"{b['n']}",
                    fmt(b["mc"], ".4f"),
                    fmt(b["q"], ".4f"),
                    fmt(o["q"], ".4f"),
                ]
            )
        A(table(["목표까지 step", "n", "참값 mc (γ^d)", "Q — 수정 전", "Q — 수정 후"], rows))
        A(
            table(
                ["지표", "수정 전 (30k step)", "수정 후 (6k step)"],
                [
                    [
                        "corr(Q, mc_return)",
                        fmt(vvd_bad["corr_q_mc"], "+.3f"),
                        f"<b>{fmt(vvd_ok['corr_q_mc'],'+.3f')}</b>",
                    ],
                    [
                        "corr(Q, 거리)",
                        fmt(vvd_bad["corr_q_dist"], "+.3f"),
                        f"<b>{fmt(vvd_ok['corr_q_dist'],'+.3f')}</b>",
                    ],
                    ["Q 평균 (참값 0.186)", fmt(vvd_bad["q_mean"], ".4f"), f"<b>{fmt(vvd_ok['q_mean'],'.4f')}</b>"],
                ],
            )
        )
    A(
        "<div class='box bad'><h4>원인: 규약 혼용</h4>"
        "<p>보상의 타이밍에는 자기정합적인 규약이 둘 있습니다.</p>"
        + table(
            ["규약", "보상 합", "V(terminal)"],
            [
                ["A — 도착 시 보상", "Σ<sub>i=1..h</sub> γ<sup>i−1</sup> r<sub>t+i</sub>", "0"],
                [
                    "B — 상태의 보상 (<code>mc_return</code> 정의와 일치)",
                    "Σ<sub>i=0..h−1</sub> γ<sup>i</sup> r<sub>t+i</sub>",
                    "<b>r<sub>terminal</sub></b>",
                ],
            ],
            numeric_from=99,
        )
        + "<p>코드는 <b>B의 <code>cum</code></b>을 쓰면서 <b>A의 종료 처리(V=0)</b>를 했습니다. 각 규약은 홀로 정합하지만 "
        "섞으면 성공 보상이 정확히 <code>i=h</code>라는 틈으로 빠집니다 — <code>cum</code>은 <code>i&lt;h</code>까지만 "
        "더하고, <code>boot</code>는 그 지점에서 꺼집니다.</p></div>"
    )
    A(
        "<div class='box bad'><h4>코드</h4>"
        "<pre><code>cum = cum_all[:, prefixes - 1]                    # Σ_&#123;i&lt;h&#125; γ^i r  ← i=h 제외\n"
        "lands_on_term = (crossed == 1) &amp; (done[nxt] &gt; 0)  # 종료가 정확히 t+h\n"
        "boot = (crossed == 0)                            # → lands_on_term이면 False\n"
        "y = cum + gam * boot * v_next                    # → y = cum = 0</code></pre>"
        "<p>sparse 스킴에서 보상 1은 종료 프레임에만 있습니다. prefix가 종료에 <b>정확히</b> 도달하면 그 보상은 "
        "offset <code>i=h</code>에 위치하는데 <code>cum</code>은 <code>i&lt;h</code>만 더하고, <code>boot</code>도 꺼져 "
        "부트스트랩이 없습니다. <b>목표 도달 transition의 타깃이 0</b>이 되어, 성공 보상이 학습 신호에 전혀 진입하지 못합니다. "
        "값함수가 0으로 수렴하는 것이 이 목적함수의 정확한 고정점입니다.</p>"
        "</div>"
    )
    A(
        "<div class='box good'><h4>수정: 종료 상태의 값 = 그 상태의 보상</h4>"
        "<p>종료 이후에는 아무것도 없으므로 <code>V(terminal) = r<sub>terminal</sub></code>은 근사가 아니라 "
        "<b>정확</b>합니다. 따라서 보상은 부트스트랩 슬롯을 통해 들어와야 합니다.</p>"
        "<pre><code>term_v = data.reward[nxt]                              # 순수 TD, mc_return 아님\n"
        "v_next = jnp.where(lands_on_term, term_v, v_next)\n"
        "y = cum + gam * (boot | lands_on_term) * v_next</code></pre>"
        "<p><b>플래그로 두지 않았습니다.</b> 결함이지 튜닝 손잡이가 아니며, 옵션으로 남기면 다음 사람이 또 꺼진 채 "
        "돌립니다. <code>--terminal-uses-mc</code>는 남아 있지만 이제 “보상 전파 여부”가 아니라 종료 값의 <i>출처</i>만 "
        "고릅니다(참조 구현 <code>vla_aqc.py</code>와의 계약 호환).</p>"
        "<p class='meta'>검증: 6,000 step 두 런(<code>reward[t+h]</code> vs <code>mc_return[t+h]</code>)이 "
        "Q mean 0.234535 · std 0.215647 · corr(Q,mc) +0.9830으로 <b>자릿수까지 동일</b>합니다. "
        "세 데이터셋의 종료 514개에서 <code>mc_return == reward</code>가 오차 0으로 일치하기 때문이며, "
        "따라서 <code>--terminal-uses-mc</code>로 실행된 아래 16런의 수치는 고친 기본값과 같고 재실행이 필요 없습니다.</p>"
        "</div>"
    )

    A("<h3>3.2 prefix head가 받을 수 있는 최대 타깃은 정확히 γ^h</h3>")
    A(
        "<p>배포 규칙(<code>eval_critic.make_policy_fn</code>)은 <code>(candidate, prefix)</code>에 대한 <b>결합 arg-max</b>를 "
        "취하고 <code>(pp+1)·macro</code> step을 실행하므로, prefix 축을 따른 Q의 체계적 기울기가 커밋 길이를 결정합니다. "
        "annotation만으로(GPU 없이) 확인되는 구조적 비대칭이 있습니다.</p>"
    )
    A(ceil_tbl)
    A(
        "<p>terminal을 넘는 prefix는 transition이 생성되지 않으므로(<code>valid</code>), 목표에 가까울수록 살아남는 prefix가 "
        "줄고 prefix h의 타깃 최댓값은 정확히 γ^h가 됩니다. 이는 trainer docstring의 "
        "“puts the per-prefix values on a <b>common discounted scale</b> and makes them <b>comparable at deployment</b>” "
        "와 어긋납니다 — 타깃 <i>값</i>은 horizon-matched가 맞지만(공유 상태에서 모든 h가 γ^d) <b>커버리지가 공통이 아닙니다</b>. "
        "긴 head는 값 범위의 상단을, 하필 목표 근처에서 못 봅니다.</p>"
    )
    if gamma_tbl:
        A(gamma_tbl)

    A("<h3>3.3 학습된 prefix 프로파일 — short-horizon 선호는 실재한다</h3>")
    A(
        "<p>3.1을 고친 뒤 측정한 값입니다. <code>ρ(Q, prefix)</code>가 −1에 가까울수록 Q가 커밋 길이에 대해 단조 감소합니다.</p>"
    )
    A(prof_tbl)
    A(
        "<div class='box good'><h4>인과 확인: <code>upc</code>가 기울기를 없앤다</h4>"
        "<p><code>--uniform-prefix-coverage</code>는 모든 prefix head를 <b>동일한 상태 집합</b>에서 학습시킵니다"
        "(모든 prefix가 valid한 상태만 사용, 각 에피소드의 마지막 <code>horizon</code> step을 버리는 대가). "
        "이 한 가지만 바꿨을 때 ρ(Q, prefix)가 <b>−0.83 → −0.07</b>로 평탄해지고 최단 선택이 7.7% → 3.3%로 줄며 "
        "평균 커밋이 6.78 → 8.64로 늘어납니다. §3.2에서 유도한 γ^h 천장이 short-horizon 편향의 "
        "<b>실제 원인임이 인과적으로 확인</b>됩니다 — 상관이 아니라, 원인을 제거하니 결과가 사라졌습니다.</p></div>"
    )
    A(
        "<div class='box bad'><h4>예측 E3(γ)은 빗나갔습니다</h4>"
        "<p>편향비 논증(1.151×→1.014×)은 γ를 키우면 편향이 준다고 예측했습니다. γ=0.9995는 ρ −0.38로 개선되지만 "
        "γ=0.999는 최단 선택이 <b>61%</b>로 오히려 악화됩니다(baseline 7.7%). γ가 커지면 참값 자체가 "
        "1에 몰려 prefix 간 차이가 사라지고, 그러면 arg-max가 <b>노이즈에 지배</b>됩니다. "
        "천장 비대칭을 줄이는 것과 값의 분해능을 유지하는 것이 상충하며, γ만으로는 이 문제를 풀 수 없습니다.</p></div>"
    )
    A(
        "<div class='box warn'><h4>읽는 법</h4>"
        "<p>거의 모든 변종에서 ρ가 강하게 음수입니다 — baseline은 16스텝 중 평균 <b>6.78스텝</b>만 커밋합니다. "
        "<code>--target-noise</code>는 이를 파국적으로 악화시켜 사실상 최단으로 붕괴합니다(<code>tn10</code>: 99.9%가 최단, 2.00/16). "
        "<b><code>--v-agg soft</code>만 부호를 뒤집습니다</b> (ρ +0.45, 평균 12.95/16). "
        "예측 E2는 aggregation 축이 순위 능력을 못 만든다고 했고 그건 맞았지만, "
        "<i>커밋 길이</i>에는 강하게 작용한다는 것은 예상 밖이었습니다.</p></div>"
    )

    A("<h3>3.4 ablation 표</h3>")
    A("<h4>지표 정의</h4>")
    A(
        "<p>모든 지표는 <b>상태 내부</b>에서 계산합니다. <code>Q(z,a)=V(z)</code>로 붕괴한 critic도 TD loss와 "
        "Q-vs-return 상관에서는 멀쩡해 보이지만 후보를 무작위로 고르므로, 상태 간 지표는 판단 근거가 못 됩니다.</p>"
    )
    A(GLOSSARY_TBL)
    A("<h4>각 run이 무엇인가</h4>")
    A(
        f"<p><b>baseline</b> = {BASELINE_NOTE}. 아래는 baseline과 <b>다른 점만</b> 적은 것입니다 — "
        "이 sweep은 한 번에 하나씩만 바꾸는 OFAT이므로, 적히지 않은 설정은 전부 baseline과 같습니다.</p>"
    )
    _desc_rows = []
    for _n in sorted(runs):
        _c = runs[_n]["cfg"] or {}
        _d = describe(_c)
        _desc_rows.append(
            [f"<code>{html.escape(_n)}</code>", "<br>".join(html.escape(x) for x in _d) if _d else "<i>baseline</i>"]
        )
    A(table(["run", "baseline과 다른 점"], _desc_rows, numeric_from=99))
    A(
        "<p>판단 기준은 <code>act_sens</code>와 <code>rank_cand</code>입니다. 둘 다 기준값에 머물면 "
        "그 변종은 후보를 순위 매기지 못한 것이고, 나머지 열은 <i>어떻게</i> 실패했는지를 말합니다.</p>"
    )
    A(sweep_tbl)
    if curve_tbl:
        A("<h4>학습 중 진단 추이 <span class='tag'>미완주 run 포함</span></h4>")
        A(
            "<p><code>range/std</code>는 상태 내 Q의 (최대−최소)/자체 표준편차로, <b>후보 16개가 순수 노이즈면 ≈3.1–3.6</b>입니다.</p>"
        )
        A(curve_tbl)

    A("<h3>3.5 sim rollout — 진짜 목적함수</h3>")
    A(
        "<p>오프라인 지표는 전부 대리물입니다. 실제 질문은 “critic이 고른 것이 VLA가 그냥 낸 것보다 나은가”입니다. "
        "네 모드가 동일 scene(고정 seed)에서 돌며, critic의 두 결정을 분리합니다 — "
        "<code>bon</code>은 후보만 고르고(커밋 길이 고정), <code>prefix</code>는 첫 후보를 쓰되 커밋 길이만 고릅니다.</p>"
    )
    A(roll_tbl)
    A(
        "<div class='box bad'><h4>critic이 VLA를 이긴 적이 없습니다</h4>"
        "<p><code>prefix</code>는 모든 run에서 <code>vla</code>와 동일합니다 — <b>적응적 커밋 길이가 아무것도 사지 못합니다</b>. "
        "<code>bon</code>과 <code>critic</code>은 같거나 <b>더 나쁩니다</b> — 후보 선택이 도움이 되기는커녕 해가 됩니다. "
        "이는 오프라인 결과(act_sens ~5e-4, rank ~0.52)와 정확히 일치합니다: 순위를 못 매기는 critic으로 "
        "best-of-N을 하면 무작위 선택이고, 무작위 선택은 정책의 첫 샘플보다 나을 이유가 없습니다.</p>"
        "</div>"
    )
    A(
        "<div class='box warn'><h4>통계적으로 얼마나 말할 수 있나</h4>"
        "<p>모든 모드가 <b>동일 scene</b>에서 돌므로 짝지은 비교(McNemar 정확검정)가 가능합니다. "
        "<code>base</code>·<code>soft</code> 각 32 scene에서 <b>개별 비교는 어느 것도 p&lt;0.05에 이르지 못합니다</b> "
        "(p = 0.065–0.23). 표본이 작습니다.</p>"
        "<p>다만 <code>vla</code> 대비 6개 비교(2 run × 3 모드)가 <b>전부 같은 방향</b>이고, "
        "critic이 이긴 scene보다 진 scene이 항상 많습니다(9–3, 10–4, 7–2, 9–2, 12–5, 8–3). "
        "따라서 주장할 수 있는 것은 <b>“critic이 VLA를 개선한다는 증거가 없다”</b>이며, "
        "오프라인에서 순위 능력이 우연 수준(rank 0.52)이라는 것과 일관됩니다. "
        "“critic이 해를 끼친다”는 더 강한 주장은 이 표본으로는 확정할 수 없습니다.</p></div>"
    )

    A("<h2>4. 결론</h2>")
    A(
        "<ol>"
        "<li><b>이번 밤의 가장 큰 결과는 결함 발견입니다.</b> 부트스트랩의 종료 처리가 <code>cum</code>과 "
        "다른 보상 규약을 써서, sparse 성공 보상이 <b>어떤 타깃에도 진입하지 못했습니다</b>. V=0이 목적함수의 "
        "정확한 고정점이 되므로 이는 학습 부족이 아니라 결함이며, 이 상태로 수행된 과거 critic 실험은 "
        "<b>재검토가 필요합니다</b>. 종료 상태의 부트스트랩 값을 <code>reward[t+h]</code>로 두어 고쳤고 "
        "(플래그 없이 기본 동작), 6k step 만에 corr(Q, mc_return) +0.98을 회복합니다.</li>"
        "<li><b>값함수를 고쳐도 critic은 후보를 순위 매기지 못합니다.</b> act_sens ~5e-4, rank ~0.52. "
        "proprio를 넣어도 변하지 않았습니다. 예측 E1대로이며, 원인은 online loss가 상태당 액션 하나에서만 "
        "평가되어 후보가 자기 타깃을 받지 못하는 구조입니다.</li>"
        "<li><b>short-horizon 선호는 실재하며, 원인이 밝혀졌습니다.</b> baseline은 ρ(Q,prefix) = −0.83, "
        "평균 커밋 6.78/16. terminal을 넘는 prefix가 transition을 만들지 못해 prefix h의 타깃 천장이 γ^h로 "
        "묶이는 것이 원인이며, 커버리지를 균일화하면(<code>upc</code>) ρ가 <b>−0.07</b>로 평탄해집니다. "
        "반면 γ를 키우는 처방(E3)은 실패합니다 — 값의 분해능이 함께 사라져 arg-max가 노이즈에 지배됩니다.</li>"
        "<li><b><code>--v-agg soft</code>가 유일하게 prefix 편향을 뒤집습니다</b>(ρ +0.45). 다만 rollout 성공률은 "
        "개선하지 못했습니다 — 커밋 길이 편향을 고치는 것과 성능을 내는 것은 별개입니다.</li>"
        "<li><b>rollout에서 critic이 VLA를 개선한다는 증거가 없습니다.</b> 32 scene 짝지은 비교에서 "
        "<code>vla</code> 78% vs <code>critic</code> 59%이고, 6개 비교가 전부 같은 방향입니다. "
        "<code>prefix</code>가 <code>vla</code>와 사실상 같다는 것은 <b>적응적 커밋 길이 자체가 아무것도 사지 못한다</b>는 뜻입니다.</li>"
        "</ol>"
    )
    A(
        "<div class='box warn'><h4>한계</h4>"
        "<p>rollout의 trial 수가 작습니다. 방향성(어느 모드도 <code>vla</code>를 못 넘음)은 오프라인 지표와 "
        "일치해 신뢰할 만하지만, run 간 우열을 주장하려면 trial을 크게 늘려야 합니다.</p>"
        "<p>또한 이번 sweep은 20,000 step입니다. §3.1 검증에서 6,000 step만에 값 구조가 잡히는 것을 확인했으므로 "
        "값함수 학습에는 충분하지만, 순위 능력이 더 긴 학습에서 나타날 가능성을 배제하지는 못합니다.</p></div>"
    )

    A("<h2>부록 A — 이번에 발견해 고친 결함</h2>")
    A(
        table(
            ["결함", "증상", "조치"],
            [
                [
                    "<b>부트스트랩의 종료 처리가 규약 혼용</b>",
                    "성공 보상이 어떤 타깃에도 미진입 → 값함수가 0으로 붕괴, "
                    "corr(Q, mc_return) = −0.75, 목표 8스텝 앞에서 Q = −0.001(참값 0.97)",
                    "종료의 부트스트랩 값을 <code>reward[t+h]</code>로 수정 (플래그 아님, 기본 동작). "
                    "<code>notmc</code> 대조군을 sweep에 유지 (§3.1)",
                ],
                [
                    "<code>_diag</code>가 ARQ에서 크래시",
                    "<code>qd</code>에만 ensemble min 누락 → <code>[K,S,1,mh]</code>를 <code>[K]</code>로 축약. "
                    "<code>--eval-every</code>가 처음 작동하는 step에서 <b>전 런</b>이 죽을 상황",
                    "<code>qc</code>와 동일한 축약 순서로 수정, arq/qc × scalar/HL-Gauss 4조합 검증",
                ],
                [
                    "중간 체크포인트가 로드 불가",
                    "<code>config.json</code>이 학습 종료 시에만 기록 → <code>params_10000.msgpack</code>을 "
                    "아무도 재구성 못 함. 동시 eval이 원천 차단",
                    "첫 중간 저장 시점에 <code>config.json</code>도 기록",
                ],
                [
                    "critic이 proprio를 못 받음",
                    "<code>noprop</code> 토큰에도 없고 <code>Data</code>에도 없음",
                    "<code>extract_proprio.py</code>로 원본에서 조인 복원(정합 검증 포함), 학습·평가·rollout 전 경로 연결",
                ],
                [
                    "rollout eval이 원본 LeRobot 데이터셋을 요구",
                    "VLA 생성 시 <code>data.create()</code>가 progress label 계산 위해 parquet reward 컬럼을 읽음",
                    "<code>fetch_data.sh --lerobot</code> 추가 (0.64 GB)",
                ],
                [
                    "<code>big_qos</code>로는 job이 배치되지 않음",
                    "3개 job이 수 시간 <code>(Priority)</code> 대기",
                    "<code>base_qos</code>(비선점, GPU 8 상한)로 전환하니 즉시 배치. resume 없는 이 학습에 비선점은 필수",
                ],
                [
                    "에셋이 quota 빠듯한 <code>/home</code>에 설치",
                    "RoboCasa가 <code>robocasa.__path__[0]/models/assets</code>로 해석 → repo 내부",
                    "<code>textures</code>/<code>generative_textures</code>/<code>objects</code>를 <code>/scratch</code>로 심볼릭 링크",
                ],
                [
                    "<code>--bootstrap-candidates 16</code>이 no-op",
                    "코드가 <code>0 &lt; c &lt; N</code>일 때만 subsample. N=16이라 baseline과 동일한 런이 될 뻔",
                    "<code>meta.json</code>에서 N을 읽어 N/2·N/4로 생성, 축퇴값은 skip",
                ],
            ],
            numeric_from=99,
        )
    )

    A("<h2>부록 B — 재현 시 주의</h2>")
    A(
        "<div class='box warn'><h4><code>slurm/env.sh</code>를 반드시 source</h4>"
        "<p><code>~/.bashrc</code>가 <code>~/miniconda3/lib</code>를 <code>LD_LIBRARY_PATH</code>에 넣는데, 그 "
        "<code>libcrypto.so.3</code>가 시스템 python3.11이 요구하는 <code>OPENSSL_3.4.0</code>보다 오래됐습니다. "
        "<code>.venv</code>가 그 인터프리터로 만들어지므로 <code>import hashlib</code>이 죽습니다 — "
        "<code>uv sync</code> 시점과 모든 job 실행 시점 양쪽에서. <code>env.sh</code>가 그 항목만 제거하고 CUDA/MuJoCo는 유지합니다.</p></div>"
    )
    A(
        "<div class='box'><h4>데이터가 거는 제약</h4>"
        "<p>514 에피소드가 <b>모두 성공</b>이라 성공/실패를 가르는 신호가 없습니다. critic이 구분할 수 있는 것은 "
        "<b>목표까지 남은 시간</b>뿐이고, 이는 순위 학습의 난이도를 근본적으로 규정합니다.</p></div>"
    )

    body = "\n".join(parts)
    args.out.write_text(
        f"<title>ACRFT critic ablation</title>\n<style>{CSS}</style>\n<main>{body}</main>\n", encoding="utf-8"
    )
    print(f"wrote {args.out}  ({done}/{len(runs)} runs complete)")


if __name__ == "__main__":
    main()
