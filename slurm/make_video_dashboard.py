"""Index every rollout video into one page, with the numbers that explain what you are watching.

The HUD answers "what did the critic believe" for a single rollout. What it cannot show is the
comparison the evaluation is actually about — the same scene under `vla` and under `critic`, or the
same variant before and after a flag. That comparison is why the videos are worth watching at all,
and hunting for the matching file under six run directories is enough friction to stop anyone doing
it. So: one page, grouped by run, modes side by side on the same scene, with each run's offline
diagnostics printed next to its clips so a video is never watched without its context.

Videos are linked, not embedded — a rollout is a few MB and the page indexes dozens.

    uv run slurm/make_video_dashboard.py --out /scratch/jellyho/acrft/videos.html
    # then, to view (the page uses paths relative to $CACHE_DIR):
    cd /scratch/jellyho/acrft && python -m http.server 8800
"""

import argparse
import collections
import html
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _describe import BASELINE_NOTE, describe

CSS = """
:root{--bg:#12130f;--fg:#eceadf;--mut:#8f8d7e;--line:#2a2c22;--panel:#191b14;--signal:#e0863f;
      --ok:#78c096;--bad:#e07a68}
@media (prefers-color-scheme:light){:root{--bg:#f7f5ef;--fg:#16180f;--mut:#6b6d5e;--line:#ddd9cc;
      --panel:#efece2;--signal:#b4531a;--ok:#2f6b4a;--bad:#8f2d1f}}
:root[data-theme=light]{--bg:#f7f5ef;--fg:#16180f;--mut:#6b6d5e;--line:#ddd9cc;--panel:#efece2;
      --signal:#b4531a;--ok:#2f6b4a;--bad:#8f2d1f}
:root[data-theme=dark]{--bg:#12130f;--fg:#eceadf;--mut:#8f8d7e;--line:#2a2c22;--panel:#191b14;
      --signal:#e0863f;--ok:#78c096;--bad:#e07a68}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem 5rem;
     font:15px/1.6 Pretendard,"Apple SD Gothic Neo",system-ui,-apple-system,sans-serif}
main{max-width:82rem;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .3rem;letter-spacing:-.02em}
.sub{color:var(--mut);font-family:ui-monospace,monospace;font-size:.82rem;margin:0 0 2rem}
h2{font-size:1.05rem;margin:2.6rem 0 .3rem;letter-spacing:-.01em}
h2 code{background:none;padding:0;color:var(--signal)}
.metrics{font-family:ui-monospace,monospace;font-size:.76rem;color:var(--mut);
         margin:0 0 1rem;padding-bottom:.6rem;border-bottom:1px solid var(--line)}
.metrics b{color:var(--fg);font-weight:600}
.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fill,minmax(19rem,1fr))}
.card{border:1px solid var(--line);border-radius:5px;background:var(--panel);overflow:hidden;
      display:flex;flex-direction:column}
.card video{width:100%;display:block;background:#000;aspect-ratio:1/1}
.cap{padding:.5rem .65rem;display:flex;align-items:center;gap:.5rem;
     font-family:ui-monospace,monospace;font-size:.76rem}
.mode{color:var(--signal);font-weight:600}
.trial{color:var(--mut)}
.badge{margin-left:auto;padding:.1em .5em;border-radius:2px;font-size:.68rem;font-weight:600}
.ok{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.bad{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.legend{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--signal);
        border-radius:0 4px 4px 0;padding:.85rem 1.1rem;margin:0 0 2rem;font-size:.88rem}
.legend p{margin:.35rem 0;color:var(--mut)}
.legend b{color:var(--fg)}
code{font-family:ui-monospace,monospace;font-size:.86em;background:var(--line);
     padding:.08em .34em;border-radius:3px}
.empty{color:var(--mut);font-style:italic}
.what{margin:.1rem 0 .5rem;font-size:.85rem;color:var(--fg);line-height:1.55}
.what.mut{color:var(--mut)}
"""

MODE_NOTE = {
    "vla": "critic 없음 — 비교 기준",
    "bon": "후보만 선택, 커밋 길이 고정",
    "prefix": "첫 후보 고정, 커밋 길이만 선택",
    "critic": "(후보, prefix) 결합 arg-max — 실제 배포 규칙",
}
MODE_ORDER = ["vla", "bon", "prefix", "critic"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=None, help="default: $CACHE_DIR")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="default: <root>/videos.html")
    ap.add_argument("--generated", default="")
    args = ap.parse_args()

    root = args.root or pathlib.Path(os.environ.get("CACHE_DIR", "/scratch/jellyho/acrft"))
    out = args.out or root / "videos.html"
    runs_root = root / "critic_runs"

    # run -> mode -> [(trial, success, relpath)]
    found = collections.defaultdict(lambda: collections.defaultdict(list))
    for mp4 in sorted(runs_root.glob("*/*/videos/*.mp4")):
        m = re.match(r".*_(vla|bon|prefix|critic)_t(\d+)_(succ|fail)\.mp4$", mp4.name)
        if not m:
            continue
        mode, trial, outcome = m.group(1), int(m.group(2)), m.group(3)
        run_key = f"{mp4.parents[2].name}/{mp4.parents[1].name}"  # sweep/run
        found[run_key][mode].append((trial, outcome == "succ", mp4.relative_to(root).as_posix()))

    parts = []
    A = parts.append
    A("<h1>ACRFT rollout 영상</h1>")
    total = sum(len(v) for r in found.values() for v in r.values())
    A(f"<p class='sub'>{len(found)} runs · {total} clips" + (f" · {html.escape(args.generated)}" if args.generated else "") + "</p>")
    A("<div class='legend'><p><b>네 모드는 같은 scene에서 돕니다</b> (seed 고정). "
      "critic의 두 결정을 분리하려는 구성이라, 나란히 봐야 의미가 있습니다.</p>"
      + "".join(f"<p><code>{m}</code> — {MODE_NOTE[m]}</p>" for m in MODE_ORDER)
      + "<p style='margin-top:.6rem'>HUD: 우측은 <b>후보 16개의 값 산포</b>(겹쳐 있으면 best-of-N은 동전던지기), "
        "아래는 <b>value trace</b>(로그축 — <code>V=γ^남은스텝</code>이므로 일관되면 직선)와 "
        "<b>replan당 커밋 스텝</b>입니다.</p>"
        "<p style='margin-top:.6rem'>런 헤더의 숫자: <code>act_sens</code> = 상태 내부 Q 분산 ÷ 상태 간 분산 "
        "(<b>0이면 critic이 액션을 무시</b>하고 <code>Q(z,a)=V(z)</code>로 붕괴한 것). "
        "<code>rank_cand</code> = 같은 상태에서 시연 chunk가 정책 후보보다 높은 Q를 받는 비율 "
        "(<b>우연 = 0.5</b>). 둘 다 기준값이면 best-of-N은 무작위 선택입니다.</p>"
        f"<p style='margin-top:.6rem'><b>baseline</b> = {BASELINE_NOTE}. "
        "런 이름 아래 ▸ 로 적힌 것이 baseline과 다른 점의 전부입니다 (한 번에 하나씩만 바꿉니다).</p></div>")

    if not found:
        A("<p class='empty'>영상이 없습니다. rollout eval을 <code>NUM_VIDEOS=2</code>로 돌리세요.</p>")

    for run_key in sorted(found):
        sweep, run = run_key.split("/")
        A(f"<h2><code>{html.escape(run)}</code> <span style='color:var(--mut);font-size:.8rem'>({html.escape(sweep)})</span></h2>")

        # 진단 수치를 옆에 둔다: 맥락 없이 영상만 보면 인상만 남는다.
        bits = []
        diag = runs_root / sweep / run / "diag.json"
        if diag.exists():
            try:
                d = json.loads(diag.read_text())
                bits.append(f"act_sens <b>{d.get('action_sensitivity', float('nan')):.4f}</b> (0=액션무시)")
                bits.append(f"rank_cand <b>{d.get('ranking_accuracy_demo_vs_candidate', float('nan')):.3f}</b> (우연 0.5)")
            except Exception:
                pass
        cfgf = runs_root / sweep / run / "config.json"
        if cfgf.exists():
            try:
                c = json.loads(cfgf.read_text())
                diffs = describe(c)
                A(f"<p class='what'>{'<br>'.join('▸ ' + html.escape(d) for d in diffs)}</p>"
                  if diffs else "<p class='what mut'>▸ baseline (아래 설정 그대로)</p>")
            except Exception:
                pass
        # 성공률은 모드별로: 영상 자체가 표본이므로 몇 개 중 몇 개인지 보여준다.
        rates = []
        for mode in MODE_ORDER:
            clips = found[run_key].get(mode)
            if clips:
                rates.append(f"{mode} {sum(1 for _, s, _ in clips if s)}/{len(clips)}")
        if rates:
            bits.append("· 이 클립들: " + "  ".join(rates))
        if bits:
            A(f"<p class='metrics'>{'  '.join(bits)}</p>")

        A("<div class='grid'>")
        for mode in MODE_ORDER:
            for trial, succ, rel in sorted(found[run_key].get(mode, [])):
                cls, label = ("ok", "성공") if succ else ("bad", "실패")
                A(f"<div class='card'><video controls preload='metadata' src='{html.escape(rel)}'></video>"
                  f"<div class='cap'><span class='mode'>{mode}</span>"
                  f"<span class='trial'>trial {trial}</span>"
                  f"<span class='badge {cls}'>{label}</span></div></div>")
        A("</div>")

    out.write_text(
        f"<title>ACRFT rollout 영상</title>\n<style>{CSS}</style>\n<main>{''.join(parts)}</main>\n",
        encoding="utf-8",
    )
    print(f"wrote {out}  ({len(found)} runs, {total} clips)")
    print(f"\n보려면:\n  cd {root} && python -m http.server 8800")
    print("  그다음 브라우저에서 http://localhost:8800/videos.html")
    print("  (VS Code 원격이면 포트 8800이 자동 포워딩됩니다)")


if __name__ == "__main__":
    main()
