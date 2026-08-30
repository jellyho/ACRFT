import json
import pathlib
import subprocess
import sys

sys.path.insert(0, "/data5/jellyho/ACRFT/openpi/.scratch")
from entry_ct_en import EN
from entry_ct_ko import KO

R = "/data5/jellyho/ACRFT/openpi"


def g(*a):
    return subprocess.run(["git", "-C", R, *a], capture_output=True, text=True, check=False).stdout.strip()


dirty = subprocess.run(["git", "-C", R, "diff", "--quiet"], capture_output=True, check=False).returncode != 0
STAMP = f"{g('rev-parse', '--abbrev-ref', 'HEAD')}@{g('rev-parse', '--short', 'HEAD')}{'+dirty' if dirty else ''}"
DATE = "2026-08-19 15:40"

STYLE = """<style>
.wbth{max-width:78rem}
.wbth p{max-width:62rem;margin:.62em 0}
.wbth h3{font-size:1.08rem;margin:1.9rem 0 .4rem;border-bottom:1px solid #e1e0d9;padding-bottom:.25rem}
:root[data-theme="dark"] .wbth h3{border-bottom-color:#3a3a34}
.wbth h4{font-size:.95rem;margin:1.25rem 0 .3rem;color:#3f3e3a}
:root[data-theme="dark"] .wbth h4{color:#d8d7cc}
.wbth ol,.wbth ul{max-width:62rem;margin:.5em 0 .5em 1.2em}
.wbth li{margin:.25em 0}
.wbth .tblwrap{overflow-x:auto;margin:.7rem 0}
.wbth table{border-collapse:collapse;font-size:.86rem;width:100%}
.wbth th,.wbth td{border:1px solid #e1e0d9;padding:5px 8px;text-align:left;vertical-align:top}
:root[data-theme="dark"] .wbth th,:root[data-theme="dark"] .wbth td{border-color:#3a3a34}
.wbth th{background:#f4f3ee;font-weight:600}
:root[data-theme="dark"] .wbth th{background:#232320}
.wbth blockquote{margin:.6rem 0;padding:.5rem .9rem;border-left:3px solid #b9b8ae;
  background:#f6f5f0;font-size:.86rem;max-width:62rem}
:root[data-theme="dark"] .wbth blockquote{background:#1d1d1a;border-left-color:#55544c}
.wbth .callout{background:#eef4f0;border-radius:9px;padding:.85rem 1rem;margin:1rem 0;max-width:64rem}
:root[data-theme="dark"] .wbth .callout{background:#17241c}
.wbth .warn{background:#fdefe7}
:root[data-theme="dark"] .wbth .warn{background:#2a1d14}
.wbth .callout .k{font:600 .7rem ui-monospace,Menlo,monospace;letter-spacing:.06em;
  text-transform:uppercase;color:#3f8a56;display:block;margin-bottom:.25rem}
.wbth .warn .k{color:#c0653a}
.wbth mjx-container{overflow-x:auto;overflow-y:hidden;max-width:100%}
</style>"""


def spec(rows):
    tr = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='tblwrap'><table class='spec w6'>{tr}</table></div>"


SPEC_KO = spec(
    [
        ("누가", "워커B(원문 정독·정리·검증) + 사용자(스토리라인 제시·반론)"),
        ("언제", DATE),
        ("어디서", "원문 PDF 정독 — QC arXiv:2507.07969 / DQC arXiv:2512.10926 / AQC arXiv:2605.05544"),
        (
            "무엇을",
            "action chunking RL의 수학 전체(OLC·value bias 정리군·AOLC·selector)와 우리 정식화(정리 3개+보조정리 2개)",
        ),
        (
            "어떻게",
            "정의·정리·증명 기계를 원문에서 직접 확인(추론은 표시) + AQC 3개 쟁점 원문 대조 검증 + 편향 단조성 수치 검증 + 사전등록 4건",
        ),
        (
            "왜",
            "adaptive chunking을 직관이 아니라 정리 위에 세우고, 무엇이 선점됐고 무엇이 비었는지 정직하게 긋기 위해",
        ),
        ("코드", f"<code>{STAMP}</code>"),
    ]
)
SPEC_EN = spec(
    [
        ("Who", "worker B (close reading, verification) + the user (storyline, objections)"),
        ("When", DATE),
        ("Where", "source PDFs — QC arXiv:2507.07969 / DQC arXiv:2512.10926 / AQC arXiv:2605.05544"),
        (
            "What",
            "the full mathematics of action-chunking RL (OLC, the value-bias theorems, AOLC, the selector) plus our formulation (3 theorems + 2 lemmas)",
        ),
        (
            "How",
            "definitions/theorems/proof machinery checked against the sources (inferences marked); three AQC claims verified verbatim; bias monotonicity checked numerically; 4 pre-registered tests",
        ),
        (
            "Why",
            "to put adaptive chunking on theorems rather than intuition, and to draw honestly what is taken and what is empty",
        ),
        ("Code", f"<code>{STAMP}</code>"),
    ]
)

entry = {
    "eid": "chunking-theory",
    "worker": "B",
    "date": DATE,
    "title": "🧪 [워커B] 📐 action chunking의 수학 — DQC의 정리군, AQC의 AOLC, 그리고 우리 정식화(정리·증명)",
    "status": "finding",
    "summary": (
        "QC/DQC/AQC 세 편의 수학을 정의·정리·증명 기계까지 전부 풀고, 그 위에 우리 기여를 정리로 세운다. "
        "DQC: nominal≠actual, open-loop consistency(weak/strong), Thm 1의 (1−ε)γ^h 수축, Cor 1이 bias를 suboptimality로 "
        "바꾸는 트릭, Prop 4(결정론이면 대가 0), Prop 1의 'lucky success' 반례, Thm 3의 2+1 구조, Prop 3 vs Thm 5/6(어느 쪽도 "
        "일반 우월 아님). AQC: AOLC는 오프셋 치환이며, selector 정규화는 보상 규약 의존적이고(그들 벤치마크에서 부호가 반대), "
        "soundness는 순환적이며, Thm H.8의 할인 정규화 단계는 부호가 뒤집혀 있다(원문 대조 확인). 그리고 AQC는 정책 개선이 없다. "
        "우리: V*₁=V*_H=V*_ada (결정론) 정리와 흡수 따름정리, floor 바운드, 개선을 k=H에 걸어야 하는 이유(Lemma B), "
        "누출 편향의 k-단조성, 사전등록 4건."
    ),
    "tags": ["워커B", "이론", "AQC"],
    "phase": "논문·교차",
    "links": ["paper-intro", "papers-tier1", "aqc-ablation", "alphaflow-pi05", "deas", "wc-aqc-method"],
    "body_html": (
        '<div class="wbx wbx-ko">' + STYLE + '<div class="wbth">' + SPEC_KO + KO + "</div></div>"
        '<div class="wbx wbx-en">' + STYLE + '<div class="wbth">' + SPEC_EN + EN + "</div></div>"
    ),
}
out = pathlib.Path(R) / ".scratch/entry_chunking_theory.json"
out.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
print("wrote", out.stat().st_size // 1024, "KB | ko", len(KO), "en", len(EN), "| stamp", STAMP)
