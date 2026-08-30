import json
import pathlib
import subprocess
import sys

S = "/data5/jellyho/ACRFT/openpi/.scratch"
R = "/data5/jellyho/ACRFT/openpi"
sys.path.insert(0, R + "/slurm/space_entries")
sys.path.insert(0, S)
from entry_easy import EN  # noqa: E402
from entry_easy import KO  # noqa: E402

ns = {}
exec(pathlib.Path(S + "/sections_new.py").read_text(), ns)
ne = {}
exec(pathlib.Path(S + "/sections_new_en.py").read_text(), ne)
fk = {}
exec(pathlib.Path(S + "/formal_ko.py").read_text(), fk)
fe = {}
exec(pathlib.Path(S + "/formal_en.py").read_text(), fe)

MDPFIG_KO = (
    '<figure><img src="figures/chunking-easy/fig_mdp.png" alt="three MDPs"/><figcaption>'
    "우리 방식이 사는 세 무대. <b>(A)</b> 기저 MDP — 매 스텝 결정(▼). <b>(B)</b> <b>full-commitment MDP \\(M_H\\)</b> — "
    "chunk 하나가 <b>행동 하나</b>이고 H스텝 open-loop 실행이 <b>전이 하나</b>다(보상은 할인합, 할인율 \\(\\gamma^H\\)). "
    '<b>우리 개선 루프가 사는 곳이 바로 여기</b>이고, 그래서 개선이 "그 MDP에서의 policy iteration"이 되어 수렴이 '
    "따라온다(정리 A). <b>(C)</b> 배포 — 상태마다 \\(\\kappa(s)\\)만큼 commit하고 다시 질의. <b>선택자는 여기에만</b> "
    '있고 개선 루프를 건드리지 않는다. 이 <b>분리</b>가 "decoupled horizon"의 정확한 뜻이다.</figcaption></figure>\n'
)
MDPFIG_EN = (
    '<figure><img src="figures/chunking-easy/fig_mdp.png" alt="three MDPs"/><figcaption>'
    "The three stages our scheme lives on. <b>(A)</b> the base MDP — a decision every step (▼). <b>(B)</b> the "
    "<b>full-commitment MDP \\(M_H\\)</b> — one chunk is <b>one action</b> and the H-step open-loop rollout is "
    "<b>one transition</b> (reward = discounted sum, discount \\(\\gamma^H\\)). <b>This is where our improvement loop "
    'lives</b>, which is why improvement becomes "policy iteration in that MDP" and convergence follows (Theorem A). '
    "<b>(C)</b> deployment — commit \\(\\kappa(s)\\) steps, then re-query. <b>The selector lives only here</b> and "
    'never touches the improvement loop. That <b>separation</b> is precisely what "decoupled horizon" means.'
    "</figcaption></figure>\n"
)

# --- KO insertions
KO = KO.replace("<h3>3. 오차 공식이 왜 그 모양인가", ns["AOLC_KO"] + "\n<h3>3. 오차 공식이 왜 그 모양인가", 1)
KO = KO.replace(
    "<h3>9. 길이가 저절로 자란다",
    MDPFIG_KO + fk["FORMAL_KO"] + ns["BON_KO"] + ns["TIE_KO"] + "\n<h3>9. 길이가 저절로 자란다",
    1,
)
KO = KO.replace(
    "<p>그래서 이 곡선(평균 길이 ↑)이", ns["DERIV_KO"] + ns["CONV_KO"] + "<p>그래서 이 곡선(평균 길이 ↑)이", 1
)
# --- EN insertions
EN = EN.replace(
    "<h3>3. Why the error formula looks like that", ne["AOLC_EN"] + "\n<h3>3. Why the error formula looks like that", 1
)
EN = EN.replace(
    "<h3>9. The length grows by itself",
    MDPFIG_EN + fe["FORMAL_EN"] + ne["BON_EN"] + ne["TIE_EN"] + "\n<h3>9. The length grows by itself",
    1,
)
EN = EN.replace(
    "<p>That curve (mean length ↑) therefore becomes",
    ne["DERIV_EN"] + ne["CONV_EN"] + "<p>That curve (mean length ↑) therefore becomes",
    1,
)


def g(*a):
    return subprocess.run(["git", "-C", R, *a], capture_output=True, text=True, check=False).stdout.strip()


dirty = subprocess.run(["git", "-C", R, "diff", "--quiet"], capture_output=True, check=False).returncode != 0
STAMP = f"{g('rev-parse', '--abbrev-ref', 'HEAD')}@{g('rev-parse', '--short', 'HEAD')}{'+dirty' if dirty else ''}"
DATE = "2026-08-19 17:30"
STYLE = pathlib.Path(S + "/style_easy.txt").read_text()


def spec(rows):
    return (
        "<div class='tblwrap'><table class='spec w6'>"
        + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
        + "</table></div>"
    )


SPEC_KO = spec(
    [
        ("누가", "워커B"),
        ("언제", DATE),
        ("어디서", "QC arXiv:2507.07969 / DQC arXiv:2512.10926 / AQC arXiv:2605.05544"),
        (
            "무엇을",
            "action chunking 수학의 쉬운 판 + <b>우리 방식(decoupled horizon)의 정식화와 증명</b>(정리 A·B·D·E, 명제 E·F~L)",
        ),
        (
            "어떻게",
            "'맛보는 셰프' 비유로 관통 + 개선 루프를 \\(M_H\\) 안에 닫아 policy iteration으로 환원 + 계산된 그림 5개",
        ),
        (
            "왜",
            "수학 배경 없이도, 왜 adaptive가 필요하고 왜 우리 방식이 원하는 결론에 <b>반드시</b> 도달하는지 이해할 수 있게",
        ),
        (
            "코드",
            f"<code>{STAMP}</code> · 그림 <code>scripts/fig_chunking_easy.py</code>, <code>fig_chunking_bias.py</code>, <code>fig_chunking_mdp.py</code>",
        ),
    ]
)
SPEC_EN = spec(
    [
        ("Who", "worker B"),
        ("When", DATE),
        ("Where", "QC arXiv:2507.07969 / DQC arXiv:2512.10926 / AQC arXiv:2605.05544"),
        (
            "What",
            "the plain-language version of the mathematics, plus <b>the formalisation and proofs of our scheme</b> (Theorems A, B, D, E; Propositions E, F–L)",
        ),
        (
            "How",
            "one analogy throughout, the improvement loop closed inside \\(M_H\\) so it reduces to policy iteration, and five computed figures",
        ),
        (
            "Why",
            "so that, without a maths background, one can see why adaptive chunking is needed and why our scheme <b>must</b> reach the intended conclusion",
        ),
        (
            "Code",
            f"<code>{STAMP}</code> · figures <code>scripts/fig_chunking_easy.py</code>, <code>fig_chunking_bias.py</code>, <code>fig_chunking_mdp.py</code>",
        ),
    ]
)

entry = {
    "eid": "chunking-easy",
    "worker": "B",
    "date": DATE,
    "title": "🧪 [워커B] 🍳 쉽게 풀어쓴 action chunking의 수학 — 맛보는 셰프, 그리고 decoupled horizon이 왜 반드시 수렴하는가",
    "status": "finding",
    "summary": (
        "수학 배경 없이 읽는 판 + 우리 방식의 정식화. '맛보는 셰프의 일지' 비유로 chunk critic의 착각(믿는 값 99 vs "
        "실제 49)을 보이고, 새는 양동이로 오차 공식을, AOLC로 '확인 시점이 상태마다 달라지면' 무엇이 필요한지 설명한다. "
        "그 위에 우리 방식을 정식화한다: 개선 루프를 full-commitment MDP M_H 안에 닫으면 그것은 그 MDP의 policy iteration이 "
        "되어 epistemic 항이 0으로 수렴하고(정리 A), 적응 실행은 그 하한을 깨지 않으며(정리 B), 배포값은 aleatoric floor "
        "이내로 수렴한다(따름정리 C). 왜 개선이 k=H여야 하는지(명제 E), BoN(AQC·ACSAC)도 개선이지만 왜 분위수·support·"
        "curriculum 부재에서 멈추는지(명제 F~I), 이상적이면 모든 k가 동률인데 실제로 무엇이 그것을 깨는지, 그리고 종점에서 "
        "selector가 편향에 지배되는 이유(명제 J~L), 마지막으로 길이가 왜 필연적으로 길어져 유한 시간에 수렴하는지(정리 E·E′). "
        "그림 5개는 전부 계산된 것."
    ),
    "tags": ["워커B", "이론", "쉽게풀어쓴"],
    "phase": "논문·교차",
    "links": ["chunking-theory", "paper-intro", "aqc-ablation", "alphaflow-pi05", "deas"],
    "body_html": (
        '<div class="wbx wbx-ko">' + STYLE + '<div class="wbe">' + SPEC_KO + KO + "</div></div>"
        '<div class="wbx wbx-en">' + STYLE + '<div class="wbe">' + SPEC_EN + EN + "</div></div>"
    ),
}
p = pathlib.Path(S) / "entry_chunking_easy.json"
p.write_text(json.dumps(entry, ensure_ascii=False, indent=1))
b = entry["body_html"]
print(
    "wrote",
    p.stat().st_size // 1024,
    "KB | figures:",
    b.count("<figure>"),
    "| display math:",
    b.count("\\["),
    "| inline:",
    b.count("\\("),
    "| balance:",
    b.count("\\[") == b.count("\\]"),
    b.count("\\(") == b.count("\\)"),
)
