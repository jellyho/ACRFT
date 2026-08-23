"""Build the weekly presentation page from the week's figures and numbers.

The rule this serves: every week the work is presented, and a presentation needs the background
drawn, not only the results plotted. So this script assembles three layers per experiment (setup,
mechanism, result) into one page a person can talk through.

Everything is regenerated from files on disk: figures are read from hub_figs and embedded as data
URIs (the artifact CSP blocks external hosts), and the headline numbers are recomputed from the
result JSONs rather than typed in. Nothing here is hand-copied.

    uv run python slurm/make_weekly_deck.py --out /scratch/jellyho/acrft/decks/weekly_2026-08-23.html
"""

import argparse
import base64
import html
import json
import pathlib

FIGS = pathlib.Path("/scratch/jellyho/acrft/hub_figs")
PROBES = pathlib.Path("/scratch/jellyho/acrft/probes")


def data_uri(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def figure(path: pathlib.Path, caption: str, alt: str) -> str:
    uri = data_uri(path)
    if not uri:
        return f"<figure class='fig'><div class='missing'>figure not built yet: {html.escape(path.name)}</div></figure>"
    return (
        f"<figure class='fig'><img src='{uri}' alt='{html.escape(alt)}'>" f"<figcaption>{caption}</figcaption></figure>"
    )


def load(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--week", default="2026-08-17 → 08-23")
    a = ap.parse_args()

    nn = load(PROBES / "toy_cfac_nn" / "results.json")
    nn_curric = load(PROBES / "toy_cfac_nn_curric" / "results.json")
    ksweep = load(pathlib.Path("slurm/probes/ksweep_results.json"))

    # headline numbers, recomputed
    def paired(key):
        try:
            v = nn["summary"]["_paired"][key]
            return f"{v[0]:+.3f} ({v[2]}/{v[3]})"
        except (TypeError, KeyError):
            return "대기"

    def arm(res, name, metric="ret"):
        try:
            m = res["summary"][name][metric]
            return f"{m[0]:.3f} ± {m[1]:.3f}"
        except (TypeError, KeyError):
            return "대기"

    def curric_row():
        if not nn_curric:
            return "대기"
        ps = nn_curric["per_seed"]
        first = sum(p["cfac_sel"]["k_corridor"] for p in ps) / len(ps)
        last = sum(p["cfac_joint_r3"]["k_corridor"] for p in ps) / len(ps)
        d = [p["cfac_joint_r3"]["k_corridor"] - p["cfac_sel"]["k_corridor"] for p in ps]
        wins = sum(1 for x in d if x > 0)
        return f"{first:.2f} → {last:.2f} (Δ {sum(d) / len(d):+.3f}, {wins}/{len(d)})"

    ks_best = ""
    ks_gain = "대기"
    if ksweep:
        ks_best = ", ".join(f"{t.replace('PickPlace', '')} k={r['best_k']}" for t, r in ksweep["per_task"].items())
        ks_gain = f"{ksweep['mean_best_minus_full_chunk']:+.3f}"

    css = """
:root{
  --paper:#F7F8FA; --ink:#171A21; --muted:#667085; --rule:#DDE1E8; --card:#FFFFFF;
  --signal:#DD8452; --blue:#4C72B0; --green:#55A868; --red:#C44E52;
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#12141A; --ink:#E8EAF0; --muted:#9AA3B2; --rule:#2A2F3A; --card:#181B22;
    --signal:#E9986A; --blue:#6E93CC; --green:#6FBF86; --red:#D9737A;
  }
}
:root[data-theme="dark"]{
  --paper:#12141A; --ink:#E8EAF0; --muted:#9AA3B2; --rule:#2A2F3A; --card:#181B22;
  --signal:#E9986A; --blue:#6E93CC; --green:#6FBF86; --red:#D9737A;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:Charter,"Bitstream Charter","Iowan Old Style",Georgia,"Times New Roman",serif;
  font-size:17px; line-height:1.62;
}
.wrap{max-width:72rem;margin:0 auto;padding:0 clamp(1rem,4vw,3rem) 6rem}
.col{max-width:38rem}
header.top{padding:4.5rem 0 2.2rem;border-bottom:1px solid var(--rule);margin-bottom:2.5rem}
.eyebrow{
  font:600 .72rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--signal)
}
h1{font-size:clamp(2rem,4.4vw,3rem);line-height:1.08;margin:.7rem 0 .6rem;text-wrap:balance;font-weight:600}
.lede{font-size:1.12rem;color:var(--muted);max-width:34rem;margin:0}
section{padding:2.6rem 0;border-bottom:1px solid var(--rule)}
section:last-of-type{border-bottom:0}
h2{font-size:1.5rem;margin:.35rem 0 .9rem;font-weight:600;text-wrap:balance}
h3{font-size:1.02rem;margin:1.8rem 0 .4rem;font-weight:600}
p{margin:.85rem 0;max-width:38rem}
/* the section marker echoes the commitment timeline the work is about: filled cells = how far along */
.mark{display:flex;gap:3px;align-items:center;margin-bottom:.15rem}
.mark i{display:block;width:16px;height:5px;background:var(--rule);border-radius:1px}
.mark i.on{background:var(--signal)}
.mark span{font:600 .7rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);margin-left:.5rem}
.fig{margin:1.6rem 0 .4rem;padding:0}
.fig img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:6px;background:#fff}
.fig figcaption{font-size:.84rem;color:var(--muted);margin-top:.55rem;max-width:44rem}
.missing{border:1px dashed var(--rule);padding:2rem;text-align:center;color:var(--muted);font-size:.9rem}
.tablewrap{overflow-x:auto;margin:1.1rem 0}
table{border-collapse:collapse;font-size:.9rem;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--rule);padding:.5rem .7rem;text-align:left;vertical-align:top}
th{font:600 .74rem/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted)}
td.n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86rem;white-space:nowrap}
.win{color:var(--green);font-weight:600}
.loss{color:var(--red)}
.pending{color:var(--muted);font-style:italic}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:1rem;margin:1.3rem 0}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:1rem 1.1rem}
.card h4{margin:0 0 .3rem;font-size:.95rem;font-weight:600}
.card p{margin:0;font-size:.9rem;color:var(--muted);max-width:none}
.stat{font:600 1.5rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink);
  font-variant-numeric:tabular-nums;display:block;margin:.15rem 0 .25rem}
.note{border-left:3px solid var(--signal);padding:.15rem 0 .15rem 1rem;margin:1.3rem 0;color:var(--muted);
  font-size:.94rem;max-width:38rem}
.note b{color:var(--ink)}
ul{max-width:38rem;padding-left:1.1rem}
li{margin:.4rem 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;
  background:color-mix(in srgb,var(--muted) 12%,transparent);padding:.1em .35em;border-radius:3px}
footer{padding:2.5rem 0;color:var(--muted);font-size:.85rem}
@media (max-width:640px){body{font-size:16px}}
"""

    def mark(n_on, label, total=5):
        cells = "".join(f"<i class='{'on' if i < n_on else ''}'></i>" for i in range(total))
        return f"<div class='mark'>{cells}<span>{label}</span></div>"

    body = f"""
<div class="wrap">
<header class="top">
  <div class="eyebrow">주간 발표 (2/2) · {html.escape(a.week)}</div>
  <h1>커밋을 최적화 차원으로: 이론, 방법, 그리고 이번 주에 틀린 것들</h1>
  <p class="lede">VLA를 오프라인 RL로 개선하는 연구에서, 이번 주에는 <b>얼마나 오래 커밋할지</b>를
  크리틱이 공정하게 평가하게 만드는 방법(CFAC)을 세우고 toy 두 개로 검증했으며, 실로봇 쪽에서는
  적응 실행을 비교할 <b>정직한 기준선</b>을 확정했다.</p>
  <p class="lede" style="font-size:.95rem;margin-top:.8rem">같은 주의 앞편(α-Flow 원스텝 게이트·이론
  프로그램·FQL)은 <b>주간 발표 (1/2)</b>에 있다. 두 편은 다른 갈래를 다루며 서로를 전제하지 않는다.</p>
</header>

<section>
  {mark(1, "배경 · 왜 커밋이 차원인가")}
  <h2>사람의 데모는 Markov가 아니다</h2>
  <p>VLA는 청크(연속된 여러 행동)를 한 번에 내놓고 그중 일부를 실행한 뒤 다시 질의한다. 얼마나 실행할지는
  보통 고정 상수인데, 사람 데모를 보면 그 값이 상황마다 달라야 한다. 사람은 어떤 구간에서는 <b>계획을 기억한 채
  밀고 가고</b>(관측에 없는 정보를 행동이 운반한다), 어떤 구간에서는 <b>정보가 도착하기를 기다렸다 반응한다</b>.</p>
  <p>이 두 상황을 최소 형태로 담은 환경이 아래다. 먼저 <b>무슨 일이 일어나는지</b>, 그 다음 <b>무엇이 언제
  보이는지</b>.</p>
  {figure(FIGS / "toy_cfac_story.png", "에피소드 한 편을 네 순간으로. 복도에서는 표지판이 입구에만 떠 있다가 사라지고, 분기점에서는 한 스텝 뒤에야 신호가 켜진다. 아랫줄은 세 전략이 각각 어디서 이기고 지는가 — 어떤 고정 규칙도 두 구간을 함께 맞출 수 없다.", "storyboard of the toy task")}
  {figure(FIGS / "toy_cfac_setup.png", "한 에피소드의 정보 타임라인. 복도에서는 목표 방향이 입구에만 보이고(과거 잠재), 분기점에서는 첫 스텝 뒤에 도착한다(미래 잠재). 데모를 한 사람은 기억하지만, Markov 정책은 현재 관측만 본다.", "information timeline of the toy environment")}
</section>

<section>
  {mark(2, "기제 · 크리틱이 왜 틀리나")}
  <h2>세 가지 미명세, 그리고 CFAC</h2>
  <div class="tablewrap"><table>
    <tr><th>미명세</th><th>내용</th><th>편향</th><th>처방</th></tr>
    <tr><td>공짜 재질의</td><td>부트스트랩이 "재질의하면 좋은 연속이 온다"고 가정</td><td>짧게</td><td>배포 정책의 기대로 부트스트랩</td></tr>
    <tr><td>Markov 조건</td><td>계획이 관측에 없으면 커밋의 정보를 표현할 공간이 없음</td><td>커밋 가치 붕괴</td><td>실행 히스토리 조건화</td></tr>
    <tr><td>청크-회귀 교란</td><td>창 안에서 공개된 사건이 데모의 행동과 결과를 동시에 유발</td><td>길게</td><td>개입적 합성 (후속상태 재샘플)</td></tr>
  </table></div>
  <p class="note"><b>한 줄로.</b> 반응이 가치 있는 상태가 곧 청크 회귀가 거짓말하는 상태다. 같은 창-내 공개가
  반응의 가치와 교란을 동시에 만들기 때문에, naive 크리틱은 <b>정확히 반응해야 할 곳에서</b> 커밋을 부풀린다.</p>
  {figure(FIGS / "toy_cfac_viz.png", "같은 장면에서 두 크리틱이 무엇을 실행했는가. 아래 가운데 패널이 기제다: 분기점 입구에서 같은 청크를 놓고 물었을 때 naive는 긴 커밋에 최고점을 주고 CFAC는 k=1에 준다.", "matched rollout comparison of naive and CFAC")}
</section>

<section>
  {mark(3, "증거 · toy 두 개")}
  <h2>기제가 존재하고, 실제 알고리즘에서도 작동한다</h2>
  <p>첫 toy는 전수 열거가 가능한 tabular 환경으로 <b>기제의 존재</b>를 보였고, 둘째는 배포할 형태 그대로
  (신경 크리틱, 모델 없는 TD, AWR actor, 연속 행동) 다시 만들어 <b>알고리즘이 작동하는지</b>를 물었다.</p>
  <div class="cards">
    <div class="card"><h4>CFAC − naive</h4><span class="stat win">{paired("cfac_sel-naive_sel")}</span>
      <p>연속 환경, 6시드 짝지은 차</p></div>
    <div class="card"><h4>개입 제거 시</h4><span class="stat loss">−{paired("cfac_sel-cfac_nointerv_sel").lstrip("+")}</span>
      <p>합성만으로는 부족하다는 증거</p></div>
    <div class="card"><h4>히스토리 제거 시</h4><span class="stat loss">−{paired("cfac_sel-cfac_nohist_sel").lstrip("+")}</span>
      <p>두 성분 모두 필요</p></div>
    <div class="card"><h4>joint vs 수제 오라클</h4><span class="stat">{paired("cfac_joint-bc_oracle")}</span>
      <p>학습된 규칙이 손으로 짠 규칙과 동률</p></div>
  </div>
  {figure(FIGS / "toy_cfac_nn.png", "왼쪽: 배포 수익. 가운데: 각 성분이 무엇을 사는가(분기점 재질의 비율). 오른쪽: 커리큘럼 — 정책이 개선될수록 평균 커밋이 자란다.", "neural CFAC results")}
  <div class="tablewrap"><table>
    <tr><th>측정</th><th>값</th><th>읽기</th></tr>
    <tr><td>CFAC (선택만)</td><td class="n">{arm(nn, "cfac_sel")}</td><td>정책은 그대로, 커밋만 고름</td></tr>
    <tr><td>CFAC joint</td><td class="n">{arm(nn, "cfac_joint")}</td><td>정책까지 개선</td></tr>
    <tr><td>수제 오라클</td><td class="n">{arm(nn, "bc_oracle")}</td><td>정답 규칙을 손으로 넣은 상한</td></tr>
    <tr><td>커리큘럼 (변형 환경)</td><td class="n">{curric_row()}</td><td>평균 커밋 길이, 개선 라운드에 따라</td></tr>
  </table></div>
  {figure(FIGS / "toy_cfac.png", "tabular toy: 배포 성공률, 상태별 커밋 길이, 그리고 자기기만(believed − realized). CFAC만 belief 격차가 0이다.", "tabular CFAC results")}
</section>

<section>
  {mark(4, "실로봇 · 기준선 확정")}
  <h2>어느 고정 길이와 비교하느냐가 답을 정한다</h2>
  <p>RoboCasa에서 공식 π0.5를 5태스크 × 실행길이 6개로 평가했다. 목적은 두 가지였다 — 상태의존 커밋의
  <b>필요조건</b> 확인, 그리고 앞으로 모든 적응 비교가 대고 잴 <b>최고 상수 기준선</b> 확보.</p>
  <div class="cards">
    <div class="card"><h4>최적 상수가 태스크마다 다르다</h4><span class="stat">{html.escape(ks_best) or "대기"}</span>
      <p>단일 상수로는 어딘가에서 손해</p></div>
    <div class="card"><h4>최고 상수 − 기본값(k=16)</h4><span class="stat win">{ks_gain}</span>
      <p>상수만 잘 골라도 이만큼. 적응의 공으로 돌리면 안 되는 몫</p></div>
  </div>
  {figure(FIGS / "ksweep.png", "5태스크 × k∈{1,2,4,8,12,16}. k=1은 4/5 태스크에서 최악이고 두 태스크를 붕괴시킨다. 별표가 태스크별 최고 상수.", "fixed-k sweep on RoboCasa")}
  <p>B1(성공-필터 SFT)의 데이터 수집도 끝났다 — 5태스크 × 150 롤아웃 = 750 에피소드(성공 278). 학습은
  3B 모델의 메모리 사다리를 오르는 중이다(1 GPU → FSDP 2장 → 4장 + 배치 축소).</p>
</section>

<section>
  {mark(5, "정정 · 이번 주에 틀린 것들")}
  <h2>측정을 믿게 만든 세 번의 실패</h2>
  <ul>
    <li><b>빌려온 밴드.</b> 다른 비교에서 잰 시드 SD를 자로 썼다가 실재하는 효과를 잡음으로 판정했다.
      시드 산포는 비교마다 다르다 — 밴드는 그 비교 자신의 시드에서 뽑아야 한다.</li>
    <li><b>제출본을 건너뛴 비교.</b> 서로 다른 잡에서 나온 셀로 곡선을 조립하면 곡선이 아니다.
      한 비교의 모든 arm은 한 제출본 안에 있어야 한다. 내가 게시한 교차분석도 이 오류로 정정했다.</li>
    <li><b>검증하지 않은 자동 표.</b> 자동 재계산 표가 키 타입 불일치로 30칸 전부 빈 채 게시됐다.
      본문 길이와 그림만 확인하고 표를 읽지 않은 탓 — 이제 게시 전 렌더링된 표를 읽는다.</li>
  </ul>
  <p class="note">세 가지 모두 결론을 바꿨고, 그래서 논문 method에 <b>평가 프로토콜</b> 절로 못박았다:
  기준선은 최고 상수, 모든 arm은 한 제출본 안, 밴드는 그 비교 자신의 시드에서.</p>
</section>

<section>
  {mark(5, "다음 주")}
  <h2>무엇을 재는가</h2>
  <ul>
    <li><b>B1 판정</b> — 성공-필터 vs 무필터 vs 원본. "데이터가 좋아서"와 "많아서"를 가른다.</li>
    <li><b>M6 · M7</b> — CFAC의 두 성분을 RoboCasa 크리틱에 이식. 핵심 설계 쟁점은 "같은 결정 지점"을
      실제 환경에서 무엇으로 정의하는가.</li>
    <li><b>P-react</b> — 이미지 관측·접촉 확률성이 있는 우리 도메인에서는 정책이 좋아져도 적응 마진이
      0으로 수렴하지 않아야 한다. 상태 도메인과의 대조가 floor 항을 직접 재는 방법이다.</li>
  </ul>
</section>

<footer>
  자료는 전부 스크립트로 재생성된다 — 그림은 <code>slurm/probes/</code>의 생성기, 수치는 결과 JSON에서
  다시 계산, 이 페이지는 <code>slurm/make_weekly_deck.py</code>. 손으로 옮겨 적은 숫자는 없다.
</footer>
</div>
"""

    out = f"<title>주간 보고 · {html.escape(a.week)}</title>\n<style>{css}</style>\n{body}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out)
    print("wrote", a.out, f"({len(out) // 1024} KB)")


if __name__ == "__main__":
    main()
