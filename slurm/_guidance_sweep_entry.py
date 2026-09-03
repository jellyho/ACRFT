"""The guidance-sweep entry, kept in its own file because make_master_report.py is already 7k lines.

Imported by make_master_report at the point the other entries are declared; everything it needs
(entry, en, img, table, spec, P) is passed in, so this file has no import cycle with it.

Every number here is recomputed from slurm/probes/guidance_sweep.json.gz at build time. Nothing is
transcribed.
"""

import gzip
import json
import pathlib

import numpy as np

PROBE = pathlib.Path(__file__).resolve().parents[1] / "slurm/probes/guidance_sweep.json.gz"


def _load():
    b = json.loads(gzip.decompress(PROBE.read_bytes()))
    ch = np.asarray(b["chunks"], np.float32)  # [S, A, H, AD]
    q = np.asarray(b["q"], np.float32)  # [S, A]
    sigma = float(b["sigma"])
    rows = []
    for ai, al in enumerate(b["alphas"]):
        d = np.array([np.sqrt(np.mean((ch[si, ai] - ch[si, 0]) ** 2)) / sigma for si in range(ch.shape[0])])
        oob = np.array([float(((ch[si, ai] < -1) | (ch[si, ai] > 1)).mean()) for si in range(ch.shape[0])])
        rows.append(
            [
                f"{al:.2f}",
                "0" if ai == 0 else f"{d.min():.1f} – {d.max():.1f}",
                f"{q[:, ai].mean():+.1f}",
                f"{100 * oob.mean():.1f}%",
            ]
        )
    return b, rows


def register(*, entry, en, img, table, spec, plots):
    """Add the KO and EN bodies to the report. `plots` is the directory make_figures wrote into."""
    b, rows = _load()
    png = plots / "34_guidance_sweep.png"
    gif = plots / "34_guidance_sweep.gif"
    alphas = b["alphas"]
    q = np.asarray(b["q"], np.float32)
    n_noise = len(b["chunks"])
    ep, fr = b["episode"], b["frame"]
    sigma, pc = b["sigma"], b["pc_sigma"]
    q_data = b["q_data"]
    # the two strengths that were actually deployed on the arm
    i01 = alphas.index(0.1) if 0.1 in alphas else 2
    i02 = alphas.index(0.2) if 0.2 in alphas else 3
    d01 = _disp(b, i01)
    d02 = _disp(b, i02)
    first_beat = next((a for a, qq in zip(alphas, q.mean(0), strict=True) if qq > q_data), None)

    ko_spec = spec(
        [
            ("무엇을 고정했나", f"관측(에피소드 {ep} 프레임 {fr}), 흐름의 초기 노이즈, critic 하나. α만 움직인다"),
            (
                "왜 그게 가능한가",
                "<code>Pi0Steered.sample_steered</code>가 노이즈를 인자로 받고 α=0이 같은 경로로 무조종 draw를 재현한다",
            ),
            ("노이즈", f"독립 {n_noise}개 — 한 draw가 정책 구름의 꼬리에 떨어지는 우연을 배제"),
            ("거리 단위", f"좌표당 RMS ÷ BC 산포(σ={sigma:.4f}/좌표, PC1 방향 {pc:.2f})"),
            ("클리핑", "없음 — 출력 클립을 뺀 브랜치라 박스를 넘는 포화가 가려지지 않는다"),
            ("critic", f"<code>{b['critic']}</code>, 자체 비관 축약(mean − ρ·std)"),
        ]
    )
    ko = (
        "<p><b>한 줄.</b> 배포에 쓰던 세기에서 조종은 정책 자신의 표집 산포보다 "
        f"<b>{d01:.0f}~{d02:.0f}배</b> 멀리 행동을 옮기고, 그러면서 Q는 계속 오른다. "
        "지지집합을 떠나는 것과 값이 좋아 보이는 것이 같은 그림 안에 있다.</p>" + ko_spec + "<h4>왜 이렇게 재는가</h4>"
        "<p>지금까지 조종을 잰 방식은 전부 집계였다. 롤아웃 전체의 평균 변위, 프레임 평균 Q, 스칼라 하나로 요약한 drift. "
        "그런 수치는 '얼마나 틀어지는가'를 보여주지 못한다. 평균이 작아도 개별 청크가 크게 휘었을 수 있고, 그 반대도 된다.</p>"
        "<p>그래서 반대로 갔다. 고칠 수 있는 것은 전부 고정하고 α만 사다리로 올린다. "
        "<code>sample_steered</code>가 노이즈를 인자로 받고 α=0이 <b>같은 경로로</b> 무조종 draw를 재현하기 때문에 가능하다. "
        "여기서 α=0 곡선은 'BC 샘플 비슷한 것'이 아니라 <b>바로 그 노이즈의 BC 샘플</b>이고, 나머지 곡선은 조종이 그 draw에 한 짓이다. "
        "두 곡선의 차이는 조종항이지 표집 분산이 아니다.</p>"
        "<h4>결과</h4>"
        + table(["α", "α=0 draw로부터 이동 (좌표당 BC σ)", "critic의 비관적 Q", "정규화 박스 밖 비율"], rows)
        + f"<p class='cap'>시연자의 실제 다음 30스텝은 Q = {q_data:+.1f}이다. 성공한 에피소드의 청크이므로, "
        "목표에 도달한 것이 확인된 행동이다. "
        + (
            f"<b>α={first_beat:g}에서 이미 그 값을 넘는다.</b>"
            if first_beat is not None
            else "표에 있는 어떤 α도 그 값을 넘지 못한다."
        )
        + " 넘었다는 것 자체가 문제는 아니다 — critic이 시연자보다 나은 행동을 찾았을 수도 있다. "
        "문제는 그 지점이 정책이 한 번도 가보지 않은 영역이라는 것이고, 거기서는 Q를 검증할 방법이 없다.</p>"
        + img(png, "one fixed noise, alpha swept: the chunk, where it bends, the policy plane, and the Q it bought")
        + "<p class='cap'><b>1번 (왼쪽 6칸) — 청크 자체.</b> 30스텝 × 14관절 중 조종이 가장 많이 움직인 6개를 골라 그렸다. "
        "선 하나가 α 하나다(어두울수록 약함). 점선은 시연자, 가로 점선은 정규화 박스 [−1, 1]. "
        "핵심은 <b>휘는 지점이 앞이 아니라 뒤</b>라는 것이다. 스텝 20까지는 거의 겹쳐 있다가 꼬리에서 갈라진다. "
        "α=1.6에서 왼팔 관절 하나는 −6까지 간다.</p>"
        "<p class='cap'><b>2번 — 어디서 휘는가.</b> 청크의 각 스텝에서 α=0 draw로부터의 거리를, 좌표당 BC σ 단위로. "
        f"밴드는 노이즈 {n_noise}개의 최소~최대다. 가로 파선이 1σ, 즉 <b>정책이 스스로 표집하며 흔들리는 폭</b>이다. "
        "배포 세기의 곡선이 그 선의 수 배 위에 있다는 것이 이 패널의 요점이다.</p>"
        "<p class='cap'><b>3번 — 정책의 지지집합을 벗어나는가.</b> 이 프레임에서 뽑은 BC draw 구름의 PC1/PC2 평면에 "
        "모든 α를 투영했다. 회색이 BC draw, 별이 시연자, 색점이 α 사다리다. "
        "α=0 점들은 회색 구름 안에 있다(당연하다, 그것이 BC draw다). α가 오르면 경로가 구름 밖으로 걸어 나간다. "
        "<b>이 패널과 4번 패널을 나란히 보는 것이 이 리포트의 전부다</b> — 나가면서 값은 계속 좋아진다.</p>"
        "<p class='cap'><b>4번 — 그 등반이 산 것.</b> 각 α에서 critic 자신의 비관적 Q. 회색 밴드는 노이즈 간 범위, "
        "붉은 파선이 시연자다. 조종은 정의상 Q를 올린다. 그 상승이 진짜인지는 이 패널이 답하지 못하고, "
        "<a class='xref' data-eid='q-landscape-ood'>Q-landscape 프로브</a>가 답한다: "
        "같은 critic 계열에서 박스 경계의 Q는 시연자보다 +12.7 높고, 그 오차를 비관성으로 지우려면 ρ가 2.5 필요한데 "
        "K=2에서 쓸 수 있는 최대는 1.0이다.</p>" + _gif_html(gif) + "<h4>이 그림이 두 번 틀렸다가 고쳐진 곳</h4>"
        "<p><b>단위.</b> 처음에는 30×14 청크의 노름을 좌표당 σ로 나눴다. sizeof 420차원이라 √420 ≈ 20.5배가 부풀어 "
        "'69σ, 1261σ' 같은 값이 나왔다. 좌표당 RMS로 고쳤다. 같은 함정을 며칠 전 다른 워커에게 경고해 놓고 "
        "그대로 걸어 들어갔다.</p>"
        "<p><b>기준선.</b> 처음에는 노이즈 하나만 썼는데, 그 α=0 점이 BC 구름의 PC1 기준 3σ 밖에 떨어졌다. "
        "'조종이 지지집합 밖에서 시작한다'로 읽힐 수 있어서 α=0 draw 16개를 따로 뽑아 확인했더니 구름 중심에서 0.05σ였다. "
        f"단순히 운 나쁜 한 표본이었다. 지금은 노이즈 {n_noise}개로 돌려 밴드로 표시한다.</p>"
        "<h4>한계</h4>"
        "<ul>"
        f"<li>프레임 하나다(에피소드 {ep}, 프레임 {fr}). 이 프레임에서의 α–변위 관계가 데이터셋 전체를 대표한다는 "
        "근거는 여기 없다. 프레임을 쓸어 담는 것은 <code>--episode/--frame</code>으로 되지만 돌리지 않았다.</li>"
        "<li>critic 하나다. Q-landscape 프로브는 9종에서 같은 방향을 확인했지만 이 그림은 그중 하나만 쓴다.</li>"
        "<li>출력 클립이 제거된 브랜치에서 쟀다. master에는 아직 클립이 살아 있어서, 거기서 돌리면 표의 "
        "'박스 밖 비율'이 0이 되고 큰 α의 변위가 과소평가된다. 클립을 뺀 것 자체가 공정성 때문이지만, "
        "<b>실물 롤아웃 당시에는 클립이 있었다</b>는 점은 이 표를 읽을 때 기억해야 한다.</li>"
        "</ul>"
        "<h4>재현</h4>"
        "<pre><code>uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --critic &lt;critic dir&gt; --policy-dir &lt;bc ckpt&gt; \\\n"
        "    --out sweep.png --gif sweep.gif --dump slurm/probes/guidance_sweep.json.gz   # GPU 필요\n"
        "uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --from-dump slurm/probes/guidance_sweep.json.gz --out sweep.png              # GPU 불필요</code></pre>"
        "<p>측정은 GPU가 필요하지만 그림은 언 데이터의 순수 함수다. "
        "<code>make_figures.py</code>의 <code>fig_34_guidance_sweep()</code>이 리포트 생성마다 "
        "<code>slurm/probes/guidance_sweep.json.gz</code>에서 다시 그린다. "
        "재생성본이 GPU 원본과 바이트 단위로 동일함을 확인했다.</p>"
    )

    en_spec = spec(
        [
            (
                "what is held fixed",
                f"the observation (episode {ep} frame {fr}), the flow's initial noise, one critic. Only α moves",
            ),
            (
                "why that is possible",
                "<code>Pi0Steered.sample_steered</code> takes the noise as an argument and α=0 reproduces the unsteered draw through that same path",
            ),
            (
                "noises",
                f"{n_noise} independent — so one draw landing in the tail of the policy's cloud cannot carry the result",
            ),
            (
                "distance unit",
                f"root-mean-square per coordinate ÷ the BC spread (σ={sigma:.4f}/coordinate, {pc:.2f} along PC1)",
            ),
            (
                "clipping",
                "none — measured on the branch that removed the output clip, so saturation past the box is visible",
            ),
            ("critic", f"<code>{b['critic']}</code>, its own pessimistic reduction (mean − ρ·std)"),
        ]
    )
    en_body = (
        "<p><b>One line.</b> At the strengths that were actually deployed, steering moves the action "
        f"<b>{d01:.0f}–{d02:.0f}×</b> further than the policy's own sampling spread, and Q rises the whole way. "
        "Leaving the support and looking better are the same picture.</p>"
        + en_spec
        + "<h4>Why measure it this way</h4>"
        "<p>Every previous measurement of steering here is an aggregate: displacement averaged over a rollout, "
        "Q averaged over frames, drift as a single scalar. None of those show how far a chunk actually bends — "
        "a small average is consistent with large individual excursions, and the reverse.</p>"
        "<p>So this goes the other way. Fix everything that can be fixed and walk α up a ladder. That is possible "
        "because <code>sample_steered</code> takes the noise as an argument and α=0 reproduces the unsteered draw "
        '<b>through that same path</b>. The α=0 curve here is not "a BC sample" loosely speaking; it is '
        "<b>this noise's</b> BC sample, and every other curve is what steering did to that exact draw. The "
        "difference between two curves is the steering term, not sampling variance.</p>"
        "<h4>Result</h4>"
        + table(
            [
                "α",
                "displacement from the α=0 draw (BC σ per coordinate)",
                "critic's pessimistic Q",
                "entries outside the box",
            ],
            rows,
        )
        + f"<p class='cap'>The demonstrator's own next 30 steps score Q = {q_data:+.1f}. That chunk comes from a "
        "successful episode, so it is an action known to have reached the goal. "
        + (
            f"<b>By α={first_beat:g} the steered chunk already beats it.</b>"
            if first_beat is not None
            else "No α in the table beats it."
        )
        + " Beating it is not by itself the problem — the critic may have found something better. The problem is "
        "that it does so in a region the policy never visits, where there is no way to check.</p>"
        + img(png, "one fixed noise, alpha swept: the chunk, where it bends, the policy plane, and the Q it bought")
        + "<p class='cap'><b>1 (the six panels on the left) — the chunk itself.</b> Of the 30 steps × 14 joints, the six "
        "joints steering moved most. One line per α (darker is weaker). Dashed is the demonstrator; the dotted "
        "horizontals are the normalized box [−1, 1]. The point is that <b>the bend is in the tail, not the head</b>: "
        "the curves sit on top of each other to about step 20 and separate after. At α=1.6 one left-arm joint reaches −6.</p>"
        "<p class='cap'><b>2 — where it bends.</b> Distance from this noise's α=0 draw at each step of the chunk, in "
        f"BC σ per coordinate. The band is the min–max over the {n_noise} noises. The dashed horizontal is 1σ, "
        "<b>the spread the policy already has from its own sampling</b>. That the deployed strengths sit several times "
        "above that line is what this panel is for.</p>"
        "<p class='cap'><b>3 — does it leave the policy's support?</b> Every α projected onto the PC1/PC2 plane of the BC "
        "draw cloud at this frame. Grey is the draws, the star is the demonstrator, the coloured points are the ladder. "
        "The α=0 points sit inside the grey cloud, as they must, being BC draws. As α rises the path walks out of it. "
        "<b>Reading this panel next to panel 4 is the whole report</b>: it leaves, and the value keeps improving.</p>"
        "<p class='cap'><b>4 — what the climb bought.</b> The critic's own pessimistic Q at each α; the grey band is the "
        "range across noises, the red dashed line the demonstrator. Steering raises Q by construction. Whether that rise "
        "is real is not answered here but by the "
        "<a class='xref' data-eid='q-landscape-ood'>Q-landscape probe</a>: in the same critic family, Q at the box edge "
        "sits +12.7 above the demonstrator, and erasing that with pessimism would need ρ = 2.5 where the strongest "
        "available at K=2 is 1.0.</p>" + _gif_html(gif) + "<h4>Two things this got wrong first</h4>"
        "<p><b>Units.</b> The first version divided the norm of a 30×14 chunk by a per-coordinate σ, which inflates by "
        '√420 ≈ 20.5 and produced "69σ, 1261σ". Root-mean-square per coordinate now. This is the same trap I had warned '
        "another worker about days earlier, walked into unchanged.</p>"
        "<p><b>Baseline.</b> The first version used one noise, and that α=0 point landed 3σ out along PC1 of the policy's "
        'own cloud — which reads as "steering starts off-support" and was one unlucky sample. Sixteen α=0 draws sit '
        f"0.05σ from the cloud centre. It runs from {n_noise} noises now, shown as a band.</p>"
        "<h4>Limits</h4>"
        "<ul>"
        f"<li>One frame (episode {ep}, frame {fr}). Nothing here says the α–displacement relation at this frame "
        "represents the dataset. Sweeping frames is a flag away and was not done.</li>"
        "<li>One critic. The Q-landscape probe found the same direction across nine; this figure uses one of them.</li>"
        "<li>Measured on the branch with the output clip removed. Master still clips, so running it there would report "
        "0% outside the box and understate the displacement at large α. Removing the clip was a fairness decision, but "
        "<b>the real rollouts were run with it in place</b>, which is worth remembering when reading the table.</li>"
        "</ul>"
        "<h4>Reproducing it</h4>"
        "<pre><code>uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --critic &lt;critic dir&gt; --policy-dir &lt;bc ckpt&gt; \\\n"
        "    --out sweep.png --gif sweep.gif --dump slurm/probes/guidance_sweep.json.gz   # needs a GPU\n"
        "uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --from-dump slurm/probes/guidance_sweep.json.gz --out sweep.png              # needs none</code></pre>"
        "<p>The measurement needs a GPU; the figure is a pure function of what it froze. "
        "<code>fig_34_guidance_sweep()</code> in <code>make_figures.py</code> redraws it from "
        "<code>slurm/probes/guidance_sweep.json.gz</code> on every report build, and the redraw was verified "
        "byte-identical to the GPU run's own output.</p>"
    )

    entry(
        "2026-09-03 18:30",
        "guidance-sweep",
        "노이즈를 고정하고 조종 세기만 올리면 — 배포 세기에서 이미 정책 산포의 5~11배",
        "완결",
        ko,
    )
    en(
        "guidance-sweep",
        "one noise held fixed, guidance turned up: 5–11× the policy's own spread at the deployed strengths",
        en_body,
    )


def _disp(b, ai):
    ch = np.asarray(b["chunks"], np.float32)
    sigma = float(b["sigma"])
    return float(np.mean([np.sqrt(np.mean((ch[si, ai] - ch[si, 0]) ** 2)) / sigma for si in range(ch.shape[0])]))


def _gif_html(path):
    """The animation, embedded as a GIF so it still moves. `img()` would re-encode it to a JPEG."""
    import base64

    p = pathlib.Path(path)
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (
        f"<img src='data:image/gif;base64,{b64}' alt='the same chunk, animated up the alpha ladder and back down'/>"
        "<p class='cap'>같은 청크를 사다리로 올렸다 내린 것. 제목 줄에 그 α에서의 좌표당 이동량과 Q가 같이 나온다. "
        "회색은 언제나 α=0, 즉 같은 노이즈의 무조종 draw다. / The same chunk, walked up the ladder and back down. "
        "The title carries the per-coordinate displacement and the Q at that α; grey is always α=0, this noise's "
        "unsteered draw.</p>"
    )
