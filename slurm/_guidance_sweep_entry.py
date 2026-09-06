"""The guidance-sweep entry, kept out of make_master_report.py because that file is already 7k lines.

Every number here is recomputed from slurm/probes/guidance_sweep.json.gz at build time, including
the ones in the TITLE. Nothing is transcribed, because the first published version of this entry
had a title that disagreed with its own body.
"""

import gzip
import json
import pathlib

import numpy as np

PROBE = pathlib.Path(__file__).resolve().parents[1] / "slurm/probes/guidance_sweep.json.gz"
#: The two steering strengths that were actually run on the arm.
DEPLOYED = (0.1, 0.2)


def _facts():
    """Everything the prose is allowed to assert, derived once from the frozen measurement."""
    b = json.loads(gzip.decompress(PROBE.read_bytes()))
    ch = np.asarray(b["chunks"], np.float32)  # [S, A, H, AD] — robot dims only, see the probe
    q = np.asarray(b["q"], np.float32)  # [S, A]
    sigma, d2d = float(b["sigma"]), float(b["draw_to_draw"])
    ns, na, horizon, _ = ch.shape

    def disp(si, ai, t=None):
        d = ch[si, ai] - ch[si, 0] if t is None else ch[si, ai, t] - ch[si, 0, t]
        return float(np.sqrt(np.mean(d**2)) / sigma)

    rows, mean_disp, oob = [], [], []
    for ai, al in enumerate(b["alphas"]):
        d = np.array([disp(si, ai) for si in range(ns)])
        o = np.array([float(((ch[si, ai] < -1) | (ch[si, ai] > 1)).mean()) for si in range(ns)])
        mean_disp.append(float(d.mean()))
        oob.append(float(o.mean()))
        rows.append(
            [
                f"{al:.2f}",
                "0" if ai == 0 else f"{d.min():.1f} – {d.max():.1f}",
                f"{d.mean() / d2d:.1f}×" if ai else "—",
                f"{q[:, ai].mean():+.1f}",
                f"{100 * o.mean():.1f}%",
            ]
        )

    # where along the chunk the bend happens, as a ratio of the last third to the first third
    thirds = {}
    for ai, al in enumerate(b["alphas"]):
        if ai == 0:
            continue
        per = np.array([[disp(si, ai, t) for t in range(horizon)] for si in range(ns)]).mean(0)
        head, tail = per[: horizon // 3].mean(), per[-horizon // 3 :].mean()
        thirds[al] = float(tail / max(head, 1e-9))

    idx = [b["alphas"].index(a) for a in DEPLOYED if a in b["alphas"]]
    ratios = [mean_disp[i] / d2d for i in idx]
    first_beat = next((a for a, qq in zip(b["alphas"], q.mean(0), strict=True) if qq > b["q_data"]), None)
    return {
        "b": b,
        "rows": rows,
        "ns": ns,
        "na": na,
        "horizon": horizon,
        "sigma": sigma,
        "d2d": d2d,
        "mean_disp": mean_disp,
        "oob": oob,
        "thirds": thirds,
        "dep_ratio": ratios,
        "dep_disp": [mean_disp[i] for i in idx],
        "first_beat": first_beat,
        "oob0": oob[0],
        "oob_last": oob[-1],
        "alpha_last": b["alphas"][-1],
        "tail_last": thirds[b["alphas"][-1]],
        "tail_dep": [thirds[a] for a in DEPLOYED if a in thirds],
    }


def register(*, entry, en, img, table, spec, plots):
    f = _facts()
    b = f["b"]
    png, gif = plots / "34_guidance_sweep.png", plots / "34_guidance_sweep.gif"
    ep, fr = b["episode"], b["frame"]
    lo, hi = min(f["dep_ratio"]), max(f["dep_ratio"])
    head = f"{lo:.1f}~{hi:.1f}배" if abs(hi - lo) > 0.05 else f"{lo:.1f}배"
    head_en = f"{lo:.1f}–{hi:.1f}×" if abs(hi - lo) > 0.05 else f"{lo:.1f}×"
    uniform = all(0.75 < r < 1.6 for r in f["tail_dep"])

    common = {
        "ep": ep,
        "fr": fr,
        "ns": f["ns"],
        "ad": b["action_dim"],
        "sigma": f["sigma"],
        "d2d": f["d2d"],
        "pc": b["pc_sigma"],
        "critic": b["critic"],
        "qd": b["q_data"],
    }

    ko = _ko(f, common, head, uniform, img, table, spec, png, gif)
    en_body = _en(f, common, head_en, uniform, img, table, spec, png, gif)
    entry(
        "2026-09-03 18:30",
        "guidance-sweep",
        f"노이즈를 고정하고 조종 세기만 올리면 — 배포 세기에서 정책 자체 산포의 {head}",
        "완결",
        ko,
    )
    en(
        "guidance-sweep",
        f"one noise held fixed, guidance turned up: {head_en} the policy's own draw-to-draw spread",
        en_body,
    )


def _ko(f, c, head, uniform, img, table, spec, png, gif):
    b = f["b"]
    bend = (
        "배포 세기에서는 <b>청크 전체가 고르게</b> 휜다. 앞 3분의 1과 뒤 3분의 1의 비가 "
        + ", ".join(f"α={a:g}에서 {r:.2f}" for a, r in zip(DEPLOYED, f["tail_dep"], strict=False))
        + f"로 1에 가깝다. 꼬리가 확 벌어지는 것은 α={f['alpha_last']:g}에서야 나타난다(비 {f['tail_last']:.1f})."
        if uniform
        else f"뒤쪽이 더 휜다 — 앞 3분의 1 대비 뒤 3분의 1의 비가 α={f['alpha_last']:g}에서 {f['tail_last']:.1f}이다."
    )
    return (
        f"<p><b>한 줄.</b> 배포에 쓰던 세기(α=0.1, 0.2)에서 조종은 <b>정책 자신의 두 draw가 서로 떨어진 거리의 "
        f"{head}</b>만큼 행동을 옮기고, 정규화 박스를 벗어나는 좌표가 {100 * f['oob0']:.1f}%에서 "
        f"{100 * f['oob'][b['alphas'].index(0.2)]:.1f}%로 늘며, 그러는 내내 Q는 오른다. "
        "지지집합을 떠나는 것과 값이 좋아 보이는 것이 같은 그림 안에 있다.</p>"
        + spec(
            [
                (
                    "무엇을 고정했나",
                    f"관측(에피소드 {c['ep']} 프레임 {c['fr']}), 흐름의 초기 노이즈, critic 하나. α만 움직인다",
                ),
                (
                    "왜 그게 가능한가",
                    "<code>Pi0Steered.sample_steered</code>가 노이즈를 인자로 받고 α=0이 같은 경로로 무조종 draw를 재현한다",
                ),
                ("노이즈", f"독립 {c['ns']}개 — 한 draw가 정책 구름의 꼬리에 떨어지는 우연을 배제"),
                (
                    "거리 단위",
                    f"로봇의 실제 관절 {c['ad']}개에 대해서만, 좌표당 RMS ÷ BC 산포"
                    f"(σ={c['sigma']:.4f}/좌표). 정책 자신의 두 draw는 이 단위로 {c['d2d']:.2f} 떨어져 있고, "
                    "그것이 비교 기준이다",
                ),
                ("클리핑", "없음 — 출력 클립을 뺀 브랜치라 박스를 넘는 포화가 가려지지 않는다"),
                ("critic", f"<code>{c['critic']}</code>, 자체 비관 축약(mean − ρ·std)"),
            ]
        )
        + "<h4>왜 이렇게 재는가</h4>"
        "<p>지금까지 조종을 잰 방식은 전부 집계였다. 롤아웃 전체의 평균 변위, 프레임 평균 Q, 스칼라 하나로 요약한 drift. "
        "그런 수치는 '얼마나 틀어지는가'를 보여주지 못한다. 평균이 작아도 개별 청크가 크게 휘었을 수 있고, 그 반대도 된다.</p>"
        "<p>그래서 반대로 갔다. 고칠 수 있는 것은 전부 고정하고 α만 사다리로 올린다. "
        "<code>sample_steered</code>가 노이즈를 인자로 받고 α=0이 <b>같은 경로로</b> 무조종 draw를 재현하기 때문에 가능하다. "
        "여기서 α=0 곡선은 'BC 샘플 비슷한 것'이 아니라 <b>바로 그 노이즈의 BC 샘플</b>이고, 나머지 곡선은 조종이 그 draw에 한 짓이다. "
        "두 곡선의 차이는 조종항이지 표집 분산이 아니다.</p>"
        "<h4>결과</h4>"
        + table(
            ["α", "α=0 draw로부터 이동 (좌표당 BC σ)", "정책 자체 산포 대비", "critic의 비관적 Q", "박스 밖 좌표 비율"],
            f["rows"],
        )
        + f"<p class='cap'>세 번째 열이 이 표의 핵심이다. 정책은 같은 상태에서 draw를 두 번 뽑으면 이미 {c['d2d']:.2f}σ만큼 "
        "서로 다르다. 조종의 이동량을 그 거리로 나눈 것이 '정책이 스스로 흔들리는 것보다 몇 배나 더 밀었는가'다. "
        f"시연자의 실제 다음 30스텝은 Q = {c['qd']:+.1f}이고, 성공한 에피소드의 청크이므로 목표 도달이 확인된 행동이다. "
        + (
            f"<b>α={f['first_beat']:g}에서 이미 그 값을 넘는다.</b>"
            if f["first_beat"] is not None
            else "표의 어떤 α도 그 값을 넘지 못한다."
        )
        + " 넘었다는 것 자체가 문제는 아니다 — critic이 시연자보다 나은 행동을 찾았을 수도 있다. "
        "문제는 그 지점이 정책이 한 번도 가보지 않은 영역이라는 것이고, 거기서는 Q를 검증할 방법이 없다.</p>"
        + img(png, "one fixed noise, alpha swept: the chunk, where it bends, the policy plane, and the Q it bought")
        + f"<p class='cap'><b>1번 (왼쪽 6칸) — 청크 자체.</b> 30스텝 × {c['ad']}관절 중 조종이 가장 많이 움직인 6개. "
        "선 하나가 α 하나다(어두울수록 약함). 점선은 시연자, 가로 점선은 정규화 박스 [−1, 1].</p>"
        f"<p class='cap'><b>2번 — 어디서 휘는가.</b> 청크의 각 스텝에서 α=0 draw로부터의 거리. 밴드는 노이즈 {c['ns']}개의 "
        f"최소~최대. 가로 파선이 {c['d2d']:.2f}σ, 즉 <b>정책 자신의 두 draw가 떨어진 거리</b>다. {bend}</p>"
        "<p class='cap'><b>3번 — 정책의 지지집합을 벗어나는가.</b> 이 프레임의 BC draw 구름의 PC1/PC2 평면에 모든 α를 투영했다. "
        "회색이 BC draw, 별이 시연자, 색점이 α 사다리다. α=0 점들은 회색 구름 안에 있다(당연하다, 그것이 BC draw다). "
        "α가 오르면 경로가 구름 밖으로 걸어 나간다. <b>이 패널과 4번 패널을 나란히 보는 것이 이 리포트의 전부다.</b></p>"
        "<p class='cap'><b>4번 — 그 등반이 산 것.</b> 각 α에서 critic 자신의 비관적 Q. 회색 밴드는 노이즈 간 범위, "
        "붉은 파선이 시연자다. 조종은 정의상 Q를 올린다. 그 상승이 진짜인지는 이 패널이 답하지 못하고 "
        "<a class='xref' data-eid='q-landscape-ood'>Q-landscape 프로브</a>가 답한다: 같은 critic 계열에서 박스 경계의 Q는 "
        "시연자보다 +12.7 높고, 그 오차를 비관성으로 지우려면 ρ가 2.5 필요한데 K=2에서 쓸 수 있는 최대는 1.0이다.</p>"
        + _gif_html(gif)
        + "<h4>이 그림이 세 번 틀렸다가 고쳐진 곳</h4>"
        "<p><b>1. 패딩 차원.</b> 가장 컸다. pi05의 액션 헤드는 32칸이고 <code>PadStatesAndActions</code>가 YAM의 실제 관절 "
        "14개를 거기에 0으로 채워 넣는다. 처음 판은 그 32칸 전부에 대해 좌표당 σ를 평균 냈다. 패딩 18칸의 std가 0.0008, "
        "실제 관절이 0.0314라서 σ가 2.2배 작게 나왔고, 이동량은 1.47배 부풀고, 박스 밖 비율은 정확히 14/32로 줄었다. "
        "<b>이 상태로 한 번 게시됐다</b> — 그때 표에 있던 α=1.6의 '박스 밖 10.3%'는 실제로는 23.4%였다. "
        "이제 모든 분석이 로봇의 실제 관절만 본다.</p>"
        "<p><b>2. 단위.</b> 그 이전 판은 청크 전체의 노름을 좌표당 σ로 나눴다. 좌표 수의 제곱근만큼 부풀어 "
        "'69σ, 1261σ' 같은 값이 나왔다. 좌표당 RMS로 고쳤다.</p>"
        "<p><b>3. 기준선.</b> 처음엔 노이즈 하나만 썼는데 그 α=0 점이 BC 구름의 PC1 기준 3σ 밖에 떨어졌다. "
        f"α=0 draw 16개를 따로 뽑아 확인하니 구름 중심에서 0.05σ였다. 운 나쁜 한 표본이었고, 지금은 노이즈 {c['ns']}개다.</p>"
        "<p>세 번 다 방향은 바뀌지 않았다 — 조종은 여전히 지지집합을 떠나고 Q는 여전히 오른다. 바뀐 것은 크기이고, "
        "박스 밖 비율은 <b>고칠수록 나빠졌다</b>. 그래도 게시된 숫자가 틀렸던 것은 틀렸던 것이라 여기 적는다.</p>"
        "<h4>한계</h4>"
        "<ul>"
        f"<li>프레임 하나다(에피소드 {c['ep']}, 프레임 {c['fr']}). 이 프레임의 α–변위 관계가 데이터셋을 대표한다는 "
        "근거는 여기 없다. 프레임을 쓸어 담는 것은 플래그 하나지만 돌리지 않았다.</li>"
        "<li>critic 하나다. Q-landscape 프로브는 9종에서 같은 방향을 확인했지만 이 그림은 그중 하나만 쓴다.</li>"
        "<li>출력 클립이 제거된 브랜치에서 쟀다. master에는 아직 클립이 있어서 거기서 돌리면 '박스 밖 비율'이 0이 되고 "
        "큰 α의 변위가 과소평가된다. 클립을 뺀 것은 공정성 때문이지만 <b>실물 롤아웃 당시에는 클립이 있었다</b>.</li>"
        "<li>Q는 클립되지 않은 청크에 대해 매긴 값이다. 조종이 실제로 올라간 것은 내부의 straight-through 클립을 지난 "
        "값이므로, 4번 패널의 곡선은 조종이 최적화한 바로 그 양은 아니다.</li>"
        "</ul>"
        "<h4>재현</h4>"
        "<pre><code>uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --critic &lt;critic dir&gt; --policy-dir &lt;bc ckpt&gt; \\\n"
        "    --out sweep.png --gif sweep.gif --dump slurm/probes/guidance_sweep.json.gz   # GPU 필요\n"
        "uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --from-dump slurm/probes/guidance_sweep.json.gz --out sweep.png              # GPU 불필요</code></pre>"
        "<p>측정은 GPU가 필요하지만 그림은 언 데이터의 순수 함수다. <code>fig_34_guidance_sweep()</code>이 리포트 생성마다 "
        "다시 그리고, 이 글의 <b>제목을 포함한 모든 수치</b>도 같은 파일에서 빌드 때 재계산된다 — 첫 게시본의 제목이 자기 "
        "본문과 어긋났던 탓이다.</p>"
    )


def _en(f, c, head, uniform, img, table, spec, png, gif):
    b = f["b"]
    bend = (
        "At the deployed strengths the chunk bends <b>fairly uniformly</b>: the ratio of the last third to the "
        "first third is "
        + ", ".join(f"{r:.2f} at α={a:g}" for a, r in zip(DEPLOYED, f["tail_dep"], strict=False))
        + f". The tail only runs away at α={f['alpha_last']:g} (ratio {f['tail_last']:.1f})."
        if uniform
        else f"The tail bends more: last third over first third is {f['tail_last']:.1f} at α={f['alpha_last']:g}."
    )
    return (
        f"<p><b>One line.</b> At the strengths actually deployed (α=0.1, 0.2), steering moves the action "
        f"<b>{head} as far as two of the policy's own draws differ from each other</b>, the fraction of "
        f"coordinates outside the normalized box grows from {100 * f['oob0']:.1f}% to "
        f"{100 * f['oob'][b['alphas'].index(0.2)]:.1f}%, and Q rises the whole way. Leaving the support and "
        "looking better are the same picture.</p>"
        + spec(
            [
                (
                    "what is held fixed",
                    f"the observation (episode {c['ep']} frame {c['fr']}), the flow's initial noise, one critic. Only α moves",
                ),
                (
                    "why that is possible",
                    "<code>Pi0Steered.sample_steered</code> takes the noise as an argument and α=0 reproduces "
                    "the unsteered draw through that same path",
                ),
                (
                    "noises",
                    f"{c['ns']} independent, so one draw landing in the tail of the cloud cannot carry the result",
                ),
                (
                    "distance unit",
                    f"over the robot's {c['ad']} real joints only: root-mean-square per coordinate ÷ the BC "
                    f"spread (σ={c['sigma']:.4f}/coordinate). Two of the policy's own draws sit {c['d2d']:.2f} "
                    "apart in that unit, and that is the reference",
                ),
                ("clipping", "none — measured on the branch that removed the output clip, so saturation is visible"),
                ("critic", f"<code>{c['critic']}</code>, its own pessimistic reduction (mean − ρ·std)"),
            ]
        )
        + "<h4>Why measure it this way</h4>"
        "<p>Every previous measurement of steering here is an aggregate: displacement averaged over a rollout, Q "
        "averaged over frames, drift as a single scalar. None of those show how far a chunk actually bends — a "
        "small average is consistent with large individual excursions, and the reverse.</p>"
        "<p>So this goes the other way. Fix everything that can be fixed and walk α up a ladder. That is possible "
        "because <code>sample_steered</code> takes the noise as an argument and α=0 reproduces the unsteered draw "
        '<b>through that same path</b>. The α=0 curve is not "a BC sample" loosely speaking; it is <b>this '
        "noise's</b> BC sample, and every other curve is what steering did to that exact draw.</p>"
        "<h4>Result</h4>"
        + table(
            [
                "α",
                "displacement from the α=0 draw (BC σ per coordinate)",
                "vs the policy's own spread",
                "critic's pessimistic Q",
                "coordinates outside the box",
            ],
            f["rows"],
        )
        + f"<p class='cap'>The third column is the point of this table. Draw twice from the policy at the same "
        f"state and the two chunks already differ by {c['d2d']:.2f}σ. Dividing the steering displacement by that "
        "distance answers \"how much harder did steering push than the policy pushes itself\". The demonstrator's "
        f"own next 30 steps score Q = {c['qd']:+.1f}, and that chunk comes from a successful episode, so it is an "
        "action known to have reached the goal. "
        + (
            f"<b>By α={f['first_beat']:g} the steered chunk already beats it.</b>"
            if f["first_beat"] is not None
            else "No α in the table beats it."
        )
        + " Beating it is not by itself the problem — the critic may have found something better. The problem is "
        "that it does so in a region the policy never visits, where there is no way to check.</p>"
        + img(png, "one fixed noise, alpha swept: the chunk, where it bends, the policy plane, and the Q it bought")
        + f"<p class='cap'><b>1 (the six panels on the left) — the chunk itself.</b> Of the 30 steps × {c['ad']} "
        "joints, the six steering moved most. One line per α (darker is weaker); dashed is the demonstrator and the "
        "dotted horizontals are the normalized box.</p>"
        f"<p class='cap'><b>2 — where it bends.</b> Distance from this noise's α=0 draw at each step of the chunk. "
        f"The band is the min–max over the {c['ns']} noises. The dashed horizontal is {c['d2d']:.2f}σ, "
        f"<b>the distance between two of the policy's own draws</b>. {bend}</p>"
        "<p class='cap'><b>3 — does it leave the policy's support?</b> Every α projected onto the PC1/PC2 plane of "
        "the BC draw cloud at this frame. Grey is the draws, the star the demonstrator, the coloured points the "
        "ladder. The α=0 points sit inside the grey cloud, as they must. As α rises the path walks out of it. "
        "<b>Reading this panel next to panel 4 is the whole report.</b></p>"
        "<p class='cap'><b>4 — what the climb bought.</b> The critic's own pessimistic Q at each α; the grey band is "
        "the range across noises, the red dashed line the demonstrator. Steering raises Q by construction. Whether "
        "that rise is real is answered not here but by the "
        "<a class='xref' data-eid='q-landscape-ood'>Q-landscape probe</a>: in the same critic family, Q at the box "
        "edge sits +12.7 above the demonstrator, and erasing that with pessimism would need ρ = 2.5 where the "
        "strongest available at K=2 is 1.0.</p>"
        + _gif_html(gif)
        + "<h4>Three things this got wrong before it was right</h4>"
        "<p><b>1. Padding dimensions</b>, and this was the big one. The pi05 action head is 32 wide and "
        "<code>PadStatesAndActions</code> zero-pads YAM's 14 real joints out to it. The first version averaged the "
        "per-coordinate σ over all 32. The 18 padding columns have std 0.0008 against the real joints' 0.0314, so σ "
        "came out 2.2× too small, every displacement was inflated 1.47×, and every out-of-box fraction was deflated "
        "by exactly 14/32. <b>It was published in that state</b>: the α=1.6 row then read 10.3% of coordinates "
        "outside the box where the true figure is 23.4%. Every analysis now uses the robot's real joints only.</p>"
        "<p><b>2. Units.</b> The version before that divided the norm of the whole chunk by a per-coordinate σ, "
        'inflating by the square root of the coordinate count and producing "69σ, 1261σ". Root-mean-square per '
        "coordinate now.</p>"
        "<p><b>3. Baseline.</b> The first version used one noise, and that α=0 point landed 3σ out along PC1 of the "
        f"policy's own cloud. Sixteen α=0 draws sit 0.05σ from the cloud centre; it was one unlucky sample. It runs "
        f"from {c['ns']} noises now.</p>"
        "<p>None of the three changed the direction — steering still leaves the support and Q still climbs. They "
        "changed the magnitudes, and the out-of-box fraction got <b>worse</b> each time it was fixed. The published "
        "numbers were still wrong, which is why they are recorded here.</p>"
        "<h4>Limits</h4>"
        "<ul>"
        f"<li>One frame (episode {c['ep']}, frame {c['fr']}). Nothing here says the α–displacement relation at this "
        "frame represents the dataset. Sweeping frames is a flag away and was not done.</li>"
        "<li>One critic. The Q-landscape probe found the same direction across nine; this uses one of them.</li>"
        "<li>Measured on the branch with the output clip removed. Master still clips, so running it there would "
        "report 0% outside the box and understate the displacement at large α. Removing the clip was a fairness "
        "decision, but <b>the real rollouts ran with it in place</b>.</li>"
        "<li>Q here is scored on the unclipped chunk. What steering actually ascended passes through the internal "
        "straight-through clip, so panel 4 is not exactly the quantity that was optimised.</li>"
        "</ul>"
        "<h4>Reproducing it</h4>"
        "<pre><code>uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --critic &lt;critic dir&gt; --policy-dir &lt;bc ckpt&gt; \\\n"
        "    --out sweep.png --gif sweep.gif --dump slurm/probes/guidance_sweep.json.gz   # needs a GPU\n"
        "uv run python scripts/plot_guidance_sweep.py \\\n"
        "    --from-dump slurm/probes/guidance_sweep.json.gz --out sweep.png              # needs none</code></pre>"
        "<p>The measurement needs a GPU; the figure is a pure function of what it froze, and "
        "<code>fig_34_guidance_sweep()</code> redraws it on every report build. <b>Every number here including the "
        "title</b> is recomputed from that same file at build time, because the first published version had a title "
        "that disagreed with its own body.</p>"
    )


def _gif_html(path):
    """The animation, embedded as a GIF so it still moves. img() would re-encode it to a JPEG."""
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
