"""The local model answering: why does k* grow with an accurate policy, and what sets its ceiling."""

KSTAR_KO = r"""
<h3>9.6 두 질문에 대한 정량적 답 — 최적 commit 길이의 닫힌 형태</h3>
<p class='sub'>9.5절의 정확해 모델은 <b>결정론</b> 세계였다. 거기서는 \(q=1\)에서 모든 \(k\)가 정확히 동률이 되고,
그 동률을 깨는 것은 tie-break 규칙이었다. 정당한 반문은 이것이다 — <b>"노이즈가 있는 실제 환경에서도 설명이
되어야 하지 않나?"</b> 된다. 그리고 노이즈가 있으면 <b>동률 자체가 생기지 않아 규칙이 필요 없다.</b></p>

<h4>국소 모델: 맞붙는 두 힘</h4>
<div class='tblwrap'><table>
<tr><th>힘</th><th>정체</th><th>\(k\)에 대한 거동</th></tr>
<tr><td><b>노출 손실</b></td><td>commit 중에는 반응할 수 없다. 스텝당 사고 확률은 두 원천 — 환경 불확실성 \(\varepsilon\)와 정책 tail 부정확성 \((1-q)\). 사고가 chunk <b>앞쪽</b>에서 나면 <b>남은 스텝 전부</b>를 망친다.</td><td><b>이차</b>: \(\tfrac12(\rho(s)\varepsilon+(1-q)c)\,k^2\)</td></tr>
<tr><td><b>결정 손실</b></td><td>재질의 한 번마다 <b>불완전한</b> critic·actor가 판단하므로 오차 \(\delta\)를 문다. 사이클당 1회.</td><td>상수 \(\delta\) (사이클당)</td></tr>
</table></div>
<p>\(\rho(s)\)는 "그 상태에서 반응 못 하는 것의 국소 비용"이다(접촉 근처에서 크다). 단위시간당 손실은</p>
\[ L(k)\ =\ \tfrac12\big(\rho(s)\varepsilon+(1-q)c\big)\,k\ +\ \frac{\delta}{k} \]
<p>이고, 미분해 0으로 두면 <b>닫힌 형태</b>가 나온다:</p>
\[ \boxed{\ k^\star(s)\ =\ \min\Big(H,\ \sqrt{\frac{2\delta}{\rho(s)\,\varepsilon+(1-q)\,c}}\Big)\ } \]

<figure><img src="figures/chunking-easy/fig_kstar.png" alt="optimal commitment length"/>
<figcaption><b>왼쪽</b>: 두 힘이 만드는 U자 손실. 점이 최적 \(k^\star\)이고, tail이 정확해질수록(\(q\to1\))
<b>오른쪽으로 이동</b>한다. <b>가운데(Q1)</b>: 가로축은 정책의 tail 오차 \(1-q\)(오른쪽으로 갈수록 개선).
<b>노이즈가 있어도(\(\varepsilon>0\)) \(k^\star\)는 단조 증가</b>하며, \(\varepsilon\)마다 <b>다른 천장</b>(점선)에서
멈춘다. <b>오른쪽(Q2)</b>: 그 천장은 \(1/\sqrt{\varepsilon}\)로 떨어지고, \(\rho(s)\)가 큰 접촉 구간일수록 낮다
(\(\varepsilon=10^{-2}\)에서 자유공간 18.3 vs 접촉 5.8). 모두 계산값.</figcaption></figure>

<h4>질문 1 — 왜 정확해진 정책이 더 긴 prefix로 가는가 (노이즈가 있어도)</h4>
<p>식에서 곧바로: <b>\(q\uparrow\ \Rightarrow\) 분모 \(\downarrow\ \Rightarrow\ k^\star\uparrow\).</b>
\(\varepsilon>0\)이어도 그대로다.</p>
<div class='callout'><span class='k'>직관</span>
짧게 끊는 것은 <b>두 가지 노출</b>을 피하는 수단이다 — ① 부정확한 <b>꼬리</b>에 노출, ② 예측 못 한 <b>사건</b>에
노출. full-chunk improvement는 <b>①만</b> 없앤다. 없앤 만큼 짧게 끊을 이유가 줄고, 그러면 상대적으로 커진
<b>결정 손실 \(\delta/k\)</b>가 \(k^\star\)를 밀어 올린다.</div>
<p><b>그리고 여기서 tie-break가 필요 없다</b> — 이것이 이 절의 요점이다. 노이즈가 있으면 값이 정확히 동률이 되지
않으므로, argmax가 <b>연속적으로</b> 이동한다. 완전 동률(따라서 규칙)은 \(\varepsilon=0\)인 <b>극단에서만</b>
등장한다. 9.5절이 그 극단을 다뤘다면, 이 절은 <b>일반적인 경우</b>를 다룬다.</p>

<h4>질문 2 — 환경 불확실성이 full commitment를 깎는 유도</h4>
<p>\(q=1\)(완벽한 정책)을 넣으면 <b>천장이 그대로 떨어진다</b>:</p>
\[ k_{\rm ceil}(s)\ =\ \min\Big(H,\ \sqrt{\frac{2\delta}{\rho(s)\,\varepsilon}}\Big) \]
<div class='tblwrap'><table>
<tr><th>극한</th><th>천장</th><th>뜻</th></tr>
<tr><td>\(\varepsilon\to0\)</td><td>\(\to H\)</td><td>결정론이면 full commitment가 최적 — <b>정리 2와 일치</b></td></tr>
<tr><td>\(\varepsilon\uparrow\)</td><td>\(\propto 1/\sqrt{\varepsilon}\)</td><td>환경이 불확실할수록 짧게</td></tr>
<tr><td>\(\rho(s)\uparrow\)</td><td>\(\propto 1/\sqrt{\rho}\)</td><td>접촉·정밀 구간은 <b>같은 \(\varepsilon\)에서도</b> 천장이 낮다 → <b>상태별 천장</b> = adaptive가 필요한 이유</td></tr>
<tr><td>\(\delta\to0\)</td><td>\(\to 0\)</td><td><b>완벽한 판단기라면 매 스텝 재질의가 옳다</b></td></tr>
</table></div>
<div class='callout warn'><span class='k'>가장 정직한 한 줄</span>
마지막 행이 중요하다. <b>천장은 \(\varepsilon\) 혼자 정하지 않고 \(\varepsilon\)와 \(\delta\)의 비가 정한다.</b>
"환경 불확실성이 full commitment를 막는다"는 정확히는 <b>"환경 불확실성 대비 우리 판단 정확도의 비가
막는다"</b>이다. 긴 chunk가 유리한 것은 <b>본질적 우월성이 아니라 우리 추정기가 불완전하다는 사실의
귀결</b>이다.</div>
<p>이 국소 모델은 DQC의 전역 바운드와 부합한다: 총 반응성 가치 \(\le\varepsilon_HH\bar H\),
\(\varepsilon_H=3(1-(1-\varepsilon)^{H-1})\approx3(H-1)\varepsilon\) — <b>둘 다 \(\varepsilon\)에 선형</b>이다.
국소 모델은 거기에 "사고가 <b>언제</b> 나는가"의 이차성과 \(\delta\)를 넣어 <b>상태별 \(k^\star\)</b>까지 준다.</p>
<p class='sub'><b>정직한 범위.</b> \(k^\star\) 공식은 <b>정리가 아니라 국소 선형화 모델</b>이다(사고 시점 균등,
손실 선형, 상태 국소성 가정). 정리 A~E는 일반 MDP에서 참이고, 이 공식은 그 정리들이 말하는 <b>경향의 정량적
얼개</b>다. 동시에 <b>새 사전등록 항목</b>이기도 하다 — \(\rho(s)\)와 \(\delta\)를 실측하면 \(k^\star(s)\)를
<b>예측</b>할 수 있고, 관측된 commit 길이가 그 예측을 따르는지가 이 모델의 검정이다.</p>
"""

KSTAR_EN = r"""
<h3>9.6 A quantitative answer to both questions — the closed form for the optimal commitment length</h3>
<p class='sub'>The exactly solvable model of §9.5 lived in a <b>deterministic</b> world. There, at \(q=1\), every
\(k\) ties exactly and a tie-break rule had to settle it. The fair objection is: <b>"shouldn't the account work in
a noisy environment too?"</b> It does — and with noise <b>no exact tie arises, so no rule is needed.</b></p>

<h4>The local model: two opposing forces</h4>
<div class='tblwrap'><table>
<tr><th>Force</th><th>What it is</th><th>Behaviour in \(k\)</th></tr>
<tr><td><b>Exposure loss</b></td><td>While committed you cannot react. The per-step chance of a mishap has two sources — environment stochasticity \(\varepsilon\) and the policy's tail inaccuracy \((1-q)\). A mishap <b>early</b> in a chunk ruins the <b>whole remainder</b>.</td><td><b>quadratic</b>: \(\tfrac12(\rho(s)\varepsilon+(1-q)c)\,k^2\)</td></tr>
<tr><td><b>Decision loss</b></td><td>Every re-query is a decision made by an <b>imperfect</b> critic/actor and costs \(\delta\). One per cycle.</td><td>constant \(\delta\) per cycle</td></tr>
</table></div>
<p>Here \(\rho(s)\) is the local cost of being unable to react (large near contact). The loss per unit time is</p>
\[ L(k)\ =\ \tfrac12\big(\rho(s)\varepsilon+(1-q)c\big)\,k\ +\ \frac{\delta}{k} \]
<p>and minimising gives a <b>closed form</b>:</p>
\[ \boxed{\ k^\star(s)\ =\ \min\Big(H,\ \sqrt{\frac{2\delta}{\rho(s)\,\varepsilon+(1-q)\,c}}\Big)\ } \]

<figure><img src="figures/chunking-easy/fig_kstar.png" alt="optimal commitment length"/>
<figcaption><b>Left</b>: the U-shaped loss the two forces create. The dot is the optimum \(k^\star\), and it moves
<b>right</b> as the tail becomes accurate (\(q\to1\)). <b>Middle (Q1)</b>: the x-axis is the policy's tail error
\(1-q\) (improving to the right). <b>Even with noise (\(\varepsilon>0\)), \(k^\star\) grows monotonically</b> and
stops at a <b>different ceiling</b> for each \(\varepsilon\) (dashed). <b>Right (Q2)</b>: that ceiling falls as
\(1/\sqrt{\varepsilon}\) and is lower where \(\rho(s)\) is large, i.e. near contact (at \(\varepsilon=10^{-2}\):
18.3 in free space vs 5.8 at contact). All values computed.</figcaption></figure>

<h4>Question 1 — why an accurate policy moves to longer prefixes (even with noise)</h4>
<p>Directly from the formula: <b>\(q\uparrow\Rightarrow\) denominator \(\downarrow\Rightarrow k^\star\uparrow\)</b>,
and this holds for \(\varepsilon>0\).</p>
<div class='callout'><span class='k'>the intuition</span>
Breaking short avoids <b>two</b> exposures — ① to an inaccurate <b>tail</b> and ② to unpredicted <b>events</b>.
Full-chunk improvement removes <b>only ①</b>. To that extent the reason to break shrinks, and the now-relatively
larger <b>decision loss \(\delta/k\)</b> pushes \(k^\star\) up.</div>
<p><b>And no tie-break is needed here</b> — that is this section's point. With noise the values never tie exactly,
so the argmax moves <b>continuously</b>. Exact ties (and hence rules) appear only in the \(\varepsilon=0\)
<b>extreme</b>. §9.5 covered that extreme; this section covers the <b>general</b> case.</p>

<h4>Question 2 — deriving how environment uncertainty caps full commitment</h4>
<p>Setting \(q=1\) (a perfect policy) makes the <b>ceiling</b> fall out:</p>
\[ k_{\rm ceil}(s)\ =\ \min\Big(H,\ \sqrt{\frac{2\delta}{\rho(s)\,\varepsilon}}\Big) \]
<div class='tblwrap'><table>
<tr><th>Limit</th><th>Ceiling</th><th>Meaning</th></tr>
<tr><td>\(\varepsilon\to0\)</td><td>\(\to H\)</td><td>full commitment is optimal in a deterministic world — <b>agrees with Theorem 2</b></td></tr>
<tr><td>\(\varepsilon\uparrow\)</td><td>\(\propto 1/\sqrt{\varepsilon}\)</td><td>the noisier the world, the shorter</td></tr>
<tr><td>\(\rho(s)\uparrow\)</td><td>\(\propto 1/\sqrt{\rho}\)</td><td>contact-critical states have a lower ceiling <b>at the same \(\varepsilon\)</b> → a <b>per-state ceiling</b>, which is exactly why adaptivity is required</td></tr>
<tr><td>\(\delta\to0\)</td><td>\(\to 0\)</td><td><b>with a perfect decision maker one should re-query every step</b></td></tr>
</table></div>
<div class='callout warn'><span class='k'>the most honest line</span>
The last row matters. <b>The ceiling is not set by \(\varepsilon\) alone but by the ratio of \(\varepsilon\) to
\(\delta\).</b> "Environment uncertainty prevents full commitment" is, precisely, "<b>uncertainty relative to our
decision accuracy</b> prevents it". Long chunks are not intrinsically superior; their advantage is a
<b>consequence of our estimators being imperfect</b>.</div>
<p>The local model agrees with DQC's global bound: the total value of reactivity is
\(\le\varepsilon_HH\bar H\) with \(\varepsilon_H=3(1-(1-\varepsilon)^{H-1})\approx3(H-1)\varepsilon\) — both
<b>linear in \(\varepsilon\)</b>. The local model adds the quadratic effect of <b>when</b> the mishap occurs, plus
\(\delta\), and thereby delivers a <b>per-state \(k^\star\)</b>.</p>
<p class='sub'><b>Honest scope.</b> The \(k^\star\) formula is a <b>local linearised model, not a theorem</b>
(uniform mishap time, linear loss, state locality). Theorems A–E hold in general MDPs; this formula is the
<b>quantitative skeleton</b> of the tendency they establish. It is also a <b>new pre-registered test</b>: measuring
\(\rho(s)\) and \(\delta\) lets us <b>predict</b> \(k^\star(s)\), and whether the observed commitment lengths follow
that prediction is the test of this model.</p>
"""
