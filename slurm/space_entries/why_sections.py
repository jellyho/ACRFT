WHY_KO = r"""
<h3>9.5 왜 길이가 <b>필연적으로</b> 길어지는가 — 규칙이 아니라 기제로</h3>
<p class='sub'>여기서 회의적인 독자가 정당하게 반박할 수 있다: <i>"방금 것은 '이득이 작으면 길게 간다'는
<b>규칙을 선언</b>한 것 아닌가? 그건 가정이지 결과가 아니다."</i> 맞는 지적이다. 그래서 이 절에서는
<b>어떤 tie-break 규칙도 쓰지 않고</b>, 정확히 풀리는 모델 하나로 <b>기제 자체</b>를 보인다.</p>

<h4>정확히 풀리는 모델</h4>
<p>복도를 걷는다. 매 스텝 정답 행동이 하나 있고, 맞으면 보상 1, 틀리면 0. chunk 정책이 \(H\)개 행동을
제안하는데 — 여기가 핵심 관찰이다 — <b>첫 슬롯은 항상 맞고, 뒤 슬롯들은 각각 확률 \(q\)로 맞는다</b>.
(이것은 인위적 가정이 아니라 chunk 정책의 잘 알려진 성질이다: <b>앞 타임스텝이 뒤 타임스텝보다 정확하다</b>.
\(q\)를 "tail 정확도"라 부르자.) \(k\)만큼 commit한다는 것은 슬롯 \(1..k\)를 실행하고 다시 질의하는 것이므로,
<b>짧게 끊는 것은 정확한 앞 슬롯만 골라 쓰는 방법</b>이다.</p>
<p>모든 값이 닫힌 형태로 나온다:</p>
\[ V(k)=\frac{1+q\,\gamma\,\frac{1-\gamma^{k-1}}{1-\gamma}}{1-\gamma^{k}},\qquad
\boxed{\ V(1)=\frac{1}{1-\gamma}\ } \]
\[ \underbrace{V(1)-V(k)}_{\text{끊는 것의 이득}}\ =\ \frac{\gamma\,(1-q)\,(1-\gamma^{k-1})}{(1-\gamma)(1-\gamma^{k})} \]

<figure><img src="figures/chunking-easy/fig_why.png" alt="why the length must grow"/>
<figcaption><b>왼쪽</b>: tail 정확도 \(q\)가 올라갈수록 \(V(k)\) 곡선이 <b>평평해지고</b>, \(q=1\)에서는
<b>모든 \(k\)의 값이 정확히 같다</b>(완전 동률 — 끊어서 얻을 것이 0). <b>오른쪽</b>: 끊는 것의 이득은
<b>\((1-q)\)에 정확히 비례</b>하는 직선이고 \(q=1\)에서 정확히 0이 된다. 즉 <b>끊을 이유의 정체가
tail의 부정확성 하나</b>임이 식으로 드러난다. 모두 계산된 값이다(모의실험 아님).</figcaption></figure>

<h4>이 식에서 곧바로 나오는 세 가지</h4>
<div class='tblwrap'><table>
<tr><th></th><th>관찰</th><th>뜻</th></tr>
<tr><td><b>①</b></td><td>이득 \(\propto(1-q)\), 그리고 \(q=1\)에서 정확히 0</td><td><b>끊을 이유는 오직 tail 부정확성</b>이다. 다른 이유가 없다(이 모델엔 확률성도 부분관측도 없으므로 aleatoric floor가 0이다 — 정리 2의 조건).</td></tr>
<tr><td><b>②</b></td><td>\(V(1)=\frac{1}{1-\gamma}\)에는 <b>\(q\)가 아예 없다</b></td><td>짧은 \(k\)로 평가·개선하면 <b>tail 정확도가 목적함수에 문자 그대로 등장하지 않는다</b>. 그러니 아무리 오래 개선해도 \(q\)는 움직이지 않는다 — <b>명제 E의 산수 버전</b>이다.</td></tr>
<tr><td><b>③</b></td><td>\(k=H\)에서 개선하면 \(q\)가 올라간다</td><td>그리고 이득은 \(q\)에 대해 <b>단조 감소</b>한다. 즉 개선은 끊을 이유를 <b>소비</b>한다.</td></tr>
</table></div>

<h4>그래서 "필연적"이라는 말의 정확한 뜻</h4>
<div class='callout'><span class='k'>핵심 논증</span>
<b>끊을 이유는 오직 소비되기만 하고 결코 다시 채워지지 않는 자원이다.</b><br><br>
① 끊는 유일한 동기는 "내가 이미 정해둔 꼬리가, 지금 다시 정하는 것보다 나쁘다"이다.
② 개선 단계가 하는 일은 <b>바로 그 꼬리를 좋게 만드는 것</b>이다(정리 A: \(M_H\)에서의 policy iteration,
따라서 <b>단조</b>).
③ 그리고 이 루프에는 <b>꼬리를 나쁘게 만드는 어떤 기제도 없다</b>.<br><br>
따라서 끊을 이유는 <b>단조적으로 줄어들 수밖에 없고</b>, 줄어들 수 없는 부분(aleatoric)만 남는다. 길이가
길어지는 것은 우리가 <b>그렇게 되도록 규칙을 만든 결과가 아니라</b>, 짧게 붙잡아 두던 힘이 <b>사라진 결과</b>다.</div>
<p>위 모델에서는 이것이 <b>tie-break 규칙 없이</b> 그대로 보인다: \(q<1\)이면 greedy argmax가 \(k=1\)을
고르고(끊는 이득이 <b>엄격히</b> 양수), \(q\to1\)에서 그 이득이 <b>연속적으로 0으로</b> 가며, \(q=1\)에서는
모든 \(k\)가 동률이 된다. 규칙은 그 <b>동률 지점에서만</b> 개입한다 — 그리고 실제 세계에서 그 동률 지점이
편향에 잠기기 때문에(명제 L) 허용오차가 필요한 것이다.</p>

<h4>일반적 진술과 정직한 범위</h4>
<p>일반 MDP에서는 정책 품질이 하나의 스칼라 \(q\)가 아니므로, 위 단조성은 <b>상한 수준</b>에서 성립한다
(정리 E): \(A(s;\pi_n)\le\Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(n)\)이고 \(\Delta_{\mathrm{epis}}(n)\)은
정리 A에 의해 <b>단조 감소</b>한다. 개별 상태의 길이가 도중에 일시적으로 줄어드는 것은 배제되지 않는다.
보장되는 것은 <b>(i) 끊을 이유의 상한이 단조 감소</b>, <b>(ii) 극한에서 남는 것은 aleatoric floor뿐</b>,
<b>(iii) \(\epsilon>\Delta_{\mathrm{alea}}\)이면 유한 시간에 전면 commit</b>이다.</p>
<div class='callout warn'><span class='k'>그래서 이것은 반증 가능하다</span>
<b>개선을 끄면 \(q\)는 절대 오르지 않는다</b>(관찰 ②에 의해 짧은 \(k\) 목적함수엔 \(q\)가 없고, 선택만 하는
방식은 정책을 아예 안 건드린다 — 명제 I). 그러면 끊는 이득이 줄지 않으므로 <b>평균 길이가 자라지 않아야
한다</b>. 그것이 우리 사전등록 ablation이고, <b>길이 성장이 정책 개선 때문임을 보이는 유일한 방법</b>이다.
자란다면 우리 설명이 틀린 것이다.</div>
"""

WHY_EN = r"""
<h3>9.5 Why the length <b>must</b> grow — mechanism, not rule</h3>
<p class='sub'>A sceptical reader can object, fairly: <i>"you just <b>declared a rule</b> that says commit long when
the gain is small. That is an assumption, not a result."</i> Correct. So this section uses <b>no tie-break rule at
all</b> and exhibits the <b>mechanism itself</b> in an exactly solvable model.</p>

<h4>An exactly solvable model</h4>
<p>Walk down a corridor. At each step exactly one action is correct: correct → reward 1, wrong → 0. The chunk
policy proposes \(H\) actions and — this is the key observation — <b>its first slot is always correct while each
later slot is correct with probability \(q\)</b>. (This is not an artificial assumption but the well-documented
shape of chunk policies: <b>early timesteps are more accurate than late ones</b>. Call \(q\) the "tail
accuracy".) Committing \(k\) means executing slots \(1..k\) and re-querying — so <b>breaking is a way of using
only the accurate early slots</b>.</p>
<p>Everything is closed-form:</p>
\[ V(k)=\frac{1+q\,\gamma\,\frac{1-\gamma^{k-1}}{1-\gamma}}{1-\gamma^{k}},\qquad
\boxed{\ V(1)=\frac{1}{1-\gamma}\ } \]
\[ \underbrace{V(1)-V(k)}_{\text{advantage of breaking}}\ =\ \frac{\gamma\,(1-q)\,(1-\gamma^{k-1})}{(1-\gamma)(1-\gamma^{k})} \]

<figure><img src="figures/chunking-easy/fig_why.png" alt="why the length must grow"/>
<figcaption><b>Left</b>: as the tail accuracy \(q\) rises the \(V(k)\) curve <b>flattens</b>, and at \(q=1\)
<b>every \(k\) has exactly the same value</b> (a perfect tie — nothing to gain by breaking). <b>Right</b>: the
advantage of breaking is a straight line <b>exactly proportional to \((1-q)\)</b>, hitting zero at \(q=1\). The
identity of the reason to break — <b>the tail's inaccuracy, and nothing else</b> — is visible in the formula.
All values computed, not simulated.</figcaption></figure>

<h4>Three things that follow immediately</h4>
<div class='tblwrap'><table>
<tr><th></th><th>Observation</th><th>Meaning</th></tr>
<tr><td><b>①</b></td><td>the advantage \(\propto(1-q)\), and is exactly 0 at \(q=1\)</td><td><b>The only reason to break is the tail's inaccuracy.</b> There is no other (this model has no stochasticity and no partial observability, so the aleatoric floor is zero — the conditions of Theorem 2).</td></tr>
<tr><td><b>②</b></td><td>\(V(1)=\frac{1}{1-\gamma}\) <b>contains no \(q\) at all</b></td><td>Evaluating/improving at short \(k\) means the tail accuracy <b>literally does not appear in the objective</b>, so no amount of training moves \(q\) — <b>Proposition E, in arithmetic</b>.</td></tr>
<tr><td><b>③</b></td><td>improving at \(k=H\) raises \(q\)</td><td>and the advantage is <b>monotonically decreasing</b> in \(q\). Improvement <b>consumes</b> the reason to break.</td></tr>
</table></div>

<h4>So what "must" precisely means</h4>
<div class='callout'><span class='k'>the core argument</span>
<b>The reason to break is a resource that is only ever consumed and never replenished.</b><br><br>
① The sole motive for breaking is "the tail I already committed to is worse than deciding afresh".
② What the improvement step does is <b>make exactly that tail better</b> (Theorem A: policy iteration in
\(M_H\), hence <b>monotone</b>).
③ And nothing in the loop <b>can make the tail worse</b>.<br><br>
Therefore the reason to break <b>can only shrink</b>, until only the part that cannot shrink (aleatoric) remains.
The length grows <b>not because we wrote a rule saying so</b>, but because the force that was holding it short
<b>disappears</b>.</div>
<p>In the model above this is visible <b>without any tie-break rule</b>: for \(q<1\) the greedy argmax picks
\(k=1\) (the advantage is <b>strictly</b> positive), as \(q\to1\) that advantage goes to zero <b>continuously</b>,
and at \(q=1\) all \(k\) tie. The rule intervenes <b>only at that tie</b> — and a tolerance is needed there
because, in the real world, the tie is drowned in bias (Proposition L).</p>

<h4>The general statement, and its honest scope</h4>
<p>In a general MDP policy quality is not a single scalar \(q\), so the monotonicity above holds at the level of
the <b>bound</b> (Theorem E): \(A(s;\pi_n)\le\Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(n)\) with
\(\Delta_{\mathrm{epis}}(n)\) <b>monotonically decreasing</b> by Theorem A. A transient dip in an individual
state's length is not excluded. What is guaranteed is <b>(i)</b> the upper bound on the reason to break decreases
monotonically, <b>(ii)</b> only the aleatoric floor survives in the limit, and <b>(iii)</b> full commitment is
reached in finite time whenever \(\epsilon>\Delta_{\mathrm{alea}}\).</p>
<div class='callout warn'><span class='k'>and therefore it is falsifiable</span>
<b>With improvement switched off, \(q\) can never rise</b> (by observation ②, the short-\(k\) objective does not
contain \(q\); and a selection-only method never touches the policy at all — Proposition I). Then the advantage of
breaking never shrinks and the <b>mean length must not grow</b>. That is our pre-registered ablation and the
<b>only way to show the growth is caused by policy improvement</b>. If it grows anyway, our account is wrong.</div>
"""
