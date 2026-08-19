# ---------------------------------------------------------------- AOLC (easy, KO)
AOLC_KO = r"""
<h3>2.5 AOLC — "언제 확인할 것인가"를 상태마다 바꾸면</h3>
<p>2절의 ε는 <b>고정된 시점</b>에서 잰 것이다: "레시피를 눈 감고 따라 한 뒤 <b>정확히 h스텝 뒤</b>에 어디 있나"를
데이터와 비교한다. 그런데 우리(그리고 AQC)는 <b>상태마다 다른 길이</b> \(\kappa(s)\)만큼만 실행하고 다시 본다.
그러면 확인 시점이 <b>들쭉날쭉</b>해진다.</p>
<div class='callout'><span class='k'>비유</span>
OLC는 "모든 요리를 <b>10분 뒤에</b> 확인한다"는 규칙이고, AOLC는 "<b>요리마다 다른 시각에</b> 확인한다"는
규칙이다. 후자에서 품질 보장을 하려면, 그 <b>제각각인 확인 시각 전부</b>에 대해 어긋남이 작아야 한다.</p></div>
<p>AQC의 <b>Adaptive Open-Loop Consistency (AOLC)</b>가 정확히 그것이다 — OLC의 두 조건에서 고정 오프셋
\(t+h\)를 상태 의존 오프셋 \(t+\kappa(s_t)\)로 바꾼다:</p>
\[ D_{TV}\big(P_D(s_{t+\kappa(s_t)},a_{t+\kappa(s_t)}\mid s_t)\ \big\|\ P^\circ_{D,\kappa}(\cdot\mid s_t)\big)\le\varepsilon_{\mathcal K} \]
<p>그리고 \(\kappa\)가 상수 \(k\)이면 원래 OLC로 정확히 되돌아간다(Prop H.3) — 그래서 "일반화"라고 부른다.</p>
<div class='callout warn'><span class='k'>정직한 평가 — 우리가 확인한 것</span>
정의만 놓고 보면 이것은 <b>확인 시점을 치환한 것</b>이고, 새 기계가 추가되지는 않는다. 그래서 두 가지가
미해결로 남는다. ① 이 조건은 <b>한 번의 적응 스텝</b>만 제약하는데 실제 실행은 그것을 <b>여러 번 이어 붙인다</b>
(스텝별 보장이 자동으로 전체 보장이 되지는 않는다). ② 정의에서 \(\kappa\)는 <b>미리 주어진 고정 함수</b>인데,
실제 selector는 critic에 의존해 <b>학습 중 계속 변한다</b> — 실제로 AQC의 Theorem H.14는 <b>oracle 선택자</b>
아래의 AOLC를 가정하면서 정작 <b>학습된 선택자</b>로 실행되는 정책을 바운드한다.</div>
"""

# ---------------------------------------------------------------- BoN comparison (KO)
BON_KO = r"""
<h3>8.6 "BoN도 정책 개선 아닌가?" — 맞다, 그런데 어디까지인가</h3>
<p>좋은 반론이다. AQC·ACSAC가 쓰는 <b>best-of-N(BoN)</b>도 분명 개선이다: 후보 \(N\)개 중 critic이 가장 높게
치는 것을 고르므로 평균보다는 낫다. 그러니 "우리만 개선한다"는 말은 틀렸다. <b>정확한 차이는 '개선의 한계'가
어디에 걸리느냐다.</b></p>
<p>형식적으로 \(\pi^{BoN}_N(s):=\arg\max_{a\in\{a^1,\dots,a^N\},\,a^i\sim\beta(\cdot\mid s)}Q(s,a)\)로 두자.</p>

<h4>명제 F — BoN의 개선량은 "분위수"에서 멈춘다</h4>
<p>\(N\)개 표본의 최댓값은 \(\beta\) 아래 \(Q\) 분포의 <b>최대 순서통계량</b>이고, 이는 그 분포의
\(\tfrac{N-1}{N}\)-<b>분위수</b>를 추정한다. 이것은 우리 주장이 아니라 DQC 자신이 \(\kappa_b=(N-1)/N\)를
고르는 근거로 명시한 사실이다:</p>
<blockquote>"the Q-value obtained from best-of-N sampling can be seen as the largest order statistic of a random
batch (of size N) of the behavior Q-values. Such statistic estimates the behavior Q-value distribution's
\((N-1)/N\)-quantile" (DQC §5)</blockquote>
<p>즉 <b>BoN은 최댓값이 아니라 고정 분위수를 향한다.</b> \(N=10\)이면 90번째 백분위수다. \(N\)을 늘리는 것 외에
그 천장을 올릴 방법이 없다.</p>

<h4>명제 G — 그리고 그 분위수는 chunk 차원의 저주를 받는다</h4>
<p>chunk 공간은 \(\mathcal A^H\)이고 실제 차원은 \(D=H\times\dim\mathcal A\)다 (예: \(H=30\), 14-DoF →
<b>420차원</b>). 좌표마다 독립적으로 확률 \(p<1\)로 "좋은 범위"에 들어간다고 보면, 무작위 표본 하나가 전 좌표에서
좋을 확률은 \(p^D\)이고, 최댓값에 다가가려면 \(N\sim p^{-D}\)가 필요하다. 실제 \(N\)은 <b>10~32</b>다.
반면 <b>gradient 기반 개선(우리 actor)은 이 저주를 지불하지 않는다</b> — 표본을 고르는 게 아니라 파라미터를
움직이기 때문이다.</p>

<h4>명제 H — support 천장</h4>
<p>BoN이 내놓는 행동은 언제나 \(\mathrm{supp}\,\beta(\cdot\mid s)\) 안에 있다. 따라서 임의의 \(N\)에 대해</p>
\[ V^{\pi^{BoN}_N}\ \le\ V^{\beta\text{-in-support greedy}}\ \le\ V^\star_H \]
<p>이고, 등호는 <b>도달 가능한 모든 상태에서 최적 chunk가 \(\beta\)의 support 안에 있을 때만</b> 성립한다.
우리 actor의 제약은 <b>부드럽다</b> — \(-Q+\alpha\cdot\|\mu_\omega-\mu_\theta\|^2\)에서 \(\alpha\)를 지불하면
support 밖으로 <b>이동할 수 있다</b>. (정직하게: 그래서 우리 천장도 \(V^\star_H\)가 아니라
<b>\(\alpha\)-정규화된 최적</b>이며, \(\alpha\to0\)에서 \(V^\star_H\)로 간다. 대신 OOD 위험이 커진다 —
이것이 우리가 조절하는 손잡이다.)</p>

<h4>명제 I — 결정적 차이: 선택만으로는 curriculum이 생기지 않는다</h4>
<p><b>명제.</b> \(\beta\)가 갱신되지 않으면(BoN/선택 전용), "엄격히 짧은 것이 나은" 집합
\(\mathcal S_<\)는 \((\beta,Q)\)만의 함수다. critic이 수렴하면 \(\mathcal S_<\)도 <b>어떤 고정 집합</b>으로
수렴하며 \(\varnothing\)으로 가지 않는다(\(\beta\)가 이미 최적이 아닌 한). 따라서 <b>평균 commitment 길이는
정상 상태에 머문다 — curriculum이 발생하지 않는다.</b></p>
<p><i>증명.</i> \(\mathcal S_<\)의 정의는 \(\beta\)와 값함수에만 의존하고, 알고리즘의 어느 단계도 \(\beta\)를
바꾸지 않는다. ∎</p>
<div class='callout'><span class='k'>그래서 정리하면</span>
<b>BoN도 개선이다 — 단, "고정된 제안 분포의 분위수"까지의 개선이다.</b> 제안 분포 자체는 영원히 그대로다.
우리 방식은 개선을 <b>actor의 파라미터에 amortize</b>하므로 제안 분포가 움직이고, 그래서 정리 A가 발동하며
(\(\Delta_{\mathrm{epis}}\to0\)), 그 결과로 정리 D의 curriculum이 나온다. <b>명제 I는 동시에 우리의 사전등록
ablation이기도 하다</b>: 정책 개선을 끄면 평균 길이가 자라지 <b>않아야</b> 한다.</div>
<div class='tblwrap'><table>
<tr><th></th><th>개선이 탐색하는 집합</th><th>천장</th><th>제안 분포가 움직이나</th><th>curriculum</th></tr>
<tr><td><b>BoN (AQC·ACSAC)</b></td><td>고정 \(\beta\)에서 뽑은 \(N\)개 표본</td><td>\(\tfrac{N-1}{N}\)-분위수 ≤ in-support greedy</td><td><b>아니오</b></td><td><b>없음</b> (명제 I)</td></tr>
<tr><td><b>우리 (DH + one-step actor)</b></td><td>\(\mathcal A^H\) 전체, \(\alpha\)로 부드럽게 제약</td><td>\(\alpha\)-정규화 최적 → \(\alpha\to0\)에서 \(V^\star_H\)</td><td><b>예</b> (amortize)</td><td><b>있음</b> (정리 D)</td></tr>
</table></div>
"""

# ---------------------------------------------------------------- the tie question (KO)
TIE_KO = r"""
<h3>8.7 "이상적이면 모든 \(k\)가 같다" — 그렇다. 그런데 왜 실제로는 아닌가</h3>
<p>오래 걸렸던 걱정을 이제 정확히 답할 수 있다. <b>이상적인 극한에서는 정말로 모든 \(k\)가 동률</b>이다.</p>
<p><b>명제 J (이상적 동률).</b> 완전관측·결정론적 dynamics이고 \(\pi=\pi^\star_H\)이면, 모든 \(s\)와 모든
\(k\in\mathcal K\)에 대해 \(Q_k(s,\pi(s))=Q_H(s,\pi(s))\) — 즉 <b>어느 \(k\)를 골라도 값이 같고, 고를 근거가
전혀 없다</b>. (증명은 정리 D와 동일.)</p>
<p>그렇다면 실제로 무엇이 동률을 깨는가? 항등식 하나로 정리된다:</p>
\[ Q_k(s,a)-Q_H(s,a)\ =\ \gamma^k\,\mathbb E\Big[\underbrace{V^{\pi,\kappa}(s_k)}_{\text{지금 새로 정하면}}-\underbrace{W_a(s_k)}_{\text{이미 정해둔 꼬리를 계속하면}}\Big] \]
<p>즉 동률을 깨는 것은 오직 <b>"지금 다시 정하는 것이 이미 정해둔 꼬리보다 얼마나 나은가"</b>다. 그 원천은
정확히 셋이다.</p>
<div class='tblwrap'><table>
<tr><th></th><th>원천</th><th>정책이 좋아지면?</th></tr>
<tr><td><b>(1)</b></td><td><b>epistemic</b> — 정해둔 꼬리가 지금 제안했을 것보다 나쁘다</td><td><b>사라진다</b> (명제 J: \(\pi^\star_H\)에서 둘이 일치)</td></tr>
<tr><td><b>(2)</b></td><td><b>aleatoric</b> — \(s_k\)가 확률적이라 꼬리를 정할 때 볼 수 없었다</td><td>남는다 (총량 ≤ \(\Delta_{\mathrm{react}}\))</td></tr>
<tr><td><b>(3)</b></td><td><b>부분관측</b> — 그 정보가 관측에 없었다</td><td>남는다</td></tr>
</table></div>
<h4>명제 L — 그리고 이것이 위험한 이유: 종점에서 selector가 노이즈가 된다</h4>
<p>selector가 실제로 쓸 수 있는 <b>신호</b>를
\(\eta(\pi):=\sup_s\big[\max_kQ_k(s,\pi(s))-Q_H(s,\pi(s))\big]\)라 하자. 위 표에 의해 정책이 좋아질수록
(1)이 사라지므로 <b>\(\eta(\pi_n)\)은 (2)+(3)만 남을 때까지 줄고, 결정론·완전관측에서는 0으로 간다.</b></p>
<p>그런데 critic의 \(k\)-의존 편향 \(b_k\)(11절)는 <b>정책이 좋아져도 줄지 않는다</b> — 그것은 데이터의
hindsight leakage에서 오는 것이지 정책 품질에서 오는 것이 아니다. 따라서:</p>
<div class='callout warn'><span class='k'>명제 L</span>
\(|b_k-b_H|>\eta(\pi_n)\)이 되는 순간부터 <b>경험적 argmax는 값이 아니라 편향이 결정한다.</b> 그리고
\(\eta(\pi_n)\)은 <b>우리가 목표로 하는 방향으로 갈수록 작아지므로</b>, 편향을 다루지 않는 selector는
<b>정확히 우리가 도달하려는 지점에서 순수 노이즈로 수렴한다.</b></div>
<p>이것이 두 가지를 즉시 정당화한다. ① <b>사전식 규칙의 허용오차 \(\epsilon\)은 장식이 아니다</b> —
\(\epsilon\)을 편향 규모 이상으로 두면, 신호가 편향에 잠긴 구간에서 selector가 <b>추측하기를 거부하고</b>
(동률 처리로) 긴 쪽을 commit한다. ② 그 \(\epsilon\)을 정하려면 <b>\(b_k\)를 실측해야 한다</b> — 11절의 1번
실험이 선택이 아니라 필수인 이유다.</p>
"""

# ---------------------------------------------------------------- derived curve text (KO)
DERIV_KO = r"""
<h4>이 곡선이 어디서 나오는가 (파생 과정)</h4>
<p>위 그림은 임의로 그린 지수함수가 아니라 세 정리를 곱한 결과다.</p>
<ol>
<li><b>수축률</b>: 정리 A에 의해 개선은 \(M_H\)에서의 policy iteration이고, PI는 같은 MDP의 value iteration보다
느리지 않다. \(M_H\)의 할인율이 \(\gamma^H\)이므로 VI의 수축계수가 \(\gamma^H\)이고, 따라서
\[ \Delta_{\mathrm{epis}}(n)\ \le\ \gamma^{Hn}\,\Delta_{\mathrm{epis}}(0) \]</li>
<li><b>끊어서 아직 벌 수 있는 총량</b>: Part III.5의 분해에 의해
\[ B(n)\ :=\ \Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(n)\ \le\ \Delta_{\mathrm{alea}}+\gamma^{Hn}\Delta_{\mathrm{epis}}(0) \]</li>
<li><b>floor의 값</b>: DQC Cor.1 + Prop.4로 \(\Delta_{\mathrm{alea}}\le\varepsilon_HH\bar H\),
\(\varepsilon_H=3(1-(1-\varepsilon)^{H-1})\).</li>
</ol>
<p>그리고 사전식 규칙(허용오차 \(\epsilon\))을 쓰면, \(B(n)\le\epsilon\)이 되는 순간부터 <b>어느 상태에서도
짧게 끊을 근거가 남지 않으므로 full commitment가 선택된다</b>. 그 교차 시점은 닫힌 형태로 나온다:</p>
\[ n^\star\ =\ \Big\lceil \frac{\log\big((\epsilon-\Delta_{\mathrm{alea}})/\Delta_{\mathrm{epis}}(0)\big)}{H\log\gamma}\Big\rceil
\qquad(\Delta_{\mathrm{alea}}<\epsilon\text{ 일 때}) \]
<p>그림의 \(\gamma=0.99,\ H=10,\ \varepsilon=10^{-4}\)에서 floor \(\le 2.82\), 수축률 \(\gamma^H=0.904\),
교차 \(n^\star=29\)가 <b>계산되어</b> 나온 값이다. <b>정직한 단서</b>: 이 유도는 <b>상한</b>에 대한 것이므로
"언제부터 반드시 full commit인가"는 보장하지만, 그 이전 구간의 평균 길이가 <b>매끄럽게</b> 오르는 모양까지
유도하지는 않는다(그건 상태별 advantage 분포에 달렸고, 우리가 실측할 대상이다).</p>
"""

CONV_KO = r"""
<h4>정리 E — 왜 commitment 길이가 <b>필연적으로</b> 길어지고 유한 시간에 수렴하는가</h4>
<p>지금까지는 "끊을 이유가 줄어든다"까지였다. 이제 <b>반드시 길어져서 멈춘다</b>는 것을 보인다. 세 조각을 잇는다.</p>
<ol>
<li><b>압력이 한 방향이다.</b> 정리 A의 policy iteration은 <b>단조</b>다: \(V^{\pi_{n+1},H}\ge V^{\pi_n,H}\).
따라서 \(\Delta_{\mathrm{epis}}(n)\)은 <b>단조 감소</b>하고, "끊어서 벌 수 있는 총량"의 상한
\(B(n)=\Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(n)\)도 <b>단조 감소</b>한다. 짧게 유지시키는 힘은
줄기만 하고 늘지 않는다.</li>
<li><b>모든 상태에서 동시에 줄어든다.</b> 임의의 \(s\)에 대해
\(A(s;\pi_n):=\max_kQ_k(s,\pi_n(s))-Q_H(s,\pi_n(s))\le B(n)\) (Part III.5). \(B(n)\)은 \(s\)에 의존하지 않으므로,
<b>어느 상태도 \(B(n)\)보다 더 강한 "끊을 이유"를 가질 수 없다.</b></li>
<li><b>허용오차가 유한 시간에 이긴다.</b> 사전식 규칙(허용오차 \(\epsilon\))은 \(A(s)\le\epsilon\)인 상태에서
동률로 보고 <b>가장 긴 \(k\)</b>를 택한다. \(B(n)\to\Delta_{\mathrm{alea}}\)이고 기하적으로 감소하므로,
\(\epsilon>\Delta_{\mathrm{alea}}\)이면 유한한 \(n^\star\) 이후 \(B(n)\le\epsilon\)이 되어 <b>모든 상태에서
\(\kappa_n\equiv H\)</b>가 된다.</li>
</ol>
<p><b>정리 E.</b> \(\epsilon>\Delta_{\mathrm{alea}}\)이면
\(n^\star=\big\lceil\log\!\big((\epsilon-\Delta_{\mathrm{alea}})/\Delta_{\mathrm{epis}}(0)\big)/(H\log\gamma)\big\rceil\)
이후 모든 \(n\ge n^\star\)에서 \(\kappa_n\equiv H\)이다. 즉 <b>평균 commitment 길이는 유한 시간에 \(H\)에
도달하고 그 뒤로 변하지 않는다.</b></p>
<p><b>정리 E′ (floor가 허용오차보다 큰 경우).</b> \(\epsilon\le\Delta_{\mathrm{alea}}\)이면 대신</p>
\[ \limsup_{n\to\infty}\ \{s:\kappa_n(s)<H\}\ \subseteq\ \{s:\ A_{\mathrm{react}}(s)>\epsilon\} \]
<p>즉 <b>끝까지 짧게 끊는 상태는 "진짜로 반응이 필요한" 상태만 남는다.</b> 이것이 aleatoric floor의 정확한
정체이며, 그 집합의 크기가 우리가 측정하려는 <b>skill별 내재적 반응성 수요</b>다.</p>
<div class='callout'><span class='k'>왜 "필연적"인가</span>
길이를 짧게 붙잡아 두는 힘은 오직 \(A(s)>\epsilon\) 하나인데, 그 힘의 <b>상한이 단조 감소</b>하고 그 극한이
<b>회수 불가능한 부분(aleatoric)</b>이다. 즉 <b>줄어들 수 있는 이유는 전부 줄어들고, 남는 이유는 원래부터
정당한 것뿐</b>이다. 그래서 길이는 늘어날 수밖에 없고, 정당한 이유가 남는 지점에서 멈춘다.</div>
<p class='sub'><b>정직한 단서.</b> 이 논증은 <b>상한</b>에 대한 것이라 "유한 시간에 반드시 \(H\)에 도달"과
"극한에서 floor 집합만 남음"을 보장하지만, 중간 과정에서 <b>어떤 개별 상태의 길이가 일시적으로 줄어드는 것</b>을
배제하지는 않는다(개별 \(A(s;\pi_n)\)의 단조성은 성립하지 않을 수 있다). 우리가 그림으로 보고할 것은
<b>평균</b> 길이이며, 이 논증이 보장하는 것도 평균과 종점이다.</p>
"""
