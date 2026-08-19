"""KO body for the full chunking-theory report (paper-depth). Math is LaTeX for the hub's MathJax."""

KO = r"""
<p class='sub'>이 글은 action chunking RL의 <b>수학</b>을 처음 보는 사람이 끝까지 따라올 수 있게 전부 푼 다음,
그 위에서 <b>우리 기여를 정리와 증명으로</b> 세운다. 다루는 것은 세 편이다. <b>QC</b>(Li·Zhou·Levine,
arXiv:2507.07969, NeurIPS'25)는 chunk critic의 <i>이득</i>만 증명했고, <b>DQC</b>(Li·Park·Levine,
arXiv:2512.10926)가 그 <i>비용</i>을 처음으로 정량화했으며, <b>AQC</b>(Gireesh·Ju·Wang, arXiv:2605.05544)가
그것을 상태 의존 재질의로 일반화했다. 아래 모든 정의·정리·증명 기계는 원문 PDF를 받아 직접 확인했고,
우리 추론은 그렇게 표시했다. 새 실험 결과가 아니라 <b>문헌의 정밀 독해 + 우리 정식화</b>다.</p>

<p>표기: \(H:=\frac{1}{1-\gamma}\) (1-step 유효지평), \(\bar H:=\frac{1}{1-\gamma^h}\) (h-step 유효지평),
\(a_{t:t+h}:=(a_t,\dots,a_{t+h-1})\), \(R_{t:t+h}:=\sum_{t'=t}^{t+h-1}\gamma^{t'-t}r(s_{t'},a_{t'})\).
보상은 \([0,1]\), 따라서 가치는 \([0,H]\).</p>

<h3>Part 0. QC — 이득만 증명되어 있다</h3>
<p>QC는 MDP를 바꾸지 않고 <b>서명</b>만 바꾼다: \(\pi(a_{t:t+h}\mid s_t)\), \(Q(s_t,a_{t:t+h})\).
백업은 중간 부트스트랩 없이 h-step 한 번이다.</p>
\[ Q(s_t,a_{t:t+h})\ \leftarrow\ \sum_{j=0}^{h-1}\gamma^j r_{t+j}\ +\ \gamma^h Q(s_{t+h},a_{t+h:t+2h}) \]
<p>QC의 유일한 정리(Proposition A.1)는 이 백업이 <b>비편향</b>이라는 것이다. 증명은 tower property 세 줄인데,
기제가 중요하다 — <b>추정 대상을 chunk-조건부 \(Q\)로 재정의했기 때문에 off-policy 편향이 정의상 사라진다</b>
(교정한 게 아니라 부재한다). 그 대가로 h배 빠른 value 전파를 공짜로 얻는다. 그런데 <b>비용에 대한 정리는
QC에 없다</b>. \(h=50\)에서 성공률이 0으로 무너지는 것을 관측하고 §5.4에 이렇게 적었을 뿐이다:</p>
<blockquote>"We suspect that overly large chunk sizes either hurt policy reactivity too much or make policy
learning too difficult, as the network must predict a much longer action sequence at once." (QC §5.4)</blockquote>
<p>두 원인(반응성 상실 vs 정책 학습 난이도)이 분리되지 않은 채 남았다. <b>그 분리가 DQC의 출발점</b>이다.</p>

<h3>Part I. DQC — chunking의 비용을 재는 법</h3>

<h4>I.1 뿌리: nominal ≠ actual</h4>
<p>DQC가 도입한 단 하나의 구분이 논문 전체를 굴린다.</p>
<div class='tblwrap'><table>
<tr><th>기호</th><th>이름</th><th>뜻</th></tr>
<tr><td>\(\hat V_{ac}\)</td><td><b>nominal</b></td><td>데이터로 chunked TD를 돌려 수렴한 값 — <b>우리가 학습하는 것</b></td></tr>
<tr><td>\(V_{ac}\)</td><td><b>actual</b></td><td>그 chunk 정책을 환경에서 <b>실제 open-loop로 굴렸을 때</b>의 값</td></tr>
</table></div>
<p><b>Definition 1 (open-loop trajectory).</b> 데이터의 chunk 주변분포를 그대로 쓰는 정책
\(\pi^\circ_D(a_{t:t+h}\mid s_t):=P_D(a_{t:t+h}\mid s_t)\)을 open-loop로 굴린 분포</p>
\[ P^\circ_D(s_{t+1:t+h+1},a_{t:t+h}\mid s_t)\ :=\ \pi^\circ_D(a_{t:t+h}\mid s_t)\prod_{k=0}^{h-1}T(s_{t+k+1}\mid s_{t+k},a_{t+k}) \]
<p>는 일반적으로 데이터 분포 \(P_D\)와 <b>다르다</b>. 이유가 핵심이다 — <b>데이터를 만든 정책이 closed-loop</b>이기
때문이다. 사람이든 스크립트든 \(a_{t+1}\)을 \(s_{t+1}\)을 <b>보고 나서</b> 골랐다. 그래서 chunk 전체를 조건으로
걸면 <b>미래의 확률적 결과가 조건부로 새어 들어온다</b>. Assumption 1은 데이터가 진짜 dynamics를 따른다는 것만
요구하므로(행동 정책은 비-Markov 허용), 이 분석은 <b>모델 오차가 아니라 순수한 open-loop 재생 편향</b>에 관한
것이다.</p>

<h4>I.2 Definition 2 — Open-Loop Consistency</h4>
<p><b>weak</b> \(\varepsilon_h\)-OLC: 모든 \(s_t\in\mathrm{supp}(P_D)\)에 대해</p>
\[ D_{TV}\!\big(P^\circ_D(s_{t+h'},a_{t+h'}\mid s_t)\,\big\|\,P_D(s_{t+h'},a_{t+h'}\mid s_t)\big)\le\varepsilon_h,\quad h'=1..h-1 \]
\[ D_{TV}\!\big(P^\circ_D(s_{t+h}\mid s_t)\,\big\|\,P_D(s_{t+h}\mid s_t)\big)\le\varepsilon_h \]
<p><b>strong</b> \(\varepsilon_h\)-OLC: 여기에 더해 support 안의 <b>개별 chunk 하나하나</b>에 대해 균일하게</p>
\[ D_{TV}\!\big(T(s_{t+h'}\mid s_t,a_{t:t+h'})\,\big\|\,P_D(s_{t+h'}\mid s_t,a_{t:t+h})\big)\le\varepsilon_h,\quad h'=1..h \]
<p>weak는 "재생 <b>주변분포</b>가 맞다", strong은 "<b>어떤 chunk를 집어도</b> 데이터의 조건부 다음-상태 법칙이
open-loop 롤아웃과 일치한다". 후자는 사실상 <b>chunk가 중간 상태와 (거의) 독립</b>이라는 뜻이고, 그래서
교란(confounding)을 제거한다. 이 구분이 I.4와 I.6에서 결정적으로 갈린다.</p>

<h4>I.3 Lemma 2 — 모든 바운드의 일꾼</h4>
<p>\(P,Q\in\Delta_X\), \(f,g:X\to[0,1]\), \(D_{TV}(P,Q)\le\varepsilon\), 그리고 \(\mathrm{supp}(P)\cap\mathrm{supp}(Q)\)
위에서 \(\|f-g\|_\infty\le\delta\)이면</p>
\[ \big|\,\mathbb E_{P}[f]-\mathbb E_{Q}[g]\,\big|\ \le\ (1-\varepsilon)\delta+\varepsilon \]
<p>증명은 질량 분해 \(P=d_P+d_{PQ},\ Q=d_{PQ}+d_Q\) (\(\int d_P=\int d_Q=\hat\varepsilon\le\varepsilon\))로,
겹치지 않는 부분은 값 범위 전체(\(\le\hat\varepsilon\))로, 겹치는 부분은 \(\|f-g\|_\infty(1-\hat\varepsilon)\)로
값매김한다. <b>가치함수에 적용할 땐 범위가 \([0,H]\)이므로 \(\varepsilon\) 항이 \(\varepsilon/(1-\gamma)\)가
된다</b> — 아래 모든 \(H\) 인자의 출처가 바로 이것이다.</p>

<h4>I.4 Theorem 1 (AC Value Bias) — 왜 하필 \(\varepsilon_h H\bar H\)인가</h4>
<p>\(\hat V_{ac}\)가 행동 chunk 백업 \(\hat V_{ac}(s_t)=\mathbb E_{P_D}[R_{t:t+h}+\gamma^h\hat V_{ac}(s_{t+h})]\)의
해이고 \(V_{ac}\)가 \(\tilde\pi_{ac}:s_t\mapsto P_D(a_{t:t+h}\mid s_t)\)의 실제 값일 때, \(D\)가 <b>weak</b>
\(\varepsilon_h\)-OLC이면</p>
\[ \big|V_{ac}(s_t)-\hat V_{ac}(s_t)\big|\ \le\ \frac{\gamma\varepsilon_h}{(1-\gamma)\big(1-(1-\varepsilon_h)\gamma^h\big)}\ \le\ \varepsilon_h H\bar H \]
<p><b>증명 기계.</b> 한 번의 백업에서 오차원이 둘이다.</p>
<ol>
<li><b>보상 항</b>: 각 중간 시점 \(h'\)마다 Lemma 2를 \(f=g=r\in[0,1]\)로 적용(\(\delta=0\)) →
\(\gamma^{h'}\varepsilon_h\). 합쳐 \(\sum_{h'=1}^{h-1}\gamma^{h'}\varepsilon_h\).</li>
<li><b>부트스트랩 항</b>: Lemma 2를 \(f=\hat V_{ac},g=V_{ac}\in[0,H]\)로 적용 →
\(\gamma^h\big[\varepsilon_h\cdot\frac{1}{1-\gamma}+(1-\varepsilon_h)\sup|\hat V_{ac}-V_{ac}|\big]\).</li>
</ol>
<p>여기가 심장이다: <b>어긋난 \(\varepsilon_h\)만큼의 질량은 최대 오차 \(1/(1-\gamma)\)로 즉시 튀고, 나머지
\((1-\varepsilon_h)\)만 재귀된다.</b> 따라서 수축계수가 \(\gamma^h\)가 아니라
\(\boldsymbol{(1-\varepsilon_h)\gamma^h}\)이다. \(\Delta:=\sup|\hat V_{ac}-V_{ac}|\)로 두면</p>
\[ \Delta\ \le\ \sum_{h'=1}^{h-1}\gamma^{h'}\varepsilon_h+\frac{\gamma^h\varepsilon_h}{1-\gamma}+\gamma^h(1-\varepsilon_h)\Delta
\ \Longrightarrow\ \Delta\le\frac{\gamma\varepsilon_h}{(1-\gamma)(1-(1-\varepsilon_h)\gamma^h)} \]
<p>재귀가 정당한 이유는 Assumption 1에 의해 \(s_{t+h}\mid s_t\)의 support가 \(s_t\)의 support에 포함되기
때문이다. 요약하면 <b>\(\bar H\)는 chunk 단위 백업 횟수에서, \(H\)는 오차가 튈 때의 최대 크기에서</b> 나온다.</p>

<h4>I.5 Theorem 2 — 상한이 tight하다 (2h-state 반례)</h4>
<p>임의의 \(h>1,\gamma,\varepsilon_h\in[0,1/2]\)에 대해 상한을 <b>정확히 달성</b>하는 MDP가 존재하고,
<b>과대·과소 양방향</b> 모두 가능하다. 구성(Figure 8): \(\varepsilon_h=2\delta(1-\delta)\)가 되도록 \(\delta\)를
잡고, 상태 \(\{X_0,X_1,\tilde X_1,\dots,Z\}\)에서 <b>행동과 무관하게</b> 확률 \(\delta\)로 tilde 분기,
\(1-\delta\)로 비-tilde 분기로 간다. 보상은 \(r(\tilde X_i,a{=}0)=r(X_i,a{=}1)=1\), 반대는 0 — 즉
<b>올바른 행동이 상태에 의존</b>한다. 데이터는 최적 closed-loop 정책이 만들었으므로 데이터에서 행동과 상태가
완벽히 상관돼 있다:</p>
\[ P_D=\begin{pmatrix}\delta&0\\0&1-\delta\end{pmatrix},\qquad
P^\circ_D=\begin{pmatrix}\delta^2&(1-\delta)\delta\\ \delta(1-\delta)&(1-\delta)^2\end{pmatrix} \]
<p>open-loop 재생은 그 상관을 <b>깨뜨린다</b>(chunk를 주변분포에서 뽑고 환경이 동전을 다시 던지므로). 그래서
매 시점 \(D_{TV}=2\delta(1-\delta)=\varepsilon_h\)이고, \(h\)시점에 정확히 \(\varepsilon_h\)의 질량이 흡수상태
\(Z\)로 샌다. 결과 \(\hat V_{ac}(X_0)=1/(1-\gamma)\)인데 실제 값은 낮아 차이가 정확히 상한과 같다.
<b>이 "chunk는 동전 결과에 반응할 수 없다" 장치가 이후 모든 반례의 골격이다.</b></p>

<h4>I.6 Corollary 1 — bias 바운드가 곧 suboptimality 바운드가 되는 트릭</h4>
<p>데이터 \(D^\star\)를 <b>최적 정책</b>이 만들었다고 하자. 그러면</p>
\[ V^\star(s_t)=\mathbb E_{P_{D^\star}}\big[R_{t:t+h}+\gamma^hV^\star(s_{t+h})\big] \]
<p>인데 이는 정확히 \(\hat V_{ac}\)의 고정점 방정식이다. 따라서 <b>\(\hat V_{ac}=V^\star\)</b>이고, Theorem 1이
<b>새 증명 없이</b> 최적성 gap 바운드로 바뀐다:</p>
\[ V^\star(s_t)-V^\star_{ac}(s_t)\ \le\ V^\star(s_t)-\tilde V_{ac}(s_t)\ \le\ \frac{\gamma\varepsilon_h}{(1-\gamma)(1-(1-\varepsilon_h)\gamma^h)}\ \le\ \varepsilon_h H\bar H \]
<p>여기서 \(V^\star_{ac}\)는 <b>최적 chunk 정책의 실제 값</b>이다. Corollary 2가 \((3h-1)\)-state MDP로
tight성까지 증명한다(그 구성에서는 <b>클론된 정책이 아니라 최적 chunk 정책 자체</b>가 막힌다 — 어떤 chunk도
\(A/B\) 분기에 반응할 수 없기 때문). <b>이것이 이 문헌에서 "open-loop commitment의 대가"를 정량화한 유일한
식이다.</b></p>

<h4>I.7 그 \(\varepsilon_h\)의 정체 — Proposition 4</h4>
<p>\(T\)가 \(\varepsilon\)-deterministic, 즉 \(T(s'\mid s,a)=(1-\varepsilon)\delta_{f(s,a)}(s')+\varepsilon\tilde T(s'\mid s,a)\)
이면 그 MDP에서 나온 <b>어떤</b> 데이터든 weak \(\varepsilon_h\)-OLC이며</p>
\[ \varepsilon_h\ =\ 3\big(1-(1-\varepsilon)^{h-1}\big) \]
<blockquote>"This bounded stochasticity allows the results of taking an action sequence (of length h) open-loop
to be deterministically determined in the event that the deterministic dynamics is 'triggered' (with a joint
\((1-\varepsilon)^{h-1}\) probability across h time steps). It is clear that under such event, there is no gap
between the 'replayed' open-loop data \(P^\circ_D\) and the original data distribution \(P_D\)." (DQC §E.1)</blockquote>
<p>증명은 지시자 \(I=1\{\text{h-1스텝 모두 결정론 분기}\}\)를 도입하고 <b>Lemma 4 (data processing
inequality)</b>로 결정론 사상 \(a_{t:t+h}\mapsto f(s_t,a_{t:t+h})\)를 통과시켜 TV를 세 갈래로 잇는다 — 상수 3은
그 <b>세 갈래의 합</b>이다.</p>
<div class='callout'><span class='k'>따라서 (우리 조합; 원문에 한 줄로는 없음)</span>
\(V^\star_1-V^\star_H\lesssim 3(H{-}1)\varepsilon\cdot H\bar H\) (작은 \(\varepsilon\)), 즉
<b>결정론적 dynamics에서는 open-loop commitment의 대가가 정확히 0이다.</b> 이 사실이 Part III 분해의 축이 된다.</div>

<h4>I.8 Proposition 1 — weak OLC로는 Q-learning이 \(\Omega(H)\)만큼 망가진다</h4>
<p>어떤 MDP와 weak \(\varepsilon_h\)-OLC 데이터에서
\(V^\star(s_t)-V^+_{ac}(s_t)=\gamma c/(1-\gamma)=\Omega(H)\)이다 (\(c\)는 \(1/2\)까지 가능 — 즉 값 범위의
상수배). 6-state 반례의 기제는 이렇다: 행동 정책이 <b>closed-loop</b>이라 \(\pi_D(B)=0,\pi_D(C)=1\), 즉
<b>두 번째 액션이 첫 전이의 결과를 드러낸다</b>. 그래서 데이터에서 chunk \((0,0)\)을 조건으로 걸면 "\(s_1=B\)였다"는
뜻이 되어</p>
\[ P_D(s_2=D\mid A,(0,0))=1\quad\text{(보상 1)},\qquad\text{그러나 open-loop 실행 시 }P(D)=\delta \]
<p>nominal Q는 \(\hat Q^+_{ac}(A,(0,0))=\gamma/(1-\gamma)\)로 최대라 \(\pi^+_{ac}\)가 그 chunk를 고르지만,
실제 실행하면 \(1-\delta\) 확률로 \(C\)에 떨어지고 거기서 \(a=0\)은 흡수상태 \(Z\)로 간다.</p>
<blockquote>"the chunked critic \(Q(s_t,a_{t:t+h})\) has no way of differentiating a low-probability, 'lucky'
success from a closed-loop, high-probability success. This can cause the learned policy \(\pi^+_{ac}\) to
erroneously prefer very low-value action chunks even when the optimal action chunks are available in the data
distribution." (DQC §4.4)</blockquote>
<div class='callout warn'><span class='k'>우리 데이터에 직결</span>
yam·RoboCasa의 teleop 데이터는 <b>사람이 보고 반응한 완전한 closed-loop</b>이다. 즉 Prop 1의 병리가 우리
데이터에 <b>구조적으로 존재한다</b>. 이것을 우리는 <b>hindsight leakage</b>라 부르고 Part III에서 \(k\)-의존
편향으로 정량화한다.</div>

<h4>I.9 Theorem 3 — strong OLC면 회복된다, 그리고 \(2+1\)의 출처</h4>
<p>\(D,D^\star\) 모두 strong \(\varepsilon_h\)-OLC이고 support 포함이면</p>
\[ V^\star(s_t)-V^+_{ac}(s_t)\ \le\ \frac{\varepsilon_h\gamma}{1-\gamma}\left[\frac{2}{1-(1-2\varepsilon_h)\gamma^h}+\frac{1}{1-(1-\varepsilon_h)\gamma^h}\right]\ \le\ 3\varepsilon_hH\bar H \]
<p><b>인자 2의 출처</b>: strong OLC가 \(D\to T\)와 \(T\to D^\star\) 각각에 \(\varepsilon_h\)를 주므로 TV의 삼각
성질로 \(D_{TV}(P_D\|P_{D^\star})\le 2\varepsilon_h\)가 된다 — "데이터→진짜 dynamics"로 한 번, "진짜
dynamics→최적 데이터"로 또 한 번 지불한다. <b>결정적 상쇄</b>는 재귀의 마지막 단계다:
\(\mathbb E_{P_{D^\star}}[\hat Q^+_{ac}]-\sup_a\hat Q^+_{ac}\le 0\)인데, 이는 \(\pi^+_{ac}\)가
<b>더 큰 support 위에서</b> 최대화하기 때문이다(support 포함 가정이 여기서 쓰인다). 그리고 마지막 \(+1\) 항은
Theorem 1을 \(\pi^+_{ac}\)에 적용한 <b>nominal↔actual 재생 편향</b>이다. 중요한 점: <b>이 바운드는 데이터가
얼마나 suboptimal한지와 무관하다.</b></p>
<p>Theorem 4는 이 \(3\)이 필요함을 tight하게 보이며, 세 조각의 의미를 원문이 직접 밝힌다:</p>
<blockquote>"(1) the optimal action chunking policy is \((\varepsilon_hH^2)\)-sub-optimal due to its inability
to react to environment stochasticity … (2) the value <b>under-estimation</b> bias can incur another factor
… (3) the action chunking value function may prefer an <b>overestimated</b> action chunking policy
\(\pi^+_{ac}\) where its actual value is again \(\varepsilon_hH\bar H\) from its estimated value, resulting in
a total sub-optimality of \(3\varepsilon_hH\bar H\)." (DQC §4.4, Theorem 4 논의)</blockquote>

<h4>I.10 Lemma 8 — "길게 commit할수록 (약하게) 나쁘다"</h4>
<p>임의의 <b>open-loop</b> 데이터 분포 \(D^\circ\)에 대해</p>
\[ V^\star(s_t)\ \ge\ \mathbb E_{T}\big[r_t+\gamma V^\star(s_{t+1})\big]\ \ge\ \mathbb E_{P_{D^\circ}(\cdot\mid s_t,a_{t:t+h})}\big[R_{t:t+h}+\gamma^hV^\star(s_{t+h})\big] \]
<p>증명은 귀납이다 — open-loop로 한 스텝 더 commit할 때마다 \(\max_{a'}Q^\star\)가 <b>고정된 행동</b>으로
대체되므로 값이 약하게 감소한다. 이것이 DQC의 <b>commitment 길이에 대한 단조성 정리</b>이며,
Prop 3·Thm 7의 "좋은 chunk의 첫 액션은 그렇게 나쁠 수 없다" 논증의 일반형이다.</p>

<h4>I.11 Proposition 3 vs Theorem 5/6 — closed-loop도 공짜가 아니다</h4>
<p>chunk 정책의 첫 액션만 실행하고 매 스텝 재질의하는 \(\pi^\bullet\)에 대해, strong OLC 아래</p>
\[ V^\star(s_t)-V^\bullet(s_t)\ \le\ \frac{\varepsilon_h\gamma}{(1-\gamma)^2}\Big[\cdots\Big]\ \le\ 3\varepsilon_hH^2\bar H \]
<p>즉 open-loop 실행(\(3\varepsilon_hH\bar H\))보다 <b>\(H\)배를 더 문다</b>. 증명 1단계가 Lemma 8의 특수형이고,
3단계의 표준 performance-difference 재귀가 \(1/(1-\gamma)\)를 한 번 더 붙인다. 그러나 이것을 "짧게 끊으면
손해"로 읽으면 틀린다. DQC 자신이 바로 "Can we do better than this?"를 묻고 <b>Definition 4 (bounded
optimality variability, BOV)</b>를 도입해 <b>Theorem 5</b>로</p>
\[ V^\star-V^\bullet\ \le\ \vartheta^L_hH+2\vartheta^G_hH\bar H \]
<p>를 얻는다 (OLC 가정 없이!). 증명의 묘미는 <b>global BOV가 과소추정을, local BOV가 과대추정을</b> 각각
막고, 둘의 \(\min\)을 취해 \(\vartheta^L\) 항에서 \(\bar H\)를 <b>깎아낸다</b>는 점이다. 그리고
<b>Theorem 6</b>은 반대 방향도 보인다 — closed-loop 실행은 거의 최적인데 <b>같은 정책의 chunk 실행은
\(\Omega(H)\) suboptimal</b>인 MDP("성"과 "꽃" 두 gadget의 합성)가 존재한다.</p>
<div class='callout'><span class='k'>정직한 결론</span>
<b>open-loop과 closed-loop 어느 쪽도 일반적으로 우월하지 않다 — MDP·데이터 구조(OLC/BOV)에 달렸다.</b>
Table 1이 이를 요약한다: weak OLC는 <i>가치추정</i>만 보장하고 정책은 못 지키며(Prop 1), BOV는 <i>closed-loop</i>만
보장하고 open-loop chunk 정책은 못 지킨다(Thm 6). <b>바로 이 사실이 상태별 적응 \(k\)를 원리적으로
정당화한다</b>: 고정 \(k\)로는 어느 구조에서든 한쪽을 반드시 잃는다.</div>

<h4>I.12 Lemma 7 / Theorem 7 — 낙관의 정체는 "확률적 지름길"</h4>
<p><b>Definition 7</b>: \(M\)이 \(\vartheta_h\)-stochastic shortcut이 없다는 것은, 양의 확률을 가진 모든 경로에서
\(\gamma^hV^\star(s_{t+h})+R_{t:t+h}-V^\star(s_t)\le\vartheta_h\)라는 뜻이다.</p>
<blockquote>"stochastic shortcuts are low-probability (but plausible) paths … that lead to returns that are much
higher than the optimal expected value. These … are particularly problematic for action chunking value backup
because the chunked critic cannot distinguish between a low-probability stochastic shortcut and an optimal
closed-loop trajectory, leading it to erroneously favor the shortcut." (DQC §E.3)</blockquote>
<p><b>Lemma 7</b>: 그런 지름길이 없으면 과대추정이
\(\hat V^+_{ac}(s_t)-V^\star(s_t)\le\vartheta_h/(1-\gamma^h)\)로 묶인다. 즉 <b>낙관의 크기 = critic이 제어와
구별하지 못하는 확률적 지름길의 가치</b>다. Theorem 7은 OLC도 BOV도 없이, 데이터가
\(\alpha\)-open-loop-mixed이기만 하면 closed-loop 실행이
\(\frac{\alpha}{(1-\gamma)^2(1-\gamma^h(1-\alpha))}+\frac{\vartheta_h\gamma^h}{(1-\gamma)(1-\gamma^h)}\)로
근최적임을 보인다. 증명의 기술적 심장은 <b>Lemma 1 (조건부 확률의 평균값 정리)</b> — <i>평균적</i> 혼합
바운드를 <i>어떤 특정 chunk에서의</i> 점별 바운드로 바꿔준다.</p>

<h4>I.13 Proposition 2 — chunking을 언제 n-step보다 선호하나</h4>
\[ V^+_{ac}(s_t)-\hat V^+_n(s_t)\ \ge\ \delta_n\bar H_n-3\varepsilon_hH\bar H \]
<p>즉 데이터가 \(3\varepsilon_hH\)보다 더 suboptimal하면(\(\delta_n\)이 크면) chunking이 이긴다. 그리고
Proposition 5는 그 역도 보인다 — <b>데이터의 suboptimality(\(\delta_n\))와 open-loop consistency(\(\varepsilon_h\))는
서로 독립</b>이다.</p>

<h4>I.14 DQC 알고리즘</h4>
<p>critic chunk \(h\)와 <b>정책 chunk \(h_a\ll h\)</b>를 분리한다. 이상적 목적은 "앞 \(h_a\)는 정책이, 뒤는
최적으로 채운다"인데 다루기 어려우므로 <b>부분 critic</b>을 <b>낙관적(expectile) distillation</b>으로 만든다:</p>
\[ \mathcal L(\psi)=f^{\kappa_d}_{\text{expectile}}\!\big(\bar Q_\phi(s_t,a_{t:t+h})-Q^P_\psi(s_t,a_{t:t+h_a})\big),\qquad
\mathcal L(\pi)=-\mathbb E\big[Q^P_\psi(s_t,a_{t:t+h_a})\big] \]
<p>즉 \(Q^P\approx\max_{\text{tail}}Q\)이며 <b>구조적으로 낙관적</b>이다. \(V_\xi\)는 \(\kappa_b=(N-1)/N\)
quantile로 맞춰 best-of-\(N\)의 최대 순서통계량과 일치시킨다. 대가: 배포가 <b>짧은 chunk</b>가 되어 I.11의
\(H\)배 항(최악의 경우)과 QC의 시간적 일관 탐색 이득을 지불한다. 그리고 DQC 스스로 §8에서 남긴 숙제가 있다:</p>
<blockquote>"our method relies on a fixed policy action chunk size \(h_a\) and critic action chunk size \(h\)
across all states, even though the optimal action chunk size may vary by state. Developing practical methods
that can support flexible, state-dependent chunk sizes would be a natural next step." (DQC §8)</blockquote>
<p>AQC가 바로 이 숙제에 답한 논문이다.</p>

<h4>I.15 \(h\)에 대한 단조성 — 흩어진 조각을 모으면</h4>
<p>모든 바운드가 \(\bar H=1/(1-\gamma^h)\)를 갖는데 이는 \(h\)에 대해 <b>감소</b>한다(= chunking이 유효지평을
줄인다). 그러나 \(\varepsilon_h\approx 3(h-1)\varepsilon\)는 <b>증가</b>한다. 곱의 거동은 자명하지 않아
<b>우리가 직접 수치로 확인</b>했다 (γ·ε 조합별, 정확한 형태
\(\gamma\varepsilon_h/[(1-\gamma)(1-(1-\varepsilon_h)\gamma^h)]\)):</p>
<div class='tblwrap'><table>
<tr><th>\(\gamma\)</th><th>\(\varepsilon\)</th><th>k=2</th><th>k=5</th><th>k=10</th><th>k=30</th><th>k=50</th><th>단조↑?</th></tr>
<tr><td>0.99</td><td>0.001</td><td>13.0</td><td>19.6</td><td>22.2</td><td>26.2</td><td>29.5</td><td>예</td></tr>
<tr><td>0.999</td><td>0.001</td><td>600.2</td><td>707.8</td><td>734.4</td><td>759.6</td><td>773.7</td><td>예</td></tr>
<tr><td>0.99964</td><td>0.001</td><td>2240.7</td><td>2418.1</td><td>2457.2</td><td>2491.5</td><td>2509.6</td><td>예</td></tr>
<tr><td>0.99964</td><td>0.010</td><td>2713.6</td><td>2740.0</td><td>2748.6</td><td>2767.3</td><td>2776.8</td><td>예</td></tr>
</table></div>
<p><b>결론(우리 계산)</b>: 편향 상한은 \(k\)에 대해 <b>단조 증가</b>한다. 다만 <b>매우 빨리 포화</b>한다
(\(\gamma=0.99964\)에서 \(k=2\)에 이미 2241, \(k=50\)에 2510, 값 범위 상한은 2778). 즉 이 바운드는
<b>방향은 확립하지만 큰 \(k\)에서는 정량적으로 공허</b>하다 — 그래서 실제 편향은 실측해야 한다(Part III 사전등록).</p>
"""

KO += r"""
<h3>Part II. AQC — AOLC와 selector, 그리고 우리가 검증한 것</h3>

<h4>II.1 Definition H.2 — Adaptive Open-Loop Consistency</h4>
<p>DQC의 OLC는 <b>고정 길이 \(h\)</b> 재생을 전제한다. AQC는 선택함수 \(\kappa:\mathcal S\to\mathcal K\)를 넣어
<b>상태 의존 재질의</b>로 확장한다:</p>
\[ D_{TV}\!\big(P_D(s_{t+\kappa(s_t)},a_{t+\kappa(s_t)}\mid s_t)\ \big\|\ P^\circ_{D,\kappa}(\cdot\mid s_t)\big)\le\varepsilon_{\mathcal K} \]
<p>(상태 주변분포에 대해서도 동일.) 취지는 <b>재질의 지점이 상태마다 달라 무작위 간격이 되므로 TV 바운드가
\(\kappa\)가 만드는 재질의 시각 분포 전체에 균일해야 한다</b>는 것이다. <b>Proposition H.3</b>: \(\kappa\equiv k\)
(상수)이면 DQC의 weak OLC로 정확히 환원된다.</p>
<div class='callout warn'><span class='k'>정직한 평가</span>
정의만 보면 이는 DQC의 두 TV 조건에서 고정 오프셋 \(t+h\)를 \(t+\kappa(s_t)\)로 <b>치환</b>한 것이다.
새 기계(재질의 지점 <i>수열</i>에 대한 결합 조건, 정지시각 측도, 다중 결정시점의 합성)는 도입되지 않는다.
그래서 두 가지가 미해결로 남는다 — ① 정의는 \(s_t\)에서의 <b>한 번의</b> 적응 스텝만 제약하는데 실제 실행은
이를 여러 번 합성한다(스텝별 TV는 자동으로 \(T\)-스텝으로 합성되지 않는다); ② \(\kappa\)가 정의에서는
<b>주어진 고정 함수</b>인데 실제 selector는 critic에 의존해 학습 중 계속 변한다. Theorem H.14는 실제로
<b>oracle \(k^\dagger\) 하의 AOLC</b>를 가정하는데, 이는 실행되는 스케줄(\(\hat k\))이 아니다.</div>

<h4>II.2 selector 기준 — 왜 \(-V^k\)이고 왜 \(/\gamma^k\)인가</h4>
<p>순진한 규칙 \(\arg\max_{k,a}Q^k\)은 두 가지로 무너진다(§4.2). ① <b>discount-scale mismatch</b>:
sparse reward에서 \(Q^k\approx\gamma^kV^h(s_{t+k})\)이므로</p>
<blockquote>"Since \(\gamma<1\), the factor \(\gamma^k\) is strictly decreasing in \(k\). Consequently,
\(Q^{k_1}>Q^{k_2}>\cdots>Q^h\) for nearly every state, regardless of which chunk size actually yields a better
policy. The selector degenerates to always choosing the smallest \(k\) in \(\mathcal K\)." (AQC §4.2)</blockquote>
<p>② <b>state-dependent baseline mismatch</b>: \(/\gamma^k\)만 하면 \(\arg\max_kV^h(s_{t+k})\)가 되는데 보상에서
먼 대다수 상태에서는 \(V^h\)가 모두 작아 <b>\(k\) 간 차이가 함수근사 노이즈에 지배</b>된다. 그래서</p>
\[ \mathrm{score}(k,a_{t:t+k})\ :=\ \frac{Q^k(s_t,a_{t:t+k})-V^k(s_t)}{\gamma^k} \]
<p><b>Proposition 5.1 (noise immunity)</b>: 신호 없는 영역에서 \(|\delta_k|\le\epsilon+2\sigma\)가 되어 모든
\(k\)가 0 근처로 몰린다 — "편향된 오답을 무편향 난수로 바꾼다"는 논지.</p>
<div class='callout warn'><span class='k'>우리가 확인한 부호 문제 (원문 대조)</span>
①의 논증은 \(V^h>0\)을 전제한다. 그런데 <b>AQC 자신의 벤치마크 보상은 음수</b>다 — 원문 §C.1: "receives −1
when the task is incomplete and 0 upon completion". 그러면 \(V^h\le0\)이므로 \(\gamma^kV^h\)는 \(k\)에 대해
<b>증가</b>하고(덜 음수), 순진한 selector는 <b>가장 긴</b> chunk로 붕괴해야 한다 — 논문이 보고한
"raw-Q variant collapses to always selecting \(k=1\)"과 반대 방향이다. 우리 <b>cost_to_goal(\(r=-1\))에서도
같은 이유로 \(/\gamma^k\)가 short를 선호</b>한다. <b>이 정규화는 보상 규약 의존적이며 우리 세팅에 그대로
옮기면 안 된다.</b></div>

<h4>II.3 Definition H.4 / Theorem H.5 — 순환성</h4>
<p>oracle과 selector를 나란히 놓으면:</p>
\[ k^\dagger(s)\in\arg\max_k\max_a\frac{Q^{k,*}(s,a)-V^{k,*}(s)}{\gamma^k},\qquad
\hat k(s)\in\arg\max_k\max_a\frac{Q^{k}(s,a)-V^{k}(s)}{\gamma^k} \]
<p><b>동일한 범함수</b>를 starred 값에서 평가하느냐 추정값에서 평가하느냐의 차이뿐이다. 따라서 Theorem H.5
("selector soundness", 조건 \(\bar\varepsilon<\Delta\gamma^{k_{\min}}/2\))는 <b>"plug-in 추정량이 자기 자신의
모집단 argmax를 복원한다"</b>는 일관성 보조정리이지, <b>그 기준이 return을 최대화한다는 주장이 아니다</b>.
증명 자체는 삼각부등식 세 줄로 올바르다.</p>
<div class='callout warn'><span class='k'>우리가 확인한 두 문제 (원문 대조)</span>
① <b>\(V^{k,*}\)가 어디에도 정의되어 있지 않다</b>(원문에서 8회 사용, 정의 없음). 자연스러운 "최적" 독법
\(V^{k,*}=\max_aQ^{k,*}\)를 택하면 \(\max_aA^{k,*}\equiv0\)이 되어 <b>\(\Delta(s)\equiv0\)</b>, 즉 전역
\(\Delta\)-separability(\(\Delta>0\))가 <b>충족 불가능</b>해지고 H.5·H.8 등이 공허해진다. 반면 §5.1이 학습
대상을 설명할 때 쓰는 "행동 정책의 \(k\)-스텝 기대 return" 독법을 택하면 \(A^{k,*}\)는 양수일 수 있으나
\(*\) 표기가 \(Q^{k,*}\)와 불일치한다.
② <b>"오선택 확률"이 sup-norm을 Markov 부등식에 넣어 얻어졌다</b>. 원문:
"By Markov's inequality applied to the estimation error: \(P(\hat k(s)\ne k^\dagger(s))\le
P(|\hat A-A^{k^\dagger,*}|\ge\Delta(s)/2)\le 2\bar\varepsilon/(\gamma^{k_{\min}}\Delta(s))\)."
그런데 \(\bar\varepsilon\)은 <b>결정론적 최악치</b>(\(\|\cdot\|_\infty\))이지 기댓값이 아니며, 애초에 오차가
결정론적이라 확률공간이 없다. 게다가 \(\bar\varepsilon<\Delta\gamma^{k_{\min}}/2\)인 영역에서는 H.5가 이미
오선택이 <b>0</b>임을 증명하고, 그 밖에서는 이 "확률"이 \(\ge1\)이 되어 공허하다. 이 항에 의존하는 결과가
여섯 개다(H.6, H.8의 오차인자, H.11, H.12, H.14 2부, H.18, H.19).</div>

<h4>II.4 Theorem H.8 — meta-MDP dominance, 그리고 우리가 확인한 부호 오류</h4>
\[ V^{AQC}(s)-V^k(s)\ \ge\ \frac{\gamma^{k_{\min}}\big(1-\tfrac{2\bar\varepsilon}{\gamma^{k_{\min}}\Delta}\big)}{1-\gamma}\,
\mathbb E_{s'\sim d^{AQC}}\big[\bar A^{k^\dagger,*}(s')-\bar A^{k,*}(s')\big] \]
<p>증명의 아이디어는 좋다: <b>행동을 쌍 \((k,a_{t:t+k})\)로, 전이를 "\(k\)스텝 open-loop 실행"으로 두는
meta-MDP</b>를 구성해 performance difference lemma를 쓴다. 그러나 3단계(할인 정규화)에서 원문은 이렇게 쓴다:</p>
<blockquote>"Since \(\gamma^{k^*(s')}\ge\gamma^{k_{\min}}\) for all \(s'\) and \(\gamma^k\le\gamma^{k_{\min}}\),
and noting that \(\bar A^{k,*}(s')\ge0\) (as the max advantage is non-negative at the behavior policy's best
action), we lower-bound: \(\gamma^{k^*(s')}\bar A^{k^\dagger,*}(s')-\gamma^k\bar A^{k,*}(s')\ge
\gamma^{k_{\min}}(\bar A^{k^\dagger,*}(s')-\bar A^{k,*}(s'))\)." (AQC Appendix I.4, Eq. 65 직전)</blockquote>
<div class='callout warn'><span class='k'>우리가 직접 확인한 사실</span>
\(k^*(s')\in\mathcal K\)이고 \(k_{\min}=\min\mathcal K\)이므로 \(k^*(s')\ge k_{\min}\), 따라서 \(\gamma<1\)에서
<b>\(\gamma^{k^*(s')}\le\gamma^{k_{\min}}\)</b>이다. 같은 문장의 두 번째 부등식(\(\gamma^k\le\gamma^{k_{\min}}\))은
옳고 <b>첫 번째는 뒤집혀 있다</b>. 부호조건 \(\bar A\ge0\)은 둘째 항
(\(-\gamma^k\bar A^k\ge-\gamma^{k_{\min}}\bar A^k\))은 실제로 살리지만, 첫째 항은 살리지 못한다. 게다가 그
부호조건은 <b>정리의 가정에 없고 괄호 안 한 줄로만</b> 등장하며, \(V^{k,*}\)의 두 독법 중 "최적" 독법에서는
\(\bar A\equiv0\)이 되어 정리의 우변이 0이 된다. 바로잡으면 하한은
\(\gamma^{k_{\max}}\bar A^{k^\dagger}-\gamma^{k_{\min}}\bar A^{k}\)가 되어, <b>\(k^\dagger>k\)일 때
(즉 적응 정책이 비교군보다 <i>긴</i> chunk를 선호할 때 — 논문의 대표 사례) 음수가 될 수 있다.</b></div>
<p>부수적으로 <b>Theorem H.14</b>는 DQC Prop 3의 적응 버전인데(\(h\to k_{\min}\)), \(k_{\min}\le h\)이므로
\(\bar H_{k_{\min}}\ge\bar H_h\)이다 — 즉 치환은 바운드를 <b>키운다(나쁘게 한다)</b>. 원문이 이를
"improves reactivity"로 서술한 것은 바운드의 방향과 반대다(같은 절의 trade-off 서술과는 일관됨).</p>

<h4>II.5 AQC의 알고리즘 — 그리고 결정적 사실</h4>
<p>손실은 넷이다: EMAQ h-step TD로 \(Q^h\), expectile로 \(V^h\), <b>\(\bar V^h\)에서 부트스트랩하는</b>
per-scale \(Q^k\) (max 없음 = 순수 회귀), expectile로 \(V^k\). 정책은 flow-matching BC. 추론은
\(N\)개 후보를 뽑아 per-scale advantage로 점수 매기고 <b>scale마다 z-score 정규화</b> 후 argmax.</p>
<div class='callout'><span class='k'>결정적 사실 — 정책 개선이 없다</span>
AQC의 정책은 <b>순수 behavior cloning</b>이다. actor loss에 \(Q\)가 들어가는 항이 논문 어디에도 없다.
Algorithm 1은 오프라인·온라인 양쪽에서 "Update \(\pi_\beta\) via flow-matching BC"만 한다. 개선은 오직
(i) \(Q^h\) 타깃 안의 EMAQ best-of-\(N\) max와 (ii) 추론 시 best-of-\(N\) 선택에서 온다. RoboCasa 실험은 더
분명하다 — "attach an AQC critic head while freezing all backbone parameters", 학습 대상 "Critic head only".
즉 <b>frozen BC 정책 위의 critic-only 학습 + best-of-10 선택</b>이며, RL 정책 갱신이 전혀 없다.
따라서 AQC의 달성 가능 return은 <b>BC 정책에서 뽑은 \(N\)개 표본의 최대치</b>로 상한이 걸린다.</div>

<h3>Part III. 우리 정식화 — 비어 있는 칸을 정리로 채우기</h3>

<h4>III.1 설정과 세 가치</h4>
<p>완전관측 MDP \(M=(\mathcal S,\mathcal A,T,r,\gamma)\), chunk 길이 \(H\). chunk 정책
\(\pi:\mathcal S\to\Delta(\mathcal A^H)\)와 <b>commitment 선택자</b> \(\kappa:\mathcal S\to\{1,\dots,H\}\)에 대해,
"\(s\)에서 \(\pi\)를 질의해 앞 \(\kappa(s)\)개를 open-loop 실행하고 재질의"를 반복하는 정책의 값을
\(V^{\pi,\kappa}\)로 쓴다. 세 최적값을 구분한다:</p>
\[ V^\star_1:=\sup_{\text{closed-loop 1-step}}V,\qquad
V^\star_H:=\sup_\pi V^{\pi,H},\qquad
V^\star_{\mathrm{ada}}:=\sup_{\pi,\kappa}V^{\pi,\kappa} \]

<p><b>Lemma A (샌드위치).</b> \(V^\star_H\le V^\star_{\mathrm{ada}}\le V^\star_1\).<br>
<i>증명.</i> 좌: \(\kappa\equiv H\)를 택하면 된다. 우: 임의의 \((\pi,\kappa)\) 실행은 관측 필트레이션에 가측인
특정 closed-loop 정책이므로 그 값은 \(V^\star_1\)을 넘을 수 없다. ∎</p>

<h4>III.2 Theorem 1 (분해) — aleatoric과 epistemic</h4>
<p>주어진(미숙한) chunk 정책 \(\pi\)에 대해</p>
\[ \underbrace{V^\star_1-V^{\pi,H}}_{\text{총 손실}}
=\underbrace{\big(V^\star_1-V^\star_H\big)}_{\Delta_{\mathrm{alea}}\ \text{(정책 무관)}}
+\underbrace{\big(V^\star_H-V^{\pi,H}\big)}_{\Delta_{\mathrm{epis}}(\pi)\ \text{(정책 의존)}} \]
<p>\(\Delta_{\mathrm{alea}}\)는 <b>DQC Corollary 1이 재는 바로 그 양</b>이며 \(\le\varepsilon_HH\bar H\),
\(\varepsilon_H=3(1-(1-\varepsilon)^{H-1})\) (Prop 4). \(\Delta_{\mathrm{epis}}(\pi)\)는 정의상
\(\sup_\pi V^{\pi,H}=V^\star_H\)이므로 <b>정책 개선으로 0에 수렴시킬 수 있다</b>.</p>

<h4>III.3 Theorem 2 (결정론 하에서 반응성의 가치는 0) — 우리 핵심 정리</h4>
<p><b>정리.</b> 완전관측 MDP에서 \(T\)와 \(r\)이 결정론적이면
\(V^\star_1=V^\star_H=V^\star_{\mathrm{ada}}\).</p>
<p><i>증명.</i> 결정론이므로 임의의 \(s\)에서 최적 정책은 단일 궤적을 만든다. 그 궤적의 처음 \(H\)개 액션을
내놓는 open-loop chunk 정책 \(\tilde\pi(s)\)를 정의하자(\(\tilde\pi:\mathcal S\to\mathcal A^H\)이므로 클래스
안에 있다). \(\tilde\pi\)를 <b>full commitment</b>로 실행하면 전이가 결정론이라 정확히 같은 \(s_{t+H}\)에
도달하고, 거기서 다시 질의하면 Markov성과 최적성의 상태 의존성에 의해 원 궤적의 연속과 일치한다. 귀납하면
\(\tilde\pi\)의 full-commitment 실행이 최적 궤적을 <b>그대로 재현</b>하므로
\(V^{\tilde\pi,H}=V^\star_1\). 따라서 \(V^\star_H\ge V^\star_1\)이고, Lemma A와 합치면 세 값이 같다. ∎</p>
<p><b>따름정리 (흡수 가능성 = recompose).</b> 결정론 하에서 임의의 \(\pi,\kappa\)에 대해</p>
\[ V^{\pi,\kappa}-V^{\pi,H}\ \le\ V^\star_{\mathrm{ada}}-V^{\pi,H}\ =\ V^\star_H-V^{\pi,H}\ =\ \Delta_{\mathrm{epis}}(\pi) \]
<p>즉 <b>적응 실행이 얻는 이득은 전부 "더 나은 full chunk"로 달성 가능</b>하다 — 정책 개선이 그것을 흡수한다.</p>
<div class='callout'><span class='k'>이 정리가 하는 일</span>
① DQC Cor 1 + Prop 4의 "\(\varepsilon=0\Rightarrow\) gap 0"을 <b>초등적으로 재유도</b>하고,
② DQC가 다루지 않은 <b>adaptive 클래스</b>까지 확장하며,
③ 우리 스토리("짧게 끊어 발견한 이득을 full chunk로 컴파일한다")를 <b>정리로 만든다</b>.</div>

<h4>III.4 반응성 정보집합 — floor의 일반형</h4>
<p>Theorem 2의 증명이 실제로 쓴 것은 "결정론"이 아니라 <b>\(t\)와 \(t+H\) 사이에 \(s_t\)로부터 예측 불가능한
정보가 도착하지 않는다</b>는 사실이다. 이를 그대로 정의로 삼는다.</p>
<p><b>정의 (반응성 정보).</b> \(\mathcal I_t^H\) := 시점 \(t\)의 정보로 \(\sigma(s_t)\)-가측이 아닌, 구간
\((t,t+H]\)에 도착하는 정보. \(\mathcal I_t^H\)가 자명하면 적응 실행의 이득은 0이다(Theorem 2의 논증이 그대로
적용된다 — 실현된 chunk가 \(s_t\)의 결정론적 함수가 되므로).</p>
<p>따라서 <b>floor의 원천은 둘</b>이다: (i) <b>dynamics 확률성</b>(DQC의 \(\varepsilon\)), (ii) <b>부분관측</b>.
VLA는 이미지 관측이므로 (ii)가 실재한다 — 가림(occlusion), 접촉 순간의 미관측 힘. 즉 우리 세팅의 floor는
0이 아니며, <b>그 크기 자체가 skill별로 측정 가능한 새 양</b>이다.</p>

<h4>III.5 Theorem 3 (floor 바운드)</h4>
<p>\(\Delta_{\mathrm{react}}:=V^\star_{\mathrm{ada}}-V^\star_H\)로 두면, Lemma A와 Theorem 1에서</p>
\[ 0\ \le\ \Delta_{\mathrm{react}}\ \le\ \Delta_{\mathrm{alea}}\ \le\ \varepsilon_HH\bar H,\qquad
\varepsilon_H=3\big(1-(1-\varepsilon)^{H-1}\big) \]
<p>이고 결정론에서 \(\Delta_{\mathrm{react}}=0\) (Theorem 2). 그리고 임의의 \(\pi,\kappa\)에 대해</p>
\[ V^{\pi,\kappa}-V^{\pi,H}\ \le\ \Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(\pi) \]
<p>— <b>적응 이득의 분해</b>다. 앞 항은 회수 불가(실행 horizon으로만 지불), 뒤 항은 정책이 흡수한다.</p>

<h4>III.6 왜 개선을 \(k=H\)에서 걸어야 하는가</h4>
<p><b>Lemma B (우리 목적함수는 배포값의 하한이고 종점에서 tight).</b> selector가 참 \(Q^\pi_k\)를 최대화하면
모든 \(\pi\)에 대해 \(V^{\pi,\kappa^*}\ge V^{\pi,H}\). 따라서 \(J_H(\pi):=\mathbb E_s[Q^\pi(s,\mu_\pi(s),H)]\)를
올리는 것은 <b>배포값의 하한을 올리는 것</b>이고, 그 하한은 적응 실행이 더 이상 도움이 되지 않는 지점
(\(V^{\pi,\kappa^*}=V^{\pi,H}\), 즉 recompose 종점)에서 <b>정확히 tight</b>해진다.<br>
<i>증명.</i> \(\kappa\equiv H\)가 후보에 있으므로 참 \(Q\)에 대한 argmax는 그보다 작을 수 없다. ∎</p>
<p><b>대조 (왜 짧은 \(k\)에서 걸면 안 되는가).</b> 개선을 선택된 \(k<H\)에서만 걸면 chunk의 <b>prefix</b>만
좋아지고 \(Q(s,\mu(s),H)\)는 개선되지 않는다. 그러면 \(\Delta_{\mathrm{epis}}\)가 줄지 않아
Theorem 2의 따름정리가 발동하지 않고, <b>chunk-length curriculum도 발생하지 않는다</b>. 이것이 AQC/ExRL/ACSAC가
선택만 해서는 길이가 자라지 않는 이유의 정리적 설명이다.</p>

<h4>III.7 Curriculum — 정리가 예측하는 곡선</h4>
<p>"엄격히 짧은 것이 나은" 상태 집합을
\(\mathcal S_<(\pi):=\{s:\max_{k<H}Q^\pi_k(s)>Q^\pi_H(s)\}\)로 두자. Theorem 2의 따름정리에 의해, 결정론
극한에서 최적 \(\pi\)에 대해 \(\mathcal S_<=\varnothing\) (\(\kappa\equiv H\)가 최적). 일반적으로는
\(\mathcal S_<\)에서 얻는 총 가치가 \(\Delta_{\mathrm{react}}\le\varepsilon_HH\bar H\)로 묶인다(Theorem 3).
따라서 <b>정책 개선이 \(\Delta_{\mathrm{epis}}\)를 흡수할수록 평균 실행 길이는 aleatoric floor까지 단조
증가</b>한다. 중요한 것은 이것이 <b>추가 보상(replan cost) 없이</b> 나온다는 점이다 — return을 건드리지 않는다.
수렴 근처의 무차별 구간만 필요하면 <b>사전식(lexicographic) 규칙</b>(return-최적 \(\pm\epsilon\) 집합 안에서
가장 긴 \(k\))으로 처리하며, 그 유일한 자유 파라미터는 <b>비교 허용오차 \(\epsilon\)</b>이지 비용의 크기가
아니다.</p>

<h4>III.8 Proposition (누출 편향은 \(k\)에 대해 증가하고, baseline이 상쇄하지 못한다)</h4>
<p>teleop 데이터는 closed-loop이므로 Prop 1의 hindsight leakage가 구조적으로 존재한다(I.8). 편향을
\(b^Q_k:=\hat Q^k-Q^k\), \(b^V_k:=\hat V^k-V^k\)로 두면 selector 점수는</p>
\[ \frac{\hat Q^k-\hat V^k}{\gamma^k}=\underbrace{\frac{Q^k-V^k}{\gamma^k}}_{\text{참 advantage}}
+\underbrace{\frac{b^Q_k-b^V_k}{\gamma^k}}_{\text{선택 편향}} \]
<p>(a) I.15에서 우리가 수치로 확인했듯 DQC Thm 1의 편향 상한은 \(k\)에 대해 <b>단조 증가</b>한다.
(b) 누출은 <b>chunk를 조건으로 걸 때</b> 생기는 현상인데 \(V^k(s)\)는 chunk를 조건으로 걸지 않으므로
\(b^V_k\)는 그 성분을 덜 갖는다 ⇒ 차감이 <b>완전히 상쇄하지 못한다</b>. (c) 남은 잔차를 \(1/\gamma^k\)가
<b>\(k\)에 따라 증폭</b>한다.</p>
<div class='callout'><span class='k'>귀결</span>
AQC는 서로 <b>반대 방향</b>의 두 계통 인공물 — 할인 스케일(→short 선호)과 누출 낙관(→long 선호) — 중
<b>하나만 교정</b>하고, 그 교정이 다른 하나를 <b>증폭</b>한다. 게다가 II.2에서 확인했듯 그 교정의 부호는
보상 규약에 의존한다. 이것이 우리가 실측해야 할 첫 번째 대상이다.</div>

<h4>III.9 사전등록 — 무엇이 나오면 우리 주장이 기각되는가</h4>
<div class='tblwrap'><table>
<tr><th>검증</th><th>측정 방법</th><th>통과 기준 / 기각 시 의미</th></tr>
<tr><td><b>k-의존 낙관 편향</b></td><td>per-prefix \(\hat Q^k\)와 할인 MC-return의 차 \(b^Q_k\), 그리고 \(b^V_k\)를 \(k\)별로 실측</td><td>\(b^Q_k-b^V_k\)가 \(k\)에 무관하면 baseline 상쇄가 유효(III.8 기각). 증가하면 selector의 long 선호가 artifact — <b>adaptive 결론 전체를 재검토</b></td></tr>
<tr><td><b>OOD 후보 calibration</b></td><td>실행 궤적뿐 아니라 argmax가 랭킹하는 <b>비선택</b> 후보의 \(\hat Q\) vs 실현 return</td><td>비선택 후보에서 과대평가가 크면 argmax의 정당성 자체가 미검증 (ACSAC의 on-policy calibration으로는 못 잡는 영역)</td></tr>
<tr><td><b>curriculum의 인과성</b></td><td>정책 개선을 끈 arm(선택만)과 켠 arm의 평균 실행 길이 궤적 비교</td><td>끈 arm에서도 길이가 자라면 <b>우리 기여가 아님</b>(AQC/ExRL 재현일 뿐). 켠 arm에서만 자라야 III.7이 지지됨</td></tr>
<tr><td><b>aleatoric floor</b></td><td>수렴 시 skill별 잔여 short-chunk 사용량</td><td>0으로 가면 그 skill은 완전관측·결정론적(우리 Thm 2 극한). 양수로 수렴하면 그 값이 <b>그 skill의 내재적 reactivity 수요</b>라는 새 측정량. floor 아래로 내려가면 과대평가 의심</td></tr>
</table></div>

<h4>III.10 정리</h4>
<p>QC는 chunk critic의 <b>이득</b>을 증명했고(비편향 h-step 백업), DQC는 그 <b>비용</b>을 처음 정량화했으며
(OLC·Thm 1·Cor 1, 그리고 그 비용이 dynamics 확률성임을 Prop 4로 밝혔다), AQC는 그것을 <b>상태 의존
재질의</b>로 일반화했다(AOLC·meta-MDP dominance). 세 논문 모두에서 <b>비어 있는 칸은 하나다 — 적응적 실행이
발견한 이득을 정책이 흡수하게 만드는 것</b>. 우리는 그것을 Theorem 2(결정론 하 \(V^\star_1=V^\star_H=V^\star_{\mathrm{ada}}\))와
그 따름정리로 <b>정리화</b>했고, 개선을 \(k=H\)에 걸어야 한다는 것을 Lemma B로 <b>정당화</b>했으며, 그
부산물인 chunk-length curriculum을 <b>추가 보상 없이</b> 얻는다. 남은 관문은 objective가 아니라 <b>critic의
\(k\)-의존 낙관과 OOD calibration</b>이며, 그 넷을 사전등록했다.</p>
"""
