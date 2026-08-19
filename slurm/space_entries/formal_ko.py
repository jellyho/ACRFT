FORMAL_KO = r"""
<h3>8.5 형식화 — 우리 방식이 정말 원하는 결론에 도달하는가 (정의·정리·증명)</h3>
<p class='sub'>여기까지는 비유였다. 이제 8절의 설계("<b>평가는 prefix별로, 실행은 적응적으로, 개선은 full
chunk에서</b>")를 정확히 정의하고, 그것이 <b>실제로 우리가 원하는 결론에 도달함을 증명</b>한다. 증명은 전부
초등적이다 — 새 기계를 만들지 않고, 개선 루프를 <b>올바른 MDP 안에 닫는 것</b>만으로 표준 정리들이 발동한다는
것이 요점이다.</p>

<h4>정의 1 — full-commitment MDP \(M_H\)</h4>
<p>기저 MDP \(M=(\mathcal S,\mathcal A,T,r,\gamma)\)와 chunk 길이 \(H\)에 대해, <b>행동이 chunk 하나이고 전이가
"그 chunk를 끝까지 open-loop 실행"인</b> MDP를 정의한다:</p>
\[ M_H:=\big(\mathcal S,\ \mathcal A^H,\ T_H,\ r_H,\ \gamma^H\big),\qquad
r_H(s,a):=\mathbb E\Big[\sum_{j=0}^{H-1}\gamma^j r(s_j,a_j)\Big],\quad
T_H(\cdot\mid s,a):=\text{law of }s_H \]
<p>(둘 다 \(s_0=s\)에서 \(a\)를 open-loop로 실행했을 때의 기대·분포.) \(M_H\)는 할인율 \(\gamma^H\)의 평범한
할인 MDP이며, 그 최적값이 바로 Part III에서 정의한 <b>\(V^\star_H\)</b>다.</p>

<h4>정의 2 — decoupled-horizon 방식 (DH)</h4>
<div class='tblwrap'><table>
<tr><th>역할</th><th>정의</th><th>무엇에 의존하나</th></tr>
<tr><td><b>개선용 critic</b></td><td>\(Q^\pi_H(s,a):=r_H(s,a)+\gamma^H\,\mathbb E_{s'\sim T_H}\big[V^{\pi,H}(s')\big]\)</td><td>\(\kappa\)에 <b>의존하지 않음</b></td></tr>
<tr><td><b>개선</b></td><td>\(\pi_{n+1}(s)\in\arg\max_{a\in\mathcal A^H}Q^{\pi_n}_H(s,a)\)</td><td>\(\kappa\)에 <b>의존하지 않음</b></td></tr>
<tr><td><b>배포용 critic</b></td><td>\(Q^{\pi,\kappa}_k(s,a):=\mathbb E\big[\sum_{j<k}\gamma^jr+\gamma^kV^{\pi,\kappa}(s_k)\big]\)</td><td>prefix \(k\)별</td></tr>
<tr><td><b>실행</b></td><td>\(\kappa(s)\in\arg\max_{k\in\mathcal K}Q^{\pi,\kappa}_k(s,\pi(s))\)</td><td>상태 적응</td></tr>
</table></div>
<div class='callout'><span class='k'>여기가 "decoupled"의 정확한 뜻</span>
개선 루프(1·2행)는 <b>\(\kappa\)를 한 번도 참조하지 않는다</b>. 즉 개선은 <b>\(M_H\) 안에서 완결</b>된다.
실행 선택자는 배포 시점에만 등장한다. 이 분리가 아래 정리 전체를 가능하게 한다.</div>

<h4>정리 A — 개선 루프는 epistemic 항을 0으로 몬다</h4>
<p><b>정리.</b> 정의 2의 \(\{\pi_n\}\)에 대해 \(V^{\pi_{n+1},H}\ge V^{\pi_n,H}\) (점별)이고
\(V^{\pi_n,H}\to V^\star_H\)이다. 따라서</p>
\[ \Delta_{\mathrm{epis}}(\pi_n)\ :=\ V^\star_H-V^{\pi_n,H}\ \longrightarrow\ 0 \]
<p><i>증명.</i> 정의 2의 1·2행은 정확히 <b>\(M_H\)에서의 policy iteration</b>이다 (상태공간 \(\mathcal S\),
행동공간 \(\mathcal A^H\), 할인 \(\gamma^H<1\), 유계 보상 \(r_H\)). 표준 policy-iteration 정리에 의해
값은 단조 증가하고 유일한 최적값 \(V^\star_H\)로 수렴한다. ∎</p>
<p><b>따름 (근사 버전 — 실제 학습에 해당).</b> 개선이 \(\delta\)-근사 greedy, 즉
\(T^{\pi_{n+1}}V^{\pi_n,H}\ge TV^{\pi_n,H}-\delta\)이면, 표준 근사 greedy 바운드가 할인 \(\gamma^H\)에서</p>
\[ \limsup_n\ \Delta_{\mathrm{epis}}(\pi_n)\ \le\ \frac{\delta}{1-\gamma^H}\ =\ \delta\,\bar H \]
<p>를 준다. 즉 <b>epistemic 잔여는 actor의 최적화 오차에 chunk-유효지평을 곱한 만큼</b>이다. 이것이 우리가
FQL one-step actor로 actor 최적화를 정확·저렴하게 만들려는 이유의 정량적 근거다.</p>

<h4>명제 E — 왜 개선을 "선택된 짧은 \(k\)"에서 하면 안 되는가 (피아노 비유의 엄밀형)</h4>
<p><b>명제.</b> 개선을 배포 길이 \(k(s)<H\)의 prefix 값 \(Q_k\)에 대해 수행한다고 하자. 그러면 그 목적함수는
chunk의 꼬리 \(a_{k(s):H}\)에 <b>전혀 의존하지 않는다</b>. 따라서 꼬리는 어떤 압력도 받지 않으며
\(V^{\pi,H}\)는 증가할 이유가 없다 — \(\Delta_{\mathrm{epis}}\)가 0에서 떨어진 채 영구히 남을 수 있다.</p>
<p><i>증명.</i> \(Q^{\pi,\kappa}_k(s,a)=\mathbb E[\sum_{j<k}\gamma^jr(s_j,a_j)+\gamma^kV^{\pi,\kappa}(s_k)]\)는
\(a\)에 오직 \(a_{0:k}\)를 통해서만 의존한다. 그러므로 \(a_{k:H}\)를 임의로 바꿔도 목적함수 값이 같고, argmax는
꼬리를 결정하지 않는다. ∎</p>
<div class='callout warn'><span class='k'>이것이 선행 연구와 갈리는 지점</span>
AQC·ExRL·ACSAC는 <b>고르기만</b> 한다 — 정책에 어떤 개선 압력도 넣지 않으므로 명제 E의 극단이다
(꼬리는커녕 머리도 개선되지 않는다). 그래서 그들의 성능 천장은 base 정책이 제안하는 것에 갇힌다.</div>

<h4>정리 B — 적응 실행은 안전하다 (하한을 깨지 않는다)</h4>
<p><b>정리.</b> 임의의 \(\pi\)에 대해, \(V^{\pi,H}\)를 연속값으로 써서 greedy 선택자
\(\kappa_1(s)\in\arg\max_{k\in\mathcal K}Q_k(s,\pi(s))\)를 만들면 \(V^{\pi,\kappa_1}\ge V^{\pi,H}\) (점별).</p>
<p><i>증명.</i> 고정된 \((\pi,\kappa)\)에 대해 연산자
\((\mathcal T^{\pi,\kappa}V)(s):=\mathbb E\big[\sum_{j<\kappa(s)}\gamma^jr+\gamma^{\kappa(s)}V(s_{\kappa(s)})\big]\)는
<b>단조</b>이고 계수 \(\max_s\gamma^{\kappa(s)}\le\gamma^{k_{\min}}<1\)의 <b>수축</b>이므로 유일한 고정점
\(V^{\pi,\kappa}\)를 갖는다. 이제 \(\mathcal K\ni H\)이므로</p>
\[ V^{\pi,H}(s)=Q_H(s,\pi(s))\ \le\ \max_{k\in\mathcal K}Q_k(s,\pi(s))=\big(\mathcal T^{\pi,\kappa_1}V^{\pi,H}\big)(s) \]
<p>단조성으로 \(\mathcal T^{\pi,\kappa_1}\)를 반복 적용하면 좌변은 계속 증가하며 고정점으로 수렴하므로
\(V^{\pi,H}\le V^{\pi,\kappa_1}\). ∎</p>

<h4>따름정리 C — 주 결과: 방식이 도달하는 곳</h4>
<p>정리 A와 B를 합치면, 배포되는 값이</p>
\[ V^{\pi_n,\kappa}\ \ \ge\ \ V^{\pi_n,H}\ \ \longrightarrow\ \ V^\star_H \]
<p>이고, 따라서 진짜 closed-loop 최적과의 격차가</p>
\[ \limsup_n\ \big[V^\star_1-V^{\pi_n,\kappa}\big]\ \le\ V^\star_1-V^\star_H\ =\ \Delta_{\mathrm{alea}}
\ \le\ \varepsilon_H H\bar H,\qquad \varepsilon_H=3\big(1-(1-\varepsilon)^{H-1}\big) \]
<p>로 묶인다 (마지막 부등식은 DQC Corollary 1 + Proposition 4). 그리고 <b>결정론적·완전관측 세계에서는
\(\Delta_{\mathrm{alea}}=0\)</b> (Part III.3 정리 2).</p>
<div class='callout'><span class='k'>말로 옮기면</span>
<b>decoupled-horizon 방식은 진짜 최적값의 "aleatoric floor 이내"로 수렴하며, 그 floor는 예측 가능한 세계에서
정확히 0이다.</b> 즉 우리가 못 가져오는 것은 <b>원리적으로 아무도 못 가져오는 것</b>뿐이다. 그리고 적응 실행은
그 하한을 <b>깨지 않고 더 얹을 수만</b> 있다(정리 B).</div>
<p class='sub'><b>정직한 단서.</b> 정리 A는 개선용 critic이 <b>full-commitment 연속값 \(V^{\pi,H}\)</b>로
부트스트랩한다는 정의에 의존한다. 만약 개선용 critic이 <b>배포(적응) 연속값</b>으로 부트스트랩하면 개선 루프가
\(M_H\) 안에 닫히지 않아 위 수렴 논증이 그대로 적용되지 않는다. 이것은 추상적 걱정이 아니라 <b>구체적 설계
지시</b>다 — 그리고 우리가 ablation으로 확인할 항목이다(두 부트스트랩 방식 비교).</p>

<h4>정리 D — 종점: 길이는 어디까지 자라는가</h4>
<p>"엄격히 짧은 것이 나은" 상태 집합을
\(\mathcal S_<(\pi):=\{s:\max_{k<H}Q_k(s,\pi(s))>Q_H(s,\pi(s))\}\)로 두자.</p>
<p><b>정리.</b> 완전관측·결정론적 dynamics에서 \(\pi^\star=\pi^\star_H\)(정리 A의 극한)이면, 모든 \(s\)와 모든
\(k\in\mathcal K\)에 대해 \(Q_k(s,\pi^\star(s))=Q_H(s,\pi^\star(s))\)이다. 따라서
\(\mathcal S_<(\pi^\star)=\varnothing\)이고 \(\kappa\equiv H\)가 최적이며, 동률에서 긴 쪽을 택하는 사전식
규칙 아래 <b>평균 commitment 길이는 정확히 \(H\)</b>가 된다.</p>
<p><i>증명.</i> Part III.3 정리 2에 의해 \(V^\star_H=V^\star_1\)이고 \(\pi^\star\)가 그것을 달성하므로
\(\pi^\star\)는 임의의 \(s\)에서 <b>최적 궤적</b>을 그린다. \(k\)만큼 실행한 뒤 \(s_k\)에서 재질의하면
\(\pi^\star(s_k)\)를 얻는데, 결정론성·Markov성·최적성에 의해 이는 <b>같은 최적 궤적의 연속</b>이다. 따라서
실현 궤적이 \(k\)에 무관하게 동일하고 값도 같다. ∎</p>
<p><b>일반적인 경우.</b> \(\mathcal S_<(\pi^\star)\)에서 얻을 수 있는 이득의 총량은
\(\Delta_{\mathrm{react}}:=V^\star_{\mathrm{ada}}-V^\star_H\le\Delta_{\mathrm{alea}}\)로 묶인다(Part III.5).
즉 <b>평균 길이는 자라되, 진짜로 반응이 필요한 만큼의 floor에서 멈춘다</b> — 9절 그림의 오른쪽 패널이 바로
이 정리의 그림이다.</p>

<h4>이 네 정리가 함께 말하는 것</h4>
<div class='tblwrap'><table>
<tr><th>정리</th><th>주장</th><th>무엇을 보장하나</th></tr>
<tr><td><b>A</b></td><td>개선 루프 = \(M_H\)에서의 policy iteration</td><td>epistemic 항 → 0 (근사 시 \(\delta\bar H\) 이내)</td></tr>
<tr><td><b>E</b></td><td>짧은 \(k\)에서 개선하면 꼬리에 압력이 없다</td><td><b>왜 반드시 \(k=H\)여야 하는지</b></td></tr>
<tr><td><b>B</b></td><td>greedy 선택은 \(V^{\pi,H}\)를 깨지 않는다</td><td>적응 실행의 <b>안전성</b></td></tr>
<tr><td><b>C</b></td><td>배포값 → \(V^\star_H\) 이상, 격차 ≤ aleatoric floor</td><td><b>방식이 도달하는 곳</b> (예측 가능한 세계에선 최적)</td></tr>
<tr><td><b>D</b></td><td>종점에서 짧게 끊을 이유가 사라진다</td><td><b>curriculum</b>이 정리의 귀결임</td></tr>
</table></div>
<p>요컨대 <b>"평가는 나누고, 개선은 합친다"</b>는 설계가 임의의 취향이 아니라, (i) 개선을 올바른 MDP 안에 닫아
수렴을 얻고 (ii) 선택은 그 수렴을 깨지 않으며 (iii) 남는 격차가 정확히 아무도 못 가져오는 부분이라는,
<b>증명 가능한 구조</b>다.</p>
"""
