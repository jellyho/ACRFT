"""Plain-language companion entry for chunking-theory. KO+EN bodies."""

FIG = "figures/chunking-easy/bias_vs_k.png"

KO = r"""
<p class='sub'>이 글은 <span class='xref' data-eid='chunking-theory'>action chunking의 수학</span>(정리·증명
전체판)의 <b>쉬운 판</b>이다. 수식 없이 <b>하나의 비유</b>로 처음부터 끝까지 간다. 목표는 "왜 어떤 상황에선
짧게 끊는 게 좋고 어떤 상황에선 길게 가는 게 좋은가", 그리고 <b>"그 이득 중 무엇을 우리가 정책에 흡수할 수
있는가"</b>를 수학 배경 없이도 이해하는 것이다. 각 절 끝에 대응하는 정식 정리를 괄호로 달아 두었으니 필요하면
전체판으로 넘어가면 된다.</p>

<h3>0. 기본 비유 — 맛보는 셰프의 일지</h3>
<p>셰프가 요리하며 일지를 쓴다. <i>"A → B → C 했더니 훌륭했다."</i> 그런데 셰프는 <b>중간중간 맛을 보며</b>
간을 조절했다. 당신이 그 일지를 보고 A→B→C를 <b>맛도 안 보고 그대로</b> 따라 하면? 같은 결과가 안 나온다.</p>
<div class='tblwrap'><table>
<tr><th>비유</th><th>우리 문제</th><th>논문 용어</th></tr>
<tr><td>일지에 적힌 값</td><td>데이터로 학습한 critic 값</td><td><b>nominal</b> \(\hat V_{ac}\)</td></tr>
<tr><td>눈 감고 따라 했을 때 실제 값</td><td>chunk를 실제 open-loop 실행한 값</td><td><b>actual</b> \(V_{ac}\)</td></tr>
</table></div>
<p><b>DQC라는 논문의 전부는 "이 둘이 다르다, 그리고 얼마나 다른지 정확히 계산할 수 있다"이다.</b> 이전 연구들은
이 둘을 같다고 암묵적으로 가정했다.</p>
<p><b>우리 상황에 대입하면</b>: teleop 데이터 = <b>사람이 화면을 보며 반응한 것</b>(맛보는 셰프).
action chunk 실행 = <b>눈 감고 레시피 따라 하기</b>. 즉 이 문제는 우리에게 남의 일이 아니다.</p>

<h3>1. "운 좋은 성공" — 이 문헌 최고의 통찰</h3>
<p>가장 중요한 이야기라 아주 구체적으로 간다. 로봇이 갈림길에 있다. <b>왼쪽(행동 0)</b>으로 간다. 그러면 동전이
던져져 <b>B</b> 또는 <b>C</b>에 도착한다. 사람은 <b>어디에 왔는지 보고</b> 알맞은 다음 행동을 한다 — B면 0, C면 1.</p>
<p>이제 데이터를 보자. <b>"(0, 0)"이라는 두 스텝짜리 chunk는 언제 기록되었나?</b> 오직 <b>B에 도착했을
때</b>뿐이다. C에서는 사람이 1을 했으니까. 그래서 critic이 데이터에서 배우는 것은:</p>
<div class='callout'><span class='k'>critic이 배우는 것</span>
"(0, 0)을 하면 <b>100%</b> B를 거쳐 대박이 난다"</div>
<p>하지만 실제로 (0,0)을 <b>눈 감고</b> 실행하면? 동전이 C를 주는 절반의 경우 두 번째 0은 완전히 틀린
행동이라 망한다. <b>critic은 "사람이 반응해서 성공한 것"과 "이 chunk가 좋아서 성공한 것"을 구별할 방법이
없다.</b> 원문 표현:</p>
<blockquote>"the chunked critic has no way of differentiating a low-probability, 'lucky' success from a
closed-loop, high-probability success." (DQC §4.4)</blockquote>
<figure><img src="figures/chunking-easy/fig_lucky.png" alt="fig_lucky.png"/><figcaption><b>DQC Proposition 1의 반례를 실제로 계산한 것</b>(γ=0.99, 동전 확률 δ=0.5). 보라 = critic이 <b>믿는</b> 값, 주황 = <b>눈 감고 실행하면 실제로</b> 얻는 값. chunk (0,0)에 대해서만 둘이 크게 어긋난다 — 데이터에서 그 chunk가 <b>운 좋게 B에 도착했을 때만</b> 기록됐기 때문이다. critic은 99를 믿고 그것을 고르지만 실제로는 <b>49</b>를 얻고, 정작 다른 선택지는 <b>89</b>를 줬을 것이다. <b>가장 크게 착각하고 있는 후보를 정확히 골라내는 것</b>이 argmax의 성질이다.</figcaption></figure>
<p>우리는 이것을 <b>hindsight leakage(사후정보 누출)</b>라 부른다. 사람 teleop은 100% "맛보는 셰프"이므로
<b>이 병리가 우리 yam·RoboCasa 데이터에 구조적으로 존재한다</b>. (전체판 I.8 / DQC Proposition 1)</p>

<h3>2. ε — "눈 감고 따라 하기가 얼마나 어긋나나"</h3>
<p>DQC는 이 어긋남을 확률 거리로 재고 그것을 ε라 부른다. 조건이 두 종류다.</p>
<div class='tblwrap'><table>
<tr><th>조건</th><th>뜻</th><th>보장하는 것</th></tr>
<tr><td><b>약한(weak)</b></td><td><i>평균적으로</i> 눈 감고 따라 해도 비슷한 곳에 도착</td><td>가치 <b>추정</b>은 정확 — 그러나 <b>정책은 못 지킴</b></td></tr>
<tr><td><b>강한(strong)</b></td><td><i>어떤 레시피를 집어도 하나하나 다</i> 비슷한 곳에 도착</td><td>정책까지 보장</td></tr>
</table></div>
<p><b>왜 이 구분이 목숨인가.</b> 평균만 맞으면, 1절의 "운 좋은 (0,0)" 같은 <b>특정 레시피를 골라잡을</b> 수
있다. 그런데 critic이 하는 일이 정확히 <b>"가장 좋아 보이는 걸 고르기"</b>다. 평균 보장은 골라잡기 앞에서
무력하다. 그래서 약한 조건만으로는 학습이 <b>값 범위의 절반만큼</b>도 망할 수 있다. (전체판 I.2·I.8·I.9)</p>

<h3>3. 오차 공식이 왜 그 모양인가 — 새는 양동이</h3>
<p>두 숫자를 먼저 익히자.</p>
<div class='tblwrap'><table>
<tr><th>기호</th><th>뜻</th><th>γ=0.99일 때</th></tr>
<tr><td>\(H=\frac{1}{1-\gamma}\)</td><td>"실질적으로 몇 스텝이 중요한가"</td><td>100</td></tr>
<tr><td>\(\bar H=\frac{1}{1-\gamma^h}\)</td><td>같은 것을 <b>chunk 단위</b>로 센 것</td><td>h=10이면 약 10.5</td></tr>
</table></div>
<p>이제 그림 하나로 정리된다.</p>
<div class='callout'><span class='k'>새는 양동이</span>
chunk를 하나 실행할 때마다 확률 질량의 <b>ε만큼이 대본에서 이탈</b>한다. 이탈한 부분은 <b>최악의 경우 가능한
최대치(=\(H\))만큼</b> 틀릴 수 있다. 그리고 이 이탈이 <b>chunk마다 한 번씩, 총 \(\bar H\)번</b> 일어난다.
<br>→ 오차 ≈ <b>\(\varepsilon\times H\times\bar H\)</b></div>
<p>여기 미묘한 핵심이 하나 있다. 보통 이런 재귀는 매번 \(\gamma^h\)씩 줄어드는데, 실제 수축은
<b>\((1-\varepsilon)\gamma^h\)</b>다. 왜냐하면 <b>새어나간 ε는 "조금 줄어든 채 남는" 게 아니라 그냥
없어지기</b> 때문이다. 새는 양동이에서 샌 물은 돌아오지 않는다. 그래서 분모에 \((1-\varepsilon)\)가 붙는다.
(전체판 I.4 / DQC Theorem 1, 그리고 이 상한이 정확히 달성되는 반례가 Theorem 2)</p>

<h3>4. 가장 우아한 한 줄</h3>
<p><b>"일지를 쓴 셰프가 완벽한 셰프였다면?"</b> 그러면 일지에 적힌 값이 곧 <b>진짜 최적값</b>이다. 그러면
3절에서 잰 "일지 vs 눈 감고 따라 하기"의 차이가, 그대로 <b>"최적 vs 가능한 최선의 눈 감고 하기"</b>의 차이가
된다. <b>새 증명 한 줄 없이</b> 오차 공식이 곧 <b>"눈 감고 commit하는 것의 대가"</b> 공식이 된다.
이것이 이 문헌 전체에서 그 대가를 잰 유일한 식이다. (전체판 I.6 / DQC Corollary 1)</p>

<h3>5. 그 대가의 정체 = 세상의 예측 불가능성</h3>
<p>ε의 정체가 밝혀진다. 세상이 확률 \((1-\varepsilon)\)로 예측대로 가고 가끔 어긋난다면,</p>
\[ \varepsilon_h\ =\ 3\times\big(\text{h−1 스텝 중 한 번이라도 어긋날 확률}\big) \]
<p>두 가지가 곧바로 따라 나온다.</p>
<ul>
<li><b>세상이 완전히 예측 가능하면 대가는 정확히 0.</b> 눈 감고 따라 해도 손해가 없다.</li>
<li><b>레시피가 길수록 어긋날 기회가 많아져 대가가 커진다.</b></li>
</ul>
<figure><img src="figures/chunking-easy/fig_epsilon.png" alt="fig_epsilon.png"/><figcaption>세상이 예측 불가능해질수록(가로축 ε) 눈 감고 commit하는 대가가 커진다. 핵심은 <b>왼쪽 끝</b>이다 — <b>ε=0에서는 어떤 commitment 길이든 대가가 정확히 0</b>이다. 이것이 7절 정리의 그림이다: 뜻밖의 일이 없으면 미리 다 적어두는 것과 도중에 다시 생각하는 것이 <b>완전히 같다</b>.</figcaption></figure>
<p>아래 그림이 그 두 번째를 계산한 것이다(우리가 직접 계산; 스크립트 <code>scripts/fig_chunking_bias.py</code>).</p>
<figure><img src="FIGSRC" alt="bias bound vs commitment length"/>
<figcaption><b>왼쪽</b>: commitment 길이 k가 커질수록 편향 상한이 커진다(단조 증가 — 우리가 수치로 확인).
<b>오른쪽</b>: 그런데 같은 값을 "값 범위의 몇 %"로 다시 그리면, 상한이 <b>값 범위의 20~90%를 먹어버린다</b>.
즉 이 상한은 <b>"길수록 나쁘다"는 방향은 확립하지만, 큰 k에서는 사실상 아무 말도 하지 않는다</b>. 그래서
실제 편향은 <b>실측해야 한다</b>(9절 1번 실험). γ가 1에 가까울수록(우리 yam은 0.99964) 상황이 더 나쁘다.</figcaption></figure>

<h3>6. "그럼 짧게 끊는 게 항상 낫나?" — 아니다</h3>
<p>직관적으로 짧게 끊으면 안전할 것 같다. 그런데 DQC는 <b>매 스텝 다시 결정하는 것이 최악의 경우 \(H\)배 더
나쁠 수 있다</b>고 증명한다.</p>
<div class='callout'><span class='k'>이유</span>
<b>다시 결정할 때마다 잘못 결정할 기회가 생긴다.</b> 당신의 critic은 완벽하지 않다. 결정 횟수가 100배면
실수할 기회도 100배다.</div>
<p>그런데 <b>반대 방향도 증명된다</b>: 매 스텝 결정은 거의 최적인데 같은 정책을 chunk로 실행하면 처참한 세상도
존재한다. 즉 <b>어느 쪽도 일반적으로 우월하지 않다. 세상과 데이터의 성질에 달렸다.</b></p>
<div class='callout'><span class='k'>이것이 adaptive의 근거다</span>
고정된 하나의 길이로는 <b>어떤 세상에서든 반드시 한쪽을 잃는다</b>. "상태마다 다르게 정하자"는 취향이 아니라
정리가 요구하는 것이다. (전체판 I.11 / DQC Proposition 3 vs Theorems 5·6)</div>

<h3>7. 우리 핵심 정리 — 한 문장으로</h3>
<div class='callout'><span class='k'>정리 (쉬운 말)</span>
<b>세상에 뜻밖의 일이 없다면, 도중에 멈춰 다시 생각해서 얻을 수 있는 모든 것은, 애초에 한 번에 다 적어둘 수
있었다.</b></div>
<p><b>왜 그런가.</b> 뜻밖의 일이 없으면 당신은 <b>지금 이 순간에 이미</b> 5스텝 뒤에 어디 있을지 정확히 안다.
그러면 <i>"5스텝 뒤에 멈춰서 다시 생각하면 뭐라고 결정할까?"</i>도 <b>지금 머릿속에서 미리 굴려볼 수 있다.</b>
그 답을 지금 그냥 적어두면 된다. 그 적어둔 계획을 그대로(중간에 멈추지 않고) 실행하면 <b>똑같은 궤적</b>이
나온다. 그러므로 멈춰 생각하는 것의 값어치가 0이다. (전체판 III.3, 증명 포함)</p>
<p><b>따라서 도중에 끊는 게 도움이 되었다면, 이유는 정확히 둘 중 하나다.</b></p>
<div class='tblwrap'><table>
<tr><th></th><th>이유</th><th>이름</th><th>고칠 수 있나</th></tr>
<tr><td><b>(가)</b></td><td>정책이 미숙해서 애초에 옳은 계획을 못 적었다</td><td>epistemic</td><td><b>가능 — 정책을 개선하면 흡수된다</b></td></tr>
<tr><td><b>(나)</b></td><td>진짜로 예측 불가능한 일이 일어난다</td><td>aleatoric</td><td>불가능</td></tr>
</table></div>
<p><b>(가)를 수확해서 정책에 집어넣는 것 — 그것이 우리가 하는 일이고, 선행 연구가 비워둔 칸이다.</b></p>
<p>그리고 (나)의 원천은 둘이다: <b>세상의 확률성</b> + <b>부분관측</b>(카메라가 못 보는 것 — 가림, 접촉 순간의
힘). VLA는 이미지로 세상을 보므로 후자가 실재한다. 그래서 <b>우리 floor는 0이 아니다</b>. (전체판 III.4)</p>

<h3>8. 왜 "전체 chunk"로 개선해야 하나 — 피아노 비유</h3>
<div class='callout'><span class='k'>비유</span>
피아노 곡을 배우는데 <b>항상 앞 2마디만 연습</b>하면, 곡 전체를 한 번에 치는 실력은 영영 늘지 않는다.</div>
<p>adaptive가 "여기선 2마디만 치고 다시 봐"라고 한다고 해서 <b>개선(채점)까지 2마디로 하면</b>, 앞부분만
좋아지고 <b>전곡 실력(= full chunk 품질)은 그대로</b>다. 그러면 7절의 (가)가 줄지 않고, 평균 길이도 자라지
않는다. <b>그래서 채점은 반드시 "전곡 연주"로 해야 한다.</b> 이것이 우리와 AQC·ExRL을 가르는 지점이다 —
그들은 <b>고르기만</b> 하고 정책을 개선하지 않는다.</p>
<p>그리고 이게 안전한 이유도 증명된다: adaptive 실행은 언제나 "끝까지 commit"을 선택지로 갖고 있으므로,
<b>전곡 점수를 올리는 것은 실제 배포 성능의 하한을 올리는 것</b>이다. 손해 볼 수 없다. (전체판 III.6, Lemma B)</p>

<h3>9. 길이가 저절로 자란다 — 보상 추가 없이</h3>
<p>정책이 좋아질수록 → <b>"여기선 끊는 게 확실히 낫다"는 상황이 줄어든다</b> → 자연히 <b>평균 commit 길이가
늘어난다</b>. <b>보너스도 페널티도 넣지 않는다.</b> 그냥 (가)가 사라지면서 생기는 결과다.</p>
<figure><img src="figures/chunking-easy/fig_curriculum.png" alt="fig_curriculum.png"/><figcaption><b>모식도 — 이론이 예측하는 모양이며 측정 데이터가 아니다.</b> 왼쪽: 최적과의 격차가 두 조각으로 갈린다 — <b>정책이 흡수하는 부분(epistemic)</b>은 학습과 함께 줄고, <b>회수 불가능한 부분(aleatoric)</b>은 그대로 남는다. 오른쪽: 그 결과 <b>평균 commit 길이가 저절로 자라</b> aleatoric floor에서 멈춘다. 보상을 추가하지 않았는데도 이렇게 되는 것이 요점이며, <b>정책 개선을 끄면 이 곡선이 자라지 않아야 한다</b>(반증 조건).</figcaption></figure>
<p>그래서 이 곡선(평균 길이 ↑)이 <b>"정책이 실제로 좋아졌다"는 증거</b>가 된다 — 우리 논문의 대표 그림이 될
것이다. 그리고 이것이 <b>반증 가능</b>하다는 점이 중요하다: 정책 개선을 끄면 길이가 자라지 <b>않아야</b> 한다.
자란다면 그건 우리 기여가 아니다. (전체판 III.7)</p>

<h3>10. 선행 연구(AQC)의 문제 — 숫자로</h3>
<div class='tblwrap'><table>
<tr><th>문제</th><th>쉬운 설명</th></tr>
<tr><td><b>γ로 나누기</b></td><td>"긴 chunk는 할인 때문에 점수가 낮아 보이니 보정하자"는 것인데, 이는 <b>보상이 양수일 때만</b> 맞다. 그런데 그들 벤치마크는 실패 시 매 스텝 −1이다. 음수면 방향이 <b>정반대</b>가 된다. <b>우리 cost_to_goal도 −1이라 그대로 가져오면 안 된다.</b></td></tr>
<tr><td><b>순환 논증</b></td><td>"우리 selector가 최적 k를 찾는다"를 증명했는데, <b>"최적 k"의 정의가 "우리 selector가 완벽한 추정치로 찾았을 k"</b>다. 추정이 정확해지면 추정이 정확해진다는 말이지, 그 기준이 옳다는 증명이 아니다.</td></tr>
<tr><td><b>부호 오류</b></td><td>논문이 \(\gamma^{k^*}\ge\gamma^{k_{\min}}\)라고 쓴다. 그런데 \(k^*\)는 \(k_{\min}\)보다 크거나 같고, \(\gamma<1\)이면 <b>지수가 클수록 값은 작아진다</b>. 숫자로: γ=0.9, \(k_{\min}\)=1, \(k^*\)=5 → 0.9 vs 0.59. <b>0.59 ≥ 0.9는 거짓.</b> 중심 정리의 마지막 단계가 여기 걸려 있다.</td></tr>
<tr><td><b>정책 개선 없음</b></td><td>BC가 뽑아준 후보 중에서 <b>고르기만</b> 한다. BC 자체를 좋게 만들지 않는다(RoboCasa 실험은 아예 backbone 전부 동결, critic head만 학습). 그래서 성능 천장이 <b>"BC에서 N번 뽑은 것 중 최고"</b>로 막혀 있다.</td></tr>
</table></div>
<p>(넷 다 원문 대조로 직접 확인했다. 근거 인용은 전체판 Part II에.)</p>

<h3>11. 그래서 우리가 제일 먼저 측정할 것</h3>
<p>1절의 누출로 돌아간다. 누출은 <b>chunk를 조건으로 걸 때</b> 생긴다. 그러면 <b>긴 chunk일수록 더 많이
샌다</b> → <b>critic이 긴 chunk에 대해 더 낙관적</b>이다.</p>
<div class='callout warn'><span class='k'>그래서 생기는 의심</span>
selector가 긴 chunk를 고를 때, 그게 <b>진짜 좋아서</b>인지 아니면 <b>critic이 긴 것에 더 속고 있어서</b>인지
알 수 없다. AQC의 "baseline 빼기"가 이걸 상쇄하려는 장치인데, 빼는 값은 chunk를 조건으로 걸지 않아
<b>같은 착각을 갖고 있지 않다</b> → 완전히 상쇄되지 않는다. 게다가 γ로 나누는 것이 남은 잔차를 <b>더
증폭</b>한다.</div>
<p><b>이것을 실측하는 것이 우리 1번 실험이다.</b> 아무도 하지 않았다. 사전등록한 네 가지 검증(누출 편향의
k-의존성 / 비선택 후보의 값 보정 / curriculum의 인과성 / floor의 존재)은 전체판 III.9에 있다.</p>

<h3>12. 세 줄 요약</h3>
<ol>
<li><b>chunk critic은 "사람이 반응해서 성공한 것"을 "이 chunk가 좋아서 성공한 것"으로 착각한다.</b> 그 착각의
크기를 DQC가 정확히 쟀고, 세상이 예측 가능할수록 작아져 결국 0이 된다.</li>
<li><b>짧게 끊는 것도 길게 가는 것도 일반적으로 우월하지 않다.</b> 그래서 상태마다 정하는 것이 옳다 — 이건
취향이 아니라 정리다.</li>
<li><b>도중에 끊어 얻은 이득 중, 예측 가능했던 부분은 전부 "더 좋은 한 번의 계획"으로 바꿔 넣을 수 있다.</b>
그래서 우리는 채점을 전체 chunk로 하고, 그 결과로 평균 길이가 저절로 자라는 것을 증거로 삼는다.</li>
</ol>
"""

EN = r"""
<p class='sub'>This is the <b>plain-language companion</b> to
<span class='xref' data-eid='chunking-theory'>the mathematics of action chunking</span> (the full version with
theorems and proofs). It runs on <b>one analogy</b> from start to finish, with almost no symbols. The goal is to
understand, without a maths background, why short commitments help in some states and long ones in others — and
above all <b>which part of that gain we can absorb into the policy</b>. Each section ends with a pointer to the
corresponding formal result.</p>

<h3>0. The analogy — a chef's diary</h3>
<p>A chef cooks and keeps a diary: <i>"I did A → B → C and it came out great."</i> But the chef was
<b>tasting as they went</b> and adjusting. If you follow A→B→C <b>without ever tasting</b>, you will not get the
same dish.</p>
<div class='tblwrap'><table>
<tr><th>Analogy</th><th>Our problem</th><th>Paper's term</th></tr>
<tr><td>the value written in the diary</td><td>the critic value learned from data</td><td><b>nominal</b> \(\hat V_{ac}\)</td></tr>
<tr><td>the value you actually get following it blind</td><td>the value of executing the chunk open-loop</td><td><b>actual</b> \(V_{ac}\)</td></tr>
</table></div>
<p><b>The whole of DQC is: these two differ, and we can compute exactly by how much.</b> Prior work implicitly
assumed they were the same.</p>
<p><b>For us</b>: teleop data = <b>a human reacting to what they see</b> (the tasting chef); executing an action
chunk = <b>following the recipe blind</b>. So this is not someone else's problem.</p>

<h3>1. The "lucky success" — the best insight in this literature</h3>
<p>The most important story, so let us be concrete. A robot is at a fork. It goes <b>left (action 0)</b>. A coin
is flipped and it lands in <b>B</b> or <b>C</b>. The human <b>sees which</b> and picks the right follow-up —
0 at B, 1 at C.</p>
<p>Now look at the data. <b>When was the two-step chunk "(0, 0)" ever recorded?</b> Only when the robot landed in
<b>B</b> — because at C the human did 1. So what the critic learns from the data is:</p>
<div class='callout'><span class='k'>what the critic learns</span>
"doing (0, 0) leads through B to a great outcome <b>100%</b> of the time"</div>
<p>But executing (0,0) <b>blind</b>? In the half of cases where the coin gives C, the second 0 is exactly wrong
and it fails. <b>The critic has no way to separate "we succeeded because the human reacted" from "this chunk is
good".</b> In the paper's words:</p>
<blockquote>"the chunked critic has no way of differentiating a low-probability, 'lucky' success from a
closed-loop, high-probability success." (DQC §4.4)</blockquote>
<figure><img src="figures/chunking-easy/fig_lucky.png" alt="fig_lucky.png"/><figcaption><b>DQC Proposition 1's counterexample, actually computed</b> (γ=0.99, coin probability δ=0.5). Purple = what the critic <b>believes</b>; orange = what open-loop execution <b>actually</b> gets. The two diverge only for the chunk (0,0) — because in the data that chunk was recorded <b>only when the coin luckily gave B</b>. The critic believes 99 and picks it, actually gets <b>49</b>, while the alternatives would have given <b>89</b>. An argmax <b>selects precisely the candidate it is most deluded about</b>.</figcaption></figure>
<p>We call this <b>hindsight leakage</b>. Human teleop is 100% "tasting chef", so <b>this pathology is
structurally present in our yam and RoboCasa data</b>. (full version I.8 / DQC Proposition 1)</p>

<h3>2. ε — "how far does blind following drift?"</h3>
<p>DQC measures that drift as a probability distance and calls it ε. There are two conditions.</p>
<div class='tblwrap'><table>
<tr><th>Condition</th><th>Meaning</th><th>What it guarantees</th></tr>
<tr><td><b>weak</b></td><td><i>on average</i>, blind following lands in about the same place</td><td>value <b>estimation</b> is accurate — but the <b>policy is not protected</b></td></tr>
<tr><td><b>strong</b></td><td><i>for every individual recipe</i>, blind following lands in about the same place</td><td>the policy is protected too</td></tr>
</table></div>
<p><b>Why this distinction is everything.</b> If only the average holds, you can still <b>cherry-pick</b> a recipe
like the "lucky (0,0)" of §1 — and cherry-picking is <i>exactly</i> what a critic does ("take the best-looking
one"). An average guarantee is useless against an argmax. Hence under the weak condition alone, learning can be
off by as much as <b>half the entire value range</b>. (full version I.2, I.8, I.9)</p>

<h3>3. Why the error formula looks like that — a leaky bucket</h3>
<div class='tblwrap'><table>
<tr><th>Symbol</th><th>Meaning</th><th>At γ=0.99</th></tr>
<tr><td>\(H=\frac{1}{1-\gamma}\)</td><td>"how many steps effectively matter"</td><td>100</td></tr>
<tr><td>\(\bar H=\frac{1}{1-\gamma^h}\)</td><td>the same thing counted in <b>chunks</b></td><td>≈10.5 at h=10</td></tr>
</table></div>
<div class='callout'><span class='k'>the leaky bucket</span>
Each time you execute one chunk, an <b>ε fraction of the probability mass goes off script</b>. Whatever goes off
script can be wrong by the <b>largest amount possible (\(H\))</b>. And this happens <b>once per chunk, i.e.
\(\bar H\) times</b>.<br>→ error ≈ <b>\(\varepsilon\times H\times\bar H\)</b></div>
<p>One subtlety matters. Normally such a recursion shrinks by \(\gamma^h\) each time; here it shrinks by
<b>\((1-\varepsilon)\gamma^h\)</b>, because <b>the ε that leaks out does not "remain, slightly reduced" — it is
gone</b>. Water that leaks from a bucket does not come back. That is the \((1-\varepsilon)\) in the denominator.
(full version I.4 / DQC Theorem 1; Theorem 2 exhibits an MDP attaining the bound exactly)</p>

<h3>4. The most elegant line</h3>
<p><b>"What if the chef who wrote the diary was a perfect chef?"</b> Then the diary's value <b>is</b> the true
optimum. And the gap measured in §3 — diary vs. blind following — becomes exactly the gap between <b>the optimum
and the best possible blind execution</b>. With <b>no new proof</b>, the error formula becomes the formula for
<b>the price of committing blind</b>. It is the only such quantification in this literature.
(full version I.6 / DQC Corollary 1)</p>

<h3>5. What that price really is: the unpredictability of the world</h3>
<p>ε gets identified. If the world goes as expected with probability \((1-\varepsilon)\) and occasionally
deviates, then</p>
\[ \varepsilon_h\ =\ 3\times\big(\text{probability of at least one deviation in h−1 steps}\big) \]
<p>Two things follow immediately:</p>
<ul>
<li><b>If the world is perfectly predictable, the price is exactly zero.</b> Following blind costs nothing.</li>
<li><b>The longer the recipe, the more chances to deviate, the higher the price.</b></li>
</ul>
<figure><img src="figures/chunking-easy/fig_epsilon.png" alt="fig_epsilon.png"/><figcaption>The less predictable the world (x-axis ε), the higher the price of committing blind. The point is the <b>left edge</b>: <b>at ε=0 the price is exactly zero for every commitment length</b>. This is the picture of the theorem in §7 — with no surprises, writing the whole plan in advance and stopping to re-think are <b>exactly equivalent</b>.</figcaption></figure>
<p>The figure below computes that second point (our own computation; script
<code>scripts/fig_chunking_bias.py</code>).</p>
<figure><img src="FIGSRC" alt="bias bound vs commitment length"/>
<figcaption><b>Left</b>: the bias bound grows with commitment length k (monotonically — we verified numerically).
<b>Right</b>: replotting the same quantity as a <b>percentage of the value range</b> shows the bound eats
<b>20–90% of it</b>. So the bound <b>fixes the direction ("longer is worse") but says essentially nothing at
large k</b> — which is why the actual bias <b>must be measured</b> (test 1 in §11). The closer γ is to 1 (our yam
uses 0.99964), the worse this gets.</figcaption></figure>

<h3>6. "So is short always safer?" — No</h3>
<p>Intuitively, re-deciding often feels safe. Yet DQC proves that <b>re-deciding every step can be \(H\) times
worse in the worst case</b>.</p>
<div class='callout'><span class='k'>why</span>
<b>Every re-decision is a fresh chance to decide wrongly.</b> Your critic is not perfect. A hundred times more
decisions means a hundred times more chances to err.</div>
<p>But the <b>opposite</b> is also proved: there are worlds where step-by-step execution is near-optimal while the
same policy executed in chunks is disastrous. So <b>neither dominates in general — it depends on the world and
the data</b>.</p>
<div class='callout'><span class='k'>this is the case for adaptive</span>
A single fixed length <b>necessarily forfeits one side in some world</b>. "Decide per state" is not a preference;
it is what the theorems demand. (full version I.11 / DQC Prop. 3 vs Theorems 5, 6)</div>

<h3>7. Our key theorem — in one sentence</h3>
<div class='callout'><span class='k'>theorem (plain words)</span>
<b>If the world holds no surprises, then everything you could gain by stopping to re-plan, you could have written
down in advance as a single plan.</b></div>
<p><b>Why.</b> With no surprises you already know, <b>right now</b>, exactly where you will be five steps from
now. So you can also simulate, <b>right now, in your head</b>, <i>"if I stopped to re-plan five steps from now,
what would I decide?"</i> — and simply write that answer down. Executing that written plan straight through
(never stopping) reproduces <b>the identical trajectory</b>. Hence stopping to think is worth nothing.
(full version III.3, with proof)</p>
<p><b>So if breaking the chunk did help, the reason is exactly one of two.</b></p>
<div class='tblwrap'><table>
<tr><th></th><th>Reason</th><th>Name</th><th>Fixable?</th></tr>
<tr><td><b>(a)</b></td><td>the policy was too immature to write the right plan</td><td>epistemic</td><td><b>Yes — policy improvement absorbs it</b></td></tr>
<tr><td><b>(b)</b></td><td>genuinely unpredictable things happen</td><td>aleatoric</td><td>No</td></tr>
</table></div>
<p><b>Harvesting (a) and putting it into the policy — that is what we do, and it is the slot prior work left
empty.</b></p>
<p>And (b) has two sources: <b>stochastic dynamics</b> and <b>partial observability</b> (what the camera cannot
see — occlusions, contact forces). A VLA sees the world through images, so the second is real. <b>Our floor is
therefore not zero.</b> (full version III.4)</p>

<h3>8. Why improvement must be graded on the whole chunk — a piano analogy</h3>
<div class='callout'><span class='k'>analogy</span>
If, while learning a piano piece, you <b>only ever practise the first two bars</b>, your ability to play the whole
piece in one go never improves.</div>
<p>Even if adaptive execution says "here, play two bars and look again", <b>grading (improvement) on two bars</b>
only improves the opening — <b>the whole-piece ability (= full-chunk quality) stays put</b>. Then (a) from §7 never
shrinks and the mean length never grows. <b>So grading must be on the full performance.</b> This is exactly what
separates us from AQC and ExRL — they only <b>select</b>, and never improve the policy.</p>
<p>It is also provably safe: adaptive execution always has "commit to the end" among its options, so <b>raising
the whole-piece score raises a lower bound on the deployed performance</b>. You cannot lose.
(full version III.6, Lemma B)</p>

<h3>9. The length grows by itself — with no added reward</h3>
<p>As the policy improves → <b>the set of situations where breaking is definitely better shrinks</b> → the
<b>mean commitment length rises</b>. <b>No bonus and no penalty is introduced.</b> It is simply what happens as
(a) disappears.</p>
<figure><img src="figures/chunking-easy/fig_curriculum.png" alt="fig_curriculum.png"/><figcaption><b>Schematic — the shape our theory predicts, not measured data.</b> Left: the gap to the optimum splits in two — the <b>part the policy absorbs (epistemic)</b> shrinks with training, while the <b>unrecoverable part (aleatoric)</b> stays. Right: as a result the <b>mean commitment length grows by itself</b> and stops at the aleatoric floor. The point is that this happens with <b>no added reward</b> — and <b>with policy improvement switched off, this curve must not grow</b> (the falsification condition).</figcaption></figure>
<p>That curve (mean length ↑) therefore becomes <b>evidence that the policy actually improved</b> — it will be our
headline figure. And crucially it is <b>falsifiable</b>: with policy improvement switched off, the length must
<b>not</b> grow. If it does, it is not our contribution. (full version III.7)</p>

<h3>10. Problems in the prior work (AQC) — with numbers</h3>
<div class='tblwrap'><table>
<tr><th>Problem</th><th>Plain explanation</th></tr>
<tr><td><b>dividing by γ</b></td><td>The intent is "long chunks look worse merely because of discounting, so correct for it" — which holds <b>only when rewards are positive</b>. But their own benchmark gives −1 per step on failure. With negative rewards the direction <b>flips</b>. <b>Our cost_to_goal is also −1, so this must not be ported.</b></td></tr>
<tr><td><b>circular argument</b></td><td>They prove "our selector finds the optimal k" — but <b>"optimal k" is defined as "what our selector would find with perfect estimates"</b>. That says the estimator is consistent, not that the criterion is right.</td></tr>
<tr><td><b>sign error</b></td><td>The paper writes \(\gamma^{k^*}\ge\gamma^{k_{\min}}\). But \(k^*\ge k_{\min}\) and, for \(\gamma<1\), <b>a larger exponent gives a smaller number</b>. Numerically: γ=0.9, \(k_{\min}\)=1, \(k^*\)=5 → 0.9 vs 0.59. <b>0.59 ≥ 0.9 is false.</b> The last step of their central theorem rests on this.</td></tr>
<tr><td><b>no policy improvement</b></td><td>They only <b>select</b> among candidates BC proposes; BC itself is never improved (the RoboCasa experiment freezes the entire backbone and trains only a critic head). So the ceiling is <b>"the best of N samples from a BC policy"</b>.</td></tr>
</table></div>
<p>(All four checked against the sources; the supporting quotations are in Part II of the full version.)</p>

<h3>11. So what we measure first</h3>
<p>Back to §1's leakage. Leakage arises from <b>conditioning on the chunk</b>. Therefore <b>longer chunks leak
more</b> → <b>the critic is more optimistic about longer chunks</b>.</p>
<div class='callout warn'><span class='k'>the resulting doubt</span>
When the selector picks a long chunk, we cannot tell whether it is <b>genuinely better</b> or whether the
<b>critic is simply more deluded about long ones</b>. AQC's "subtract a baseline" is meant to cancel this, but the
subtracted quantity does not condition on a chunk, so it <b>does not carry the same delusion</b> → the
cancellation is incomplete. And dividing by γ <b>amplifies</b> whatever residual remains.</div>
<p><b>Measuring this is our first experiment.</b> Nobody has. The four pre-registered tests (k-dependence of the
leakage bias / calibration on non-selected candidates / causality of the curriculum / existence of the floor) are
in III.9 of the full version.</p>

<h3>12. Three-line summary</h3>
<ol>
<li><b>A chunked critic mistakes "we succeeded because a human reacted" for "this chunk is good."</b> DQC measures
that mistake exactly; it shrinks as the world becomes predictable, reaching zero.</li>
<li><b>Neither short nor long commitment dominates in general.</b> Hence deciding per state is right — not a
preference, but what the theorems demand.</li>
<li><b>Of the gain obtained by breaking the chunk, everything that was predictable can be folded into one better
plan.</b> So we grade on the full chunk, and take the resulting spontaneous growth of the mean length as the
evidence.</li>
</ol>
"""

KO = KO.replace("FIGSRC", FIG)
EN = EN.replace("FIGSRC", FIG)
