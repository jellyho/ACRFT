AOLC_EN = r"""
<h3>2.5 AOLC — when the checkpoint time itself depends on the state</h3>
<p>The ε of §2 is measured at a <b>fixed time</b>: "follow the recipe blind and compare where you are
<b>exactly h steps later</b>". But we (and AQC) execute a <b>state-dependent</b> number of steps \(\kappa(s)\)
and then look again — so the checkpoints become <b>irregular</b>.</p>
<div class='callout'><span class='k'>analogy</span>
OLC is the rule "check every dish <b>after 10 minutes</b>"; AOLC is "check <b>each dish at its own time</b>".
To guarantee quality under the latter, the drift must be small at <b>all of those different times</b>.</div>
<p>AQC's <b>Adaptive Open-Loop Consistency (AOLC)</b> is exactly that — OLC's two conditions with the fixed
offset \(t+h\) replaced by the state-dependent offset \(t+\kappa(s_t)\):</p>
\[ D_{TV}\big(P_D(s_{t+\kappa(s_t)},a_{t+\kappa(s_t)}\mid s_t)\ \big\|\ P^\circ_{D,\kappa}(\cdot\mid s_t)\big)\le\varepsilon_{\mathcal K} \]
<p>and for constant \(\kappa\equiv k\) it reduces exactly to the original OLC (Prop. H.3) — hence "generalisation".</p>
<div class='callout warn'><span class='k'>Honest assessment — what we checked</span>
As a definition this is a <b>substitution of the checkpoint time</b>; no new machinery is added. Two things
remain open. ① The condition constrains a <b>single</b> adaptive step, while execution <b>composes many</b> (a
per-step guarantee does not automatically become a whole-trajectory guarantee). ② In the definition \(\kappa\) is
a <b>given, fixed</b> function, whereas the deployed selector depends on the critic and <b>keeps changing during
training</b> — indeed AQC's Theorem H.14 assumes AOLC under the <b>oracle</b> selector while bounding a policy
executed with the <b>learned</b> one.</div>
"""

BON_EN = r"""
<h3>8.6 "Isn't best-of-N also policy improvement?" — Yes. The question is where it stops</h3>
<p>A fair objection. The <b>best-of-N (BoN)</b> used by AQC and ACSAC <i>is</i> improvement: picking the candidate
the critic rates highest beats the average. So "only we improve" would be wrong. <b>The precise difference is
where the improvement hits its ceiling.</b></p>
<p>Formally let \(\pi^{BoN}_N(s):=\arg\max_{a\in\{a^1,\dots,a^N\},\,a^i\sim\beta(\cdot\mid s)}Q(s,a)\).</p>

<h4>Proposition F — BoN's improvement stops at a quantile</h4>
<p>The maximum of \(N\) samples is the <b>largest order statistic</b> of \(Q\) under \(\beta\), which estimates
that distribution's \(\tfrac{N-1}{N}\)-<b>quantile</b>. This is not our claim but the reason DQC gives for setting
\(\kappa_b=(N-1)/N\):</p>
<blockquote>"the Q-value obtained from best-of-N sampling can be seen as the largest order statistic of a random
batch (of size N) of the behavior Q-values. Such statistic estimates the behavior Q-value distribution's
\((N-1)/N\)-quantile" (DQC §5)</blockquote>
<p>So <b>BoN aims at a fixed quantile, not at the maximum</b>. With \(N=10\) that is the 90th percentile, and the
only way to raise the ceiling is to raise \(N\).</p>

<h4>Proposition G — and that quantile pays the curse of chunk dimension</h4>
<p>The chunk space is \(\mathcal A^H\) with effective dimension \(D=H\times\dim\mathcal A\) (e.g. \(H=30\),
14-DoF → <b>420</b>). If each coordinate independently lands in a "good" range with probability \(p<1\), a random
draw is good in every coordinate with probability \(p^D\), so approaching the maximum needs \(N\sim p^{-D}\).
Actual \(N\) is <b>10–32</b>. A <b>gradient-based improvement (our actor) does not pay this curse</b> — it moves
parameters rather than picking samples.</p>

<h4>Proposition H — the support ceiling</h4>
<p>A BoN action always lies in \(\mathrm{supp}\,\beta(\cdot\mid s)\). Hence for every \(N\),</p>
\[ V^{\pi^{BoN}_N}\ \le\ V^{\beta\text{-in-support greedy}}\ \le\ V^\star_H \]
<p>with equality only if an optimal chunk is inside the support at every reachable state. Our actor's constraint is
<b>soft</b>: in \(-Q+\alpha\|\mu_\omega-\mu_\theta\|^2\) one can <b>pay \(\alpha\) and move outside</b>. (Honestly:
so our ceiling is not literally \(V^\star_H\) either but the <b>\(\alpha\)-regularised optimum</b>, which tends to
\(V^\star_H\) as \(\alpha\to0\) at the price of OOD risk — that is the knob we control.)</p>

<h4>Proposition I — the decisive difference: selection alone cannot produce a curriculum</h4>
<p><b>Proposition.</b> If \(\beta\) is never updated (BoN / selection-only), the strictly-short set
\(\mathcal S_<\) is a function of \((\beta,Q)\) alone. As the critic converges, \(\mathcal S_<\) converges to
<b>some fixed set</b> and not to \(\varnothing\) (unless \(\beta\) was already optimal). Hence the <b>mean
commitment length is stationary — no curriculum arises.</b></p>
<p><i>Proof.</i> \(\mathcal S_<\) depends only on \(\beta\) and the value function, and no step of the algorithm
changes \(\beta\). ∎</p>
<div class='callout'><span class='k'>So, precisely</span>
<b>BoN is improvement — but improvement up to a quantile of a fixed proposal.</b> The proposal itself never moves.
Our scheme <b>amortises</b> improvement into the actor's parameters, so the proposal does move; that is what makes
Theorem A fire (\(\Delta_{\mathrm{epis}}\to0\)) and hence Theorem D's curriculum appear. <b>Proposition I is
simultaneously our pre-registered ablation</b>: with policy improvement switched off, the mean length must
<b>not</b> grow.</div>
<div class='tblwrap'><table>
<tr><th></th><th>set the improvement searches</th><th>ceiling</th><th>proposal moves?</th><th>curriculum</th></tr>
<tr><td><b>BoN (AQC, ACSAC)</b></td><td>\(N\) samples from a fixed \(\beta\)</td><td>\(\tfrac{N-1}{N}\)-quantile ≤ in-support greedy</td><td><b>no</b></td><td><b>none</b> (Prop. I)</td></tr>
<tr><td><b>ours (DH + one-step actor)</b></td><td>all of \(\mathcal A^H\), softly penalised by \(\alpha\)</td><td>\(\alpha\)-regularised optimum → \(V^\star_H\) as \(\alpha\to0\)</td><td><b>yes</b> (amortised)</td><td><b>yes</b> (Thm. D)</td></tr>
</table></div>
"""

TIE_EN = r"""
<h3>8.7 "In the ideal case all \(k\) are tied" — true. So why not in reality?</h3>
<p>We can now answer this precisely. <b>In the ideal limit all \(k\) really are tied.</b></p>
<p><b>Proposition J (ideal tie).</b> Under full observability and deterministic dynamics with
\(\pi=\pi^\star_H\), for every \(s\) and every \(k\in\mathcal K\), \(Q_k(s,\pi(s))=Q_H(s,\pi(s))\) — <b>every
\(k\) has the same value and there is no basis whatsoever for choosing</b>. (Same proof as Theorem D.)</p>
<p>What then breaks the tie in reality? One identity settles it:</p>
\[ Q_k(s,a)-Q_H(s,a)\ =\ \gamma^k\,\mathbb E\Big[\underbrace{V^{\pi,\kappa}(s_k)}_{\text{deciding afresh now}}-\underbrace{W_a(s_k)}_{\text{continuing the tail already committed}}\Big] \]
<p>So the only thing that breaks the tie is <b>how much better a fresh decision is than the tail you already
committed to</b>. There are exactly three sources.</p>
<div class='tblwrap'><table>
<tr><th></th><th>Source</th><th>As the policy improves?</th></tr>
<tr><td><b>(1)</b></td><td><b>epistemic</b> — the committed tail is worse than what would be proposed now</td><td><b>vanishes</b> (Prop. J: they coincide at \(\pi^\star_H\))</td></tr>
<tr><td><b>(2)</b></td><td><b>aleatoric</b> — \(s_k\) is random and was unseen when the tail was fixed</td><td>remains (total ≤ \(\Delta_{\mathrm{react}}\))</td></tr>
<tr><td><b>(3)</b></td><td><b>partial observability</b> — the information was not in the observation</td><td>remains</td></tr>
</table></div>
<h4>Proposition L — and why this is dangerous: the selector degenerates at the endpoint</h4>
<p>Let the <b>signal</b> actually available to the selector be
\(\eta(\pi):=\sup_s\big[\max_kQ_k(s,\pi(s))-Q_H(s,\pi(s))\big]\). By the table, improving the policy removes (1),
so <b>\(\eta(\pi_n)\) shrinks until only (2)+(3) remain, and goes to 0 under determinism and full
observability.</b></p>
<p>But the critic's \(k\)-dependent bias \(b_k\) (§11) <b>does not shrink</b> with policy quality — it comes from
hindsight leakage in the data, not from the policy. Therefore:</p>
<div class='callout warn'><span class='k'>Proposition L</span>
As soon as \(|b_k-b_H|>\eta(\pi_n)\), the <b>empirical argmax is determined by the bias, not by value</b>. And
\(\eta(\pi_n)\) shrinks <b>precisely as we move toward our goal</b>, so a selector that does not handle the bias
<b>converges to pure noise exactly at the point we are aiming for.</b></div>
<p>This immediately justifies two design choices. ① <b>The lexicographic tolerance \(\epsilon\) is not
cosmetic</b> — set above the bias scale, it makes the selector <b>refuse to guess</b> where the signal is drowned,
committing long by the tie rule. ② To set that \(\epsilon\) we <b>must measure \(b_k\)</b> — which is why test 1
of §11 is mandatory rather than optional.</p>
"""

DERIV_EN = r"""
<h4>Where this curve comes from (the derivation)</h4>
<p>The figure is not a free-hand exponential; it is three theorems multiplied together.</p>
<ol>
<li><b>Contraction rate</b>: by Theorem A the improvement is policy iteration in \(M_H\), which is no slower than
value iteration there; VI in \(M_H\) contracts by \(\gamma^H\) per sweep, hence
\[ \Delta_{\mathrm{epis}}(n)\ \le\ \gamma^{Hn}\,\Delta_{\mathrm{epis}}(0) \]</li>
<li><b>What breaking can still buy</b>: by the decomposition in Part III.5,
\[ B(n)\ :=\ \Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(n)\ \le\ \Delta_{\mathrm{alea}}+\gamma^{Hn}\Delta_{\mathrm{epis}}(0) \]</li>
<li><b>The floor's value</b>: DQC Cor. 1 + Prop. 4 give \(\Delta_{\mathrm{alea}}\le\varepsilon_HH\bar H\) with
\(\varepsilon_H=3(1-(1-\varepsilon)^{H-1})\).</li>
</ol>
<p>With the lexicographic rule (tolerance \(\epsilon\)), once \(B(n)\le\epsilon\) <b>no state retains a reason to
break</b>, so full commitment is selected. The crossing has a closed form:</p>
\[ n^\star\ =\ \Big\lceil \frac{\log\big((\epsilon-\Delta_{\mathrm{alea}})/\Delta_{\mathrm{epis}}(0)\big)}{H\log\gamma}\Big\rceil
\qquad(\text{when }\Delta_{\mathrm{alea}}<\epsilon) \]
<p>For the figure's \(\gamma=0.99,\ H=10,\ \varepsilon=10^{-4}\), the floor \(\le 2.82\), the rate
\(\gamma^H=0.904\) and the crossing \(n^\star=29\) are all <b>computed</b>, not chosen. <b>Honest caveat</b>: the
derivation is about an <b>upper bound</b>, so it guarantees <i>when</i> full commitment must take over but does not
derive the <b>smooth</b> shape of the mean length before that point — that depends on the per-state advantage
distribution, which is what we will measure.</p>
"""

CONV_EN = r"""
<h4>Theorem E — why the commitment length <b>must</b> grow and converge in finite time</h4>
<p>So far we only had "the reason to break shrinks". Now we show it <b>must</b> lengthen and then stop. Three
pieces.</p>
<ol>
<li><b>The pressure is one-directional.</b> Policy iteration in Theorem A is <b>monotone</b>:
\(V^{\pi_{n+1},H}\ge V^{\pi_n,H}\). Hence \(\Delta_{\mathrm{epis}}(n)\) is <b>monotonically decreasing</b>, and so
is the upper bound on what breaking can buy, \(B(n)=\Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(n)\). The force
keeping commitments short only ever weakens.</li>
<li><b>It weakens everywhere at once.</b> For every \(s\),
\(A(s;\pi_n):=\max_kQ_k(s,\pi_n(s))-Q_H(s,\pi_n(s))\le B(n)\) (Part III.5), and \(B(n)\) does not depend on
\(s\) — <b>no state can hold a stronger reason to break than \(B(n)\)</b>.</li>
<li><b>The tolerance wins in finite time.</b> The lexicographic rule (tolerance \(\epsilon\)) treats
\(A(s)\le\epsilon\) as a tie and takes the <b>longest</b> \(k\). Since \(B(n)\downarrow\Delta_{\mathrm{alea}}\)
geometrically, if \(\epsilon>\Delta_{\mathrm{alea}}\) then after a finite \(n^\star\) we have
\(B(n)\le\epsilon\) and therefore <b>\(\kappa_n\equiv H\) at every state</b>.</li>
</ol>
<p><b>Theorem E.</b> If \(\epsilon>\Delta_{\mathrm{alea}}\), then for all
\(n\ge n^\star=\big\lceil\log\!\big((\epsilon-\Delta_{\mathrm{alea}})/\Delta_{\mathrm{epis}}(0)\big)/(H\log\gamma)\big\rceil\)
we have \(\kappa_n\equiv H\). That is, <b>the mean commitment length reaches \(H\) in finite time and stays
there.</b></p>
<p><b>Theorem E′ (when the floor exceeds the tolerance).</b> If \(\epsilon\le\Delta_{\mathrm{alea}}\), then
instead</p>
\[ \limsup_{n\to\infty}\ \{s:\kappa_n(s)<H\}\ \subseteq\ \{s:\ A_{\mathrm{react}}(s)>\epsilon\} \]
<p>i.e. <b>the states that keep breaking are exactly those with genuine reactivity demand</b>. That set is the
precise identity of the aleatoric floor, and its size is the <b>per-skill intrinsic reactivity demand</b> we set
out to measure.</p>
<div class='callout'><span class='k'>Why "must"</span>
The only force holding the length short is \(A(s)>\epsilon\), whose <b>upper bound decreases monotonically</b> to
the <b>unrecoverable</b> part. In other words, <b>every reason that can disappear does, and the reasons that
remain were legitimate to begin with</b>. So the length has to grow, and it stops exactly where legitimate reasons
remain.</div>
<p class='sub'><b>Honest caveat.</b> The argument is about an <b>upper bound</b>, so it guarantees "full
commitment in finite time" and "only the floor set survives in the limit", but it does not exclude an individual
state's length dipping transiently (per-state monotonicity of \(A(s;\pi_n)\) need not hold). What we report as a
figure is the <b>mean</b> length, which is what the argument covers, together with the endpoint.</p>
"""
