"""EN body for the full chunking-theory report."""

EN = r"""
<p class='sub'>This entry unpacks the <b>mathematics</b> of action-chunking RL from scratch, then builds our own
contribution on top of it <b>as theorems with proofs</b>. Three papers: <b>QC</b> (Li, Zhou, Levine,
arXiv:2507.07969, NeurIPS'25) proved only the <i>benefit</i> of chunked critics; <b>DQC</b> (Li, Park, Levine,
arXiv:2512.10926) was the first to quantify the <i>cost</i>; <b>AQC</b> (Gireesh, Ju, Wang, arXiv:2605.05544)
generalised it to state-dependent re-querying. Every definition, theorem and proof mechanism below was checked
against the source PDFs, and our own inferences are marked as such. This is not a new experiment — it is a close
reading plus our formulation.</p>

<p>Notation: \(H:=\frac{1}{1-\gamma}\) (1-step effective horizon), \(\bar H:=\frac{1}{1-\gamma^h}\) (h-step),
\(a_{t:t+h}:=(a_t,\dots,a_{t+h-1})\), \(R_{t:t+h}:=\sum_{t'=t}^{t+h-1}\gamma^{t'-t}r(s_{t'},a_{t'})\).
Rewards lie in \([0,1]\), so values lie in \([0,H]\).</p>

<h3>Part 0. QC — only the benefit is proved</h3>
<p>QC leaves the MDP alone and changes only the <b>signatures</b>: \(\pi(a_{t:t+h}\mid s_t)\),
\(Q(s_t,a_{t:t+h})\). The backup is a single h-step return with no intermediate bootstrapping.</p>
\[ Q(s_t,a_{t:t+h})\ \leftarrow\ \sum_{j=0}^{h-1}\gamma^j r_{t+j}\ +\ \gamma^h Q(s_{t+h},a_{t+h:t+2h}) \]
<p>QC's single theorem (Proposition A.1) is that this backup is <b>unbiased</b>. The three-line proof is the
tower property, but the mechanism matters: <b>the estimand is redefined to be the chunk-conditional \(Q\), so
the off-policy bias is definitionally absent</b> rather than corrected. That buys h× faster value propagation
for free. Yet <b>QC contains no theorem about the cost</b>. Observing the collapse at \(h=50\) (0% success), §5.4
says only:</p>
<blockquote>"We suspect that overly large chunk sizes either hurt policy reactivity too much or make policy
learning too difficult, as the network must predict a much longer action sequence at once." (QC §5.4)</blockquote>
<p>Two causes (loss of reactivity vs. harder policy learning) left unseparated. <b>Separating them is where DQC
begins.</b></p>

<h3>Part I. DQC — how to measure the cost of chunking</h3>

<h4>I.1 The root: nominal ≠ actual</h4>
<div class='tblwrap'><table>
<tr><th>Symbol</th><th>Name</th><th>Meaning</th></tr>
<tr><td>\(\hat V_{ac}\)</td><td><b>nominal</b></td><td>the fixed point of chunked TD on the data — <b>what we train</b></td></tr>
<tr><td>\(V_{ac}\)</td><td><b>actual</b></td><td>the value of that chunk policy when actually rolled out <b>open-loop</b></td></tr>
</table></div>
<p><b>Definition 1 (open-loop trajectory).</b> Rolling out the data's chunk marginal
\(\pi^\circ_D(a_{t:t+h}\mid s_t):=P_D(a_{t:t+h}\mid s_t)\) open-loop gives</p>
\[ P^\circ_D(s_{t+1:t+h+1},a_{t:t+h}\mid s_t)\ :=\ \pi^\circ_D(a_{t:t+h}\mid s_t)\prod_{k=0}^{h-1}T(s_{t+k+1}\mid s_{t+k},a_{t+k}) \]
<p>which generally <b>differs</b> from \(P_D\). The reason is decisive: <b>the policy that produced the data was
closed-loop</b>. A human or a script chose \(a_{t+1}\) <b>after seeing</b> \(s_{t+1}\). Conditioning on a whole
chunk therefore <b>leaks the stochastic outcome</b> into the conditional. Assumption 1 only requires that the
data obey the true dynamics (the behaviour policy may be non-Markovian), so the analysis is about <b>pure
open-loop replay bias, not model error</b>.</p>

<h4>I.2 Definition 2 — Open-Loop Consistency</h4>
<p><b>weak</b> \(\varepsilon_h\)-OLC, for every \(s_t\in\mathrm{supp}(P_D)\):</p>
\[ D_{TV}\!\big(P^\circ_D(s_{t+h'},a_{t+h'}\mid s_t)\,\big\|\,P_D(s_{t+h'},a_{t+h'}\mid s_t)\big)\le\varepsilon_h,\quad h'=1..h-1 \]
\[ D_{TV}\!\big(P^\circ_D(s_{t+h}\mid s_t)\,\big\|\,P_D(s_{t+h}\mid s_t)\big)\le\varepsilon_h \]
<p><b>strong</b> \(\varepsilon_h\)-OLC additionally requires, uniformly over <b>every individual chunk</b> in
support:</p>
\[ D_{TV}\!\big(T(s_{t+h'}\mid s_t,a_{t:t+h'})\,\big\|\,P_D(s_{t+h'}\mid s_t,a_{t:t+h})\big)\le\varepsilon_h,\quad h'=1..h \]
<p>Weak says "the replay <b>marginal</b> is right"; strong says "<b>whichever chunk you pick</b>, the data's
conditional next-state law matches open-loop rollout". The latter effectively says <b>the chunk is (nearly)
independent of the intermediate states</b>, which is what removes the confounding. Sections I.4 and I.6 turn on
exactly this difference.</p>

<h4>I.3 Lemma 2 — the workhorse behind every bound</h4>
<p>For \(P,Q\in\Delta_X\), \(f,g:X\to[0,1]\) with \(D_{TV}(P,Q)\le\varepsilon\) and
\(\|f-g\|_\infty\le\delta\) on \(\mathrm{supp}(P)\cap\mathrm{supp}(Q)\):</p>
\[ \big|\,\mathbb E_{P}[f]-\mathbb E_{Q}[g]\,\big|\ \le\ (1-\varepsilon)\delta+\varepsilon \]
<p>The proof decomposes the mass as \(P=d_P+d_{PQ}\), \(Q=d_{PQ}+d_Q\) with
\(\int d_P=\int d_Q=\hat\varepsilon\le\varepsilon\); the disjoint part is charged at the full value range and the
shared part at \(\|f-g\|_\infty(1-\hat\varepsilon)\). <b>Applied to value functions ranging over \([0,H]\), the
\(\varepsilon\) term becomes \(\varepsilon/(1-\gamma)\)</b> — this is the origin of every \(H\) factor below.</p>

<h4>I.4 Theorem 1 (AC Value Bias) — where \(\varepsilon_h H\bar H\) comes from</h4>
<p>If \(\hat V_{ac}\) solves the behaviour chunk backup and \(V_{ac}\) is the true value of
\(\tilde\pi_{ac}:s_t\mapsto P_D(a_{t:t+h}\mid s_t)\), then under <b>weak</b> \(\varepsilon_h\)-OLC:</p>
\[ \big|V_{ac}(s_t)-\hat V_{ac}(s_t)\big|\ \le\ \frac{\gamma\varepsilon_h}{(1-\gamma)\big(1-(1-\varepsilon_h)\gamma^h\big)}\ \le\ \varepsilon_h H\bar H \]
<p><b>Proof machinery.</b> One backup produces two error sources.</p>
<ol>
<li><b>Reward term</b>: apply Lemma 2 at each intermediate step with \(f=g=r\in[0,1]\) (so \(\delta=0\)) →
\(\gamma^{h'}\varepsilon_h\), summing to \(\sum_{h'=1}^{h-1}\gamma^{h'}\varepsilon_h\).</li>
<li><b>Bootstrap term</b>: apply Lemma 2 with \(f=\hat V_{ac},g=V_{ac}\in[0,H]\) →
\(\gamma^h\big[\varepsilon_h\cdot\frac{1}{1-\gamma}+(1-\varepsilon_h)\sup|\hat V_{ac}-V_{ac}|\big]\).</li>
</ol>
<p>This is the heart: <b>the \(\varepsilon_h\) mass on which the distributions disagree escapes immediately to
the maximal error \(1/(1-\gamma)\), and only \((1-\varepsilon_h)\) recurses.</b> Hence the contraction factor is
\(\boldsymbol{(1-\varepsilon_h)\gamma^h}\), not \(\gamma^h\), and unrolling gives the bound. In short:
<b>\(\bar H\) counts chunk-level backups; \(H\) is how far an error jumps when it escapes.</b></p>

<h4>I.5 Theorem 2 — the bound is tight (a 2h-state counterexample)</h4>
<p>For any \(h>1,\gamma,\varepsilon_h\in[0,1/2]\) there is an MDP attaining the bound <b>exactly</b>, in
<b>both</b> directions (over- and under-estimation). Construction (Figure 8): choose \(\delta\) with
\(\varepsilon_h=2\delta(1-\delta)\); from each state the chain branches <b>independently of the action</b> to a
"tilde" branch w.p. \(\delta\). Rewards are \(r(\tilde X_i,a{=}0)=r(X_i,a{=}1)=1\) and 0 otherwise — i.e. the
<b>correct action is state-dependent</b>. Data comes from the optimal closed-loop policy, so in the data action
and state are perfectly correlated:</p>
\[ P_D=\begin{pmatrix}\delta&0\\0&1-\delta\end{pmatrix},\qquad
P^\circ_D=\begin{pmatrix}\delta^2&(1-\delta)\delta\\ \delta(1-\delta)&(1-\delta)^2\end{pmatrix} \]
<p>Open-loop replay <b>destroys</b> that correlation (the chunk is drawn from the marginal and the environment
re-flips its coin), giving \(D_{TV}=2\delta(1-\delta)=\varepsilon_h\) at every step, and at step \(h\) exactly
\(\varepsilon_h\) of the mass leaks into the absorbing state. <b>This "the chunk cannot react to the coin flip"
gadget is the skeleton of every later counterexample.</b></p>

<h4>I.6 Corollary 1 — turning a bias bound into a suboptimality bound</h4>
<p>Let \(D^\star\) come from an <b>optimal</b> policy. Then</p>
\[ V^\star(s_t)=\mathbb E_{P_{D^\star}}\big[R_{t:t+h}+\gamma^hV^\star(s_{t+h})\big] \]
<p>which is exactly the fixed-point equation for \(\hat V_{ac}\). Hence <b>\(\hat V_{ac}=V^\star\)</b>, and
Theorem 1 becomes, <b>with no new proof</b>, a bound on the optimality gap:</p>
\[ V^\star(s_t)-V^\star_{ac}(s_t)\ \le\ V^\star(s_t)-\tilde V_{ac}(s_t)\ \le\ \frac{\gamma\varepsilon_h}{(1-\gamma)(1-(1-\varepsilon_h)\gamma^h)}\ \le\ \varepsilon_h H\bar H \]
<p>where \(V^\star_{ac}\) is the <b>true</b> value of the optimal chunk policy. Corollary 2 proves tightness with
a \((3h-1)\)-state MDP in which the <b>optimal chunk policy itself</b> — not merely the cloned one — is stuck,
because no chunk can react to which branch the coin selects. <b>This is the only quantification of "the price of
open-loop commitment" in this literature.</b></p>

<h4>I.7 What that \(\varepsilon_h\) is — Proposition 4</h4>
<p>If \(T\) is \(\varepsilon\)-deterministic, i.e.
\(T(s'\mid s,a)=(1-\varepsilon)\delta_{f(s,a)}(s')+\varepsilon\tilde T(s'\mid s,a)\), then <b>any</b> data from
that MDP is weakly \(\varepsilon_h\)-OLC with</p>
\[ \varepsilon_h\ =\ 3\big(1-(1-\varepsilon)^{h-1}\big) \]
<blockquote>"This bounded stochasticity allows the results of taking an action sequence (of length h) open-loop
to be deterministically determined in the event that the deterministic dynamics is 'triggered' (with a joint
\((1-\varepsilon)^{h-1}\) probability across h time steps). It is clear that under such event, there is no gap
between the 'replayed' open-loop data \(P^\circ_D\) and the original data distribution \(P_D\)." (DQC §E.1)</blockquote>
<p>The proof introduces the indicator \(I=1\{\text{all }h{-}1\text{ steps take the deterministic branch}\}\) and
pushes the distributions through the deterministic map with <b>Lemma 4 (data processing inequality)</b>, chaining
three TV legs — the constant 3 is <b>the union of those three legs</b>.</p>
<div class='callout'><span class='k'>Therefore (our composition; not one display in the paper)</span>
\(V^\star_1-V^\star_H\lesssim 3(H{-}1)\varepsilon\cdot H\bar H\) for small \(\varepsilon\): <b>under deterministic
dynamics the price of open-loop commitment is exactly zero.</b> This becomes the axis of the decomposition in
Part III.</div>

<h4>I.8 Proposition 1 — under weak OLC, Q-learning breaks by \(\Omega(H)\)</h4>
<p>There is an MDP and weakly \(\varepsilon_h\)-OLC data with
\(V^\star(s_t)-V^+_{ac}(s_t)=\gamma c/(1-\gamma)=\Omega(H)\) (with \(c\) up to \(1/2\) — a constant fraction of
the whole value range). The mechanism of the 6-state counterexample: the behaviour policy is <b>closed-loop</b>,
\(\pi_D(B)=0,\pi_D(C)=1\), so <b>the second action reveals the outcome of the first transition</b>. Conditioning
on the chunk \((0,0)\) therefore means "\(s_1\) was \(B\)", so</p>
\[ P_D(s_2=D\mid A,(0,0))=1\ \text{(reward 1)},\qquad\text{yet executed open-loop, }P(D)=\delta \]
<p>The nominal \(\hat Q^+_{ac}(A,(0,0))=\gamma/(1-\gamma)\) is maximal so \(\pi^+_{ac}\) picks it, but executing it
lands in \(C\) w.p. \(1-\delta\) and then \(a=0\) falls into the absorbing zero-reward state.</p>
<blockquote>"the chunked critic \(Q(s_t,a_{t:t+h})\) has no way of differentiating a low-probability, 'lucky'
success from a closed-loop, high-probability success. This can cause the learned policy \(\pi^+_{ac}\) to
erroneously prefer very low-value action chunks even when the optimal action chunks are available in the data
distribution." (DQC §4.4)</blockquote>
<div class='callout warn'><span class='k'>Directly relevant to us</span>
Our yam and RoboCasa data are human teleoperation — <b>fully closed-loop</b>. So Prop. 1's pathology is
<b>structurally present</b> in our data. We call it <b>hindsight leakage</b> and quantify it as a
\(k\)-dependent bias in Part III.</div>

<h4>I.9 Theorem 3 — strong OLC restores the guarantee, and where the \(2+1\) comes from</h4>
<p>If \(D\) and \(D^\star\) are strongly \(\varepsilon_h\)-OLC with support containment:</p>
\[ V^\star(s_t)-V^+_{ac}(s_t)\ \le\ \frac{\varepsilon_h\gamma}{1-\gamma}\left[\frac{2}{1-(1-2\varepsilon_h)\gamma^h}+\frac{1}{1-(1-\varepsilon_h)\gamma^h}\right]\ \le\ 3\varepsilon_hH\bar H \]
<p><b>Where the 2 comes from</b>: strong OLC gives \(\varepsilon_h\) for \(D\to T\) and again for
\(T\to D^\star\), so the triangle property of TV yields \(D_{TV}(P_D\|P_{D^\star})\le 2\varepsilon_h\) — you pay
once going from data to true dynamics and once coming back to optimal data. <b>The decisive cancellation</b> is
the last recursion step: \(\mathbb E_{P_{D^\star}}[\hat Q^+_{ac}]-\sup_a\hat Q^+_{ac}\le 0\), because
\(\pi^+_{ac}\) maximises over a <b>superset</b> (this is where support containment is used). The final \(+1\)
term is Theorem 1 applied to \(\pi^+_{ac}\): the <b>nominal↔actual replay bias</b>. Crucially, <b>this bound is
independent of how suboptimal the data is</b>. Theorem 4 shows the 3 is necessary, and the paper itself names the
three pieces:</p>
<blockquote>"(1) the optimal action chunking policy is \((\varepsilon_hH^2)\)-sub-optimal due to its inability to
react to environment stochasticity … (2) the value <b>under-estimation</b> bias can incur another factor … (3)
the action chunking value function may prefer an <b>overestimated</b> action chunking policy \(\pi^+_{ac}\) where
its actual value is again \(\varepsilon_hH\bar H\) from its estimated value, resulting in a total sub-optimality
of \(3\varepsilon_hH\bar H\)." (DQC §4.4)</blockquote>

<h4>I.10 Lemma 8 — "committing longer is weakly worse"</h4>
<p>For <b>any</b> open-loop data distribution \(D^\circ\):</p>
\[ V^\star(s_t)\ \ge\ \mathbb E_{T}\big[r_t+\gamma V^\star(s_{t+1})\big]\ \ge\ \mathbb E_{P_{D^\circ}(\cdot\mid s_t,a_{t:t+h})}\big[R_{t:t+h}+\gamma^hV^\star(s_{t+h})\big] \]
<p>The proof is an induction: each additional committed open-loop step replaces a \(\max_{a'}Q^\star\) by a
<b>fixed action</b>, weakly decreasing the value. This is DQC's <b>monotonicity-in-commitment-length</b> lemma and
the general form of the "the first action of a good chunk cannot be too bad" argument used in Prop. 3 and Thm. 7.</p>

<h4>I.11 Proposition 3 vs Theorems 5/6 — closed-loop is not free either</h4>
<p>For \(\pi^\bullet\) (execute only the first action, re-query every step), under strong OLC:</p>
\[ V^\star(s_t)-V^\bullet(s_t)\ \le\ \frac{\varepsilon_h\gamma}{(1-\gamma)^2}\Big[\cdots\Big]\ \le\ 3\varepsilon_hH^2\bar H \]
<p>i.e. <b>a factor \(H\) worse</b> than open-loop execution (\(3\varepsilon_hH\bar H\)). Step 1 of its proof is a
special case of Lemma 8; step 3's standard performance-difference recursion adds the extra \(1/(1-\gamma)\). But
reading this as "short commitments are harmful" is wrong. DQC immediately asks "Can we do better than this?" and
introduces <b>Definition 4 (bounded optimality variability, BOV)</b>, under which <b>Theorem 5</b> gives</p>
\[ V^\star-V^\bullet\ \le\ \vartheta^L_hH+2\vartheta^G_hH\bar H \]
<p>with <b>no OLC assumption at all</b>. The elegance of the proof: <b>global BOV bounds under-estimation, local
BOV bounds over-estimation</b>, and taking the \(\min\) of the two <b>shaves a factor \(\bar H\)</b> off the
\(\vartheta^L\) term. And <b>Theorem 6</b> establishes the opposite direction: there exist MDPs (a "castle" and a
"flower" gadget composed) where closed-loop execution is near-optimal while <b>the same policy executed in chunks
is \(\Omega(H)\) suboptimal</b>.</p>
<div class='callout'><span class='k'>Honest conclusion</span>
<b>Neither open-loop nor closed-loop dominates in general — it depends on the MDP/data structure (OLC/BOV).</b>
Table 1 summarises: weak OLC guarantees only <i>value estimation</i> and not the policy (Prop. 1); BOV guarantees
only <i>closed-loop</i> and not the open-loop chunk policy (Thm. 6). <b>Precisely this makes state-adaptive
\(k\) principled</b>: any fixed \(k\) necessarily forfeits one side.</div>

<h4>I.12 Lemma 7 / Theorem 7 — optimism is the value of "stochastic shortcuts"</h4>
<p><b>Definition 7</b>: \(M\) is free of \(\vartheta_h\)-stochastic shortcuts if
\(\gamma^hV^\star(s_{t+h})+R_{t:t+h}-V^\star(s_t)\le\vartheta_h\) along every positive-probability path.</p>
<blockquote>"stochastic shortcuts are low-probability (but plausible) paths … that lead to returns that are much
higher than the optimal expected value. These … are particularly problematic for action chunking value backup
because the chunked critic cannot distinguish between a low-probability stochastic shortcut and an optimal
closed-loop trajectory, leading it to erroneously favor the shortcut." (DQC §E.3)</blockquote>
<p><b>Lemma 7</b>: absent such shortcuts, the overestimation is bounded by
\(\hat V^+_{ac}(s_t)-V^\star(s_t)\le\vartheta_h/(1-\gamma^h)\). That is: <b>the size of the optimism = the value
of the stochastic shortcuts the critic cannot distinguish from control</b>. Theorem 7 then shows that with
neither OLC nor BOV — only \(\alpha\)-open-loop-mixed data — closed-loop execution is near-optimal. The technical
heart is <b>Lemma 1 (a mean-value theorem for conditional probabilities)</b>, which converts an <i>average</i>
mixing bound into a <i>pointwise</i> one at some specific chunk.</p>

<h4>I.13 Proposition 2 — when to prefer chunking over n-step</h4>
\[ V^+_{ac}(s_t)-\hat V^+_n(s_t)\ \ge\ \delta_n\bar H_n-3\varepsilon_hH\bar H \]
<p>So chunking wins once the data is more than \(3\varepsilon_hH\)-suboptimal. Proposition 5 shows the converse
too: <b>data suboptimality \((\delta_n)\) and open-loop consistency \((\varepsilon_h)\) are independent</b>.</p>

<h4>I.14 The DQC algorithm</h4>
<p>Decouple the critic chunk \(h\) from the <b>policy chunk \(h_a\ll h\)</b>, and build a <b>partial critic</b> by
<b>optimistic (expectile) distillation</b>:</p>
\[ \mathcal L(\psi)=f^{\kappa_d}_{\text{expectile}}\!\big(\bar Q_\phi(s_t,a_{t:t+h})-Q^P_\psi(s_t,a_{t:t+h_a})\big),\qquad
\mathcal L(\pi)=-\mathbb E\big[Q^P_\psi(s_t,a_{t:t+h_a})\big] \]
<p>so \(Q^P\approx\max_{\text{tail}}Q\), <b>optimistic by construction</b>. \(V_\xi\) is fit with the
\(\kappa_b=(N-1)/N\) quantile to match the largest order statistic of best-of-\(N\). The price: deployment runs
<b>short chunks</b>, paying (worst case) I.11's extra \(H\) and giving up QC's temporally coherent exploration.
DQC itself leaves the open problem:</p>
<blockquote>"our method relies on a fixed policy action chunk size \(h_a\) and critic action chunk size \(h\)
across all states, even though the optimal action chunk size may vary by state. Developing practical methods that
can support flexible, state-dependent chunk sizes would be a natural next step." (DQC §8)</blockquote>
<p>AQC is the paper that answers exactly this.</p>

<h4>I.15 Monotonicity in \(h\) — assembling the scattered pieces</h4>
<p>Every bound carries \(\bar H=1/(1-\gamma^h)\), which <b>decreases</b> in \(h\) (chunking shortens the effective
horizon). But \(\varepsilon_h\approx 3(h-1)\varepsilon\) <b>increases</b>. The product's behaviour is not obvious,
so <b>we checked it numerically</b> on the exact form
\(\gamma\varepsilon_h/[(1-\gamma)(1-(1-\varepsilon_h)\gamma^h)]\):</p>
<div class='tblwrap'><table>
<tr><th>\(\gamma\)</th><th>\(\varepsilon\)</th><th>k=2</th><th>k=5</th><th>k=10</th><th>k=30</th><th>k=50</th><th>monotone↑?</th></tr>
<tr><td>0.99</td><td>0.001</td><td>13.0</td><td>19.6</td><td>22.2</td><td>26.2</td><td>29.5</td><td>yes</td></tr>
<tr><td>0.999</td><td>0.001</td><td>600.2</td><td>707.8</td><td>734.4</td><td>759.6</td><td>773.7</td><td>yes</td></tr>
<tr><td>0.99964</td><td>0.001</td><td>2240.7</td><td>2418.1</td><td>2457.2</td><td>2491.5</td><td>2509.6</td><td>yes</td></tr>
<tr><td>0.99964</td><td>0.010</td><td>2713.6</td><td>2740.0</td><td>2748.6</td><td>2767.3</td><td>2776.8</td><td>yes</td></tr>
</table></div>
<p><b>Our conclusion</b>: the bias bound is <b>monotonically increasing</b> in \(k\) — but it <b>saturates very
fast</b> (at \(\gamma=0.99964\) it is already 2241 at \(k=2\) and only 2510 at \(k=50\), against a ceiling of
2778). So the bound <b>establishes the direction but is quantitatively vacuous at large \(k\)</b>, which is why
the actual bias must be measured (pre-registered in Part III).</p>

<h3>Part II. AQC — AOLC, the selector, and what we verified</h3>

<h4>II.1 Definition H.2 — Adaptive Open-Loop Consistency</h4>
<p>DQC's OLC presumes replay at a <b>fixed</b> length \(h\). AQC inserts a selection function
\(\kappa:\mathcal S\to\mathcal K\):</p>
\[ D_{TV}\!\big(P_D(s_{t+\kappa(s_t)},a_{t+\kappa(s_t)}\mid s_t)\ \big\|\ P^\circ_{D,\kappa}(\cdot\mid s_t)\big)\le\varepsilon_{\mathcal K} \]
<p>The stated point is that <b>re-query points become state-dependent, hence randomly spaced, so the TV bound must
hold uniformly over the distribution of re-query times induced by \(\kappa\)</b>. <b>Proposition H.3</b>: for
constant \(\kappa\equiv k\) this reduces exactly to DQC's weak OLC.</p>
<div class='callout warn'><span class='k'>Honest assessment</span>
As a definition this is DQC's two TV constraints with the fixed offset \(t+h\) <b>substituted</b> by
\(t+\kappa(s_t)\). No new machinery appears (no joint condition over a <i>sequence</i> of re-query points, no
stopping-time measure, no composition across decision epochs). Two things are left open: ① the definition
constrains a <b>single</b> adaptive step from \(s_t\), while execution composes many (per-step TV does not
automatically compose to a \(T\)-step bound); ② \(\kappa\) is a <b>given fixed function</b> in the definition, but
the deployed selector depends on the critic and changes throughout training. Theorem H.14 in fact assumes AOLC
<b>under the oracle \(k^\dagger\)</b>, which is not the schedule being executed.</div>

<h4>II.2 The selector criterion — why \(-V^k\) and why \(/\gamma^k\)</h4>
<p>The naive rule \(\arg\max_{k,a}Q^k\) fails two ways (§4.2). ① <b>Discount-scale mismatch</b>: with sparse
rewards \(Q^k\approx\gamma^kV^h(s_{t+k})\), hence</p>
<blockquote>"Since \(\gamma<1\), the factor \(\gamma^k\) is strictly decreasing in \(k\). Consequently,
\(Q^{k_1}>Q^{k_2}>\cdots>Q^h\) for nearly every state, regardless of which chunk size actually yields a better
policy. The selector degenerates to always choosing the smallest \(k\) in \(\mathcal K\)." (AQC §4.2)</blockquote>
<p>② <b>State-dependent baseline mismatch</b>: after dividing, the rule becomes \(\arg\max_kV^h(s_{t+k})\), and far
from reward all \(V^h\) are small, so <b>differences across \(k\) are dominated by approximation noise</b>. Hence</p>
\[ \mathrm{score}(k,a_{t:t+k})\ :=\ \frac{Q^k(s_t,a_{t:t+k})-V^k(s_t)}{\gamma^k} \]
<p><b>Proposition 5.1 (noise immunity)</b>: where there is no signal, \(|\delta_k|\le\epsilon+2\sigma\), so every
\(k\) scores near zero — "a biased wrong answer becomes an unbiased near-random one".</p>
<div class='callout warn'><span class='k'>A sign problem we verified against the source</span>
Argument ① presumes \(V^h>0\). But <b>AQC's own benchmark rewards are negative</b> — §C.1: "receives −1 when the
task is incomplete and 0 upon completion". Then \(V^h\le0\), so \(\gamma^kV^h\) is <b>increasing</b> in \(k\)
(less negative) and the naive selector should collapse to the <b>longest</b> chunk — the opposite of the reported
"raw-Q variant collapses to always selecting \(k=1\)". For our own <b>cost_to_goal (\(r=-1\)) the same reasoning
makes \(/\gamma^k\) prefer short</b>. <b>The normalisation is reward-convention dependent and must not be ported
to our setting.</b></div>

<h4>II.3 Definition H.4 / Theorem H.5 — circularity</h4>
<p>Put the oracle and the selector side by side:</p>
\[ k^\dagger(s)\in\arg\max_k\max_a\frac{Q^{k,*}(s,a)-V^{k,*}(s)}{\gamma^k},\qquad
\hat k(s)\in\arg\max_k\max_a\frac{Q^{k}(s,a)-V^{k}(s)}{\gamma^k} \]
<p>They are the <b>identical functional</b>, evaluated at starred versus estimated quantities. So Theorem H.5
("selector soundness", under \(\bar\varepsilon<\Delta\gamma^{k_{\min}}/2\)) is a <b>plug-in consistency lemma</b>
— "the estimator recovers its own population argmax" — <b>not</b> a claim that the criterion maximises return.
The proof itself (a three-line triangle inequality) is correct.</p>
<div class='callout warn'><span class='k'>Two problems we verified against the source</span>
① <b>\(V^{k,*}\) is never defined</b> (8 uses, no definition). Under the natural "optimal" reading
\(V^{k,*}=\max_aQ^{k,*}\) we get \(\max_aA^{k,*}\equiv0\), hence <b>\(\Delta(s)\equiv0\)</b>, making global
\(\Delta\)-separability with \(\Delta>0\) <b>unsatisfiable</b> and H.5/H.8 vacuous. Under the alternative
"behaviour baseline" reading used in §5.1, \(A^{k,*}\) can be positive but the \(*\) superscript is inconsistent
with \(Q^{k,*}\).
② <b>The "mis-selection probability" puts a sup-norm into Markov's inequality</b>: "By Markov's inequality applied
to the estimation error: \(P(\hat k(s)\ne k^\dagger(s))\le P(|\hat A-A^{k^\dagger,*}|\ge\Delta(s)/2)\le
2\bar\varepsilon/(\gamma^{k_{\min}}\Delta(s))\)." But \(\bar\varepsilon\) is a <b>deterministic worst case</b>
(\(\|\cdot\|_\infty\)), not a mean, and there is no probability space (the errors are deterministic). Moreover in
the regime \(\bar\varepsilon<\Delta\gamma^{k_{\min}}/2\) H.5 already proves mis-selection is <b>zero</b>, and
outside it the "probability" is \(\ge1\) and vacuous. Six downstream results depend on this term.</div>

<h4>II.4 Theorem H.8 — meta-MDP dominance, and a sign error we verified</h4>
\[ V^{AQC}(s)-V^k(s)\ \ge\ \frac{\gamma^{k_{\min}}\big(1-\tfrac{2\bar\varepsilon}{\gamma^{k_{\min}}\Delta}\big)}{1-\gamma}\,
\mathbb E_{s'\sim d^{AQC}}\big[\bar A^{k^\dagger,*}(s')-\bar A^{k,*}(s')\big] \]
<p>The idea is good: build a <b>meta-MDP whose actions are pairs \((k,a_{t:t+k})\) and whose transitions execute
\(k\) steps open-loop</b>, then apply the performance difference lemma. But step 3 (discount normalisation) reads:</p>
<blockquote>"Since \(\gamma^{k^*(s')}\ge\gamma^{k_{\min}}\) for all \(s'\) and \(\gamma^k\le\gamma^{k_{\min}}\),
and noting that \(\bar A^{k,*}(s')\ge0\) (as the max advantage is non-negative at the behavior policy's best
action), we lower-bound: \(\gamma^{k^*(s')}\bar A^{k^\dagger,*}(s')-\gamma^k\bar A^{k,*}(s')\ge
\gamma^{k_{\min}}(\bar A^{k^\dagger,*}(s')-\bar A^{k,*}(s'))\)." (AQC Appendix I.4, before Eq. 65)</blockquote>
<div class='callout warn'><span class='k'>What we checked directly</span>
Since \(k^*(s')\in\mathcal K\) and \(k_{\min}=\min\mathcal K\), we have \(k^*(s')\ge k_{\min}\), hence for
\(\gamma<1\), <b>\(\gamma^{k^*(s')}\le\gamma^{k_{\min}}\)</b>. The second inequality in the same sentence
(\(\gamma^k\le\gamma^{k_{\min}}\)) is correct; <b>the first is reversed</b>. The sign condition \(\bar A\ge0\)
does rescue the second term (\(-\gamma^k\bar A^k\ge-\gamma^{k_{\min}}\bar A^k\)) but not the first. That sign
condition, moreover, <b>is absent from the theorem's hypotheses</b> and appears only in a parenthetical; and under
the "optimal" reading of \(V^{k,*}\) it holds with \(\bar A\equiv0\), making the theorem's RHS zero. Repairing the
sign gives \(\gamma^{k_{\max}}\bar A^{k^\dagger}-\gamma^{k_{\min}}\bar A^{k}\), which <b>can be negative exactly
when \(k^\dagger>k\)</b> — i.e. when the adaptive policy prefers a <i>longer</i> chunk, the paper's own headline
behaviour.</div>
<p>Relatedly, <b>Theorem H.14</b> is DQC's Prop. 3 in adaptive form (\(h\to k_{\min}\)); since
\(k_{\min}\le h\) we have \(\bar H_{k_{\min}}\ge\bar H_h\), so the substitution makes the bound <b>larger
(worse)</b>. Describing it as "improves reactivity" runs against the direction of the bound (though it is
consistent with the honest trade-off remark in the same section).</p>

<h4>II.5 The AQC algorithm — and the decisive fact</h4>
<p>Four losses: EMAQ h-step TD for \(Q^h\); expectile for \(V^h\); per-scale \(Q^k\) <b>bootstrapped from
\(\bar V^h\)</b> (no max — a pure regression); expectile for \(V^k\). The policy is flow-matching BC. At inference,
\(N\) candidates are scored by the per-scale advantage, <b>z-scored within each scale</b>, then argmaxed.</p>
<div class='callout'><span class='k'>Decisive fact — there is no policy improvement</span>
AQC's policy is <b>pure behaviour cloning</b>. No actor loss anywhere in the paper contains \(Q\). Algorithm 1
updates "\(\pi_\beta\) via flow-matching BC" in both the offline and online loops. Improvement enters only through
(i) the EMAQ best-of-\(N\) max inside the \(Q^h\) target and (ii) best-of-\(N\) selection at inference. The
RoboCasa experiment is even more explicit — "attach an AQC critic head while freezing all backbone parameters",
trainable: "Critic head only". So in the VLA setting it is <b>critic-only training plus best-of-10 selection over
a frozen BC policy</b>, with no RL policy update at all. AQC's achievable return is therefore capped at
<b>the max over \(N\) samples from a BC policy</b>.</div>

<h3>Part III. Our formulation — filling the empty slot with theorems</h3>

<h4>III.1 Setting and three values</h4>
<p>Fully observed MDP \(M=(\mathcal S,\mathcal A,T,r,\gamma)\), chunk length \(H\). For a chunk policy
\(\pi:\mathcal S\to\Delta(\mathcal A^H)\) and a <b>commitment selector</b> \(\kappa:\mathcal S\to\{1,\dots,H\}\),
write \(V^{\pi,\kappa}\) for the value of "query \(\pi\) at \(s\), execute the first \(\kappa(s)\) actions
open-loop, re-query". Distinguish three optima:</p>
\[ V^\star_1:=\sup_{\text{closed-loop 1-step}}V,\qquad
V^\star_H:=\sup_\pi V^{\pi,H},\qquad
V^\star_{\mathrm{ada}}:=\sup_{\pi,\kappa}V^{\pi,\kappa} \]
<p><b>Lemma A (sandwich).</b> \(V^\star_H\le V^\star_{\mathrm{ada}}\le V^\star_1\).<br>
<i>Proof.</i> Left: take \(\kappa\equiv H\). Right: any \((\pi,\kappa)\) execution is a particular closed-loop
policy measurable w.r.t. the observation filtration, so its value cannot exceed \(V^\star_1\). ∎</p>

<h4>III.2 Theorem 1 (decomposition) — aleatoric and epistemic</h4>
\[ \underbrace{V^\star_1-V^{\pi,H}}_{\text{total loss}}
=\underbrace{\big(V^\star_1-V^\star_H\big)}_{\Delta_{\mathrm{alea}}\ \text{(policy-independent)}}
+\underbrace{\big(V^\star_H-V^{\pi,H}\big)}_{\Delta_{\mathrm{epis}}(\pi)\ \text{(policy-dependent)}} \]
<p>\(\Delta_{\mathrm{alea}}\) is <b>exactly what DQC's Corollary 1 measures</b>, \(\le\varepsilon_HH\bar H\) with
\(\varepsilon_H=3(1-(1-\varepsilon)^{H-1})\) (Prop. 4). \(\Delta_{\mathrm{epis}}(\pi)\) can be driven to zero by
policy improvement, since \(\sup_\pi V^{\pi,H}=V^\star_H\) by definition.</p>

<h4>III.3 Theorem 2 (under determinism, reactivity has zero value) — our key theorem</h4>
<p><b>Theorem.</b> In a fully observed MDP with deterministic \(T\) and \(r\),
\(V^\star_1=V^\star_H=V^\star_{\mathrm{ada}}\).</p>
<p><i>Proof.</i> By determinism, from any \(s\) an optimal policy induces a single trajectory. Define the
open-loop chunk policy \(\tilde\pi(s)\) that outputs the first \(H\) actions of that trajectory (it lies in the
class, since \(\tilde\pi:\mathcal S\to\mathcal A^H\)). Executing \(\tilde\pi\) with <b>full commitment</b> reaches
exactly the same \(s_{t+H}\) (transitions are deterministic), and re-querying there returns the continuation of
the same optimal trajectory by the Markov property and state-dependence of optimality. By induction the
full-commitment execution of \(\tilde\pi\) <b>reproduces the optimal trajectory</b>, so
\(V^{\tilde\pi,H}=V^\star_1\), giving \(V^\star_H\ge V^\star_1\). With Lemma A, all three coincide. ∎</p>
<p><b>Corollary (absorbability = recompose).</b> Under determinism, for any \(\pi,\kappa\):</p>
\[ V^{\pi,\kappa}-V^{\pi,H}\ \le\ V^\star_{\mathrm{ada}}-V^{\pi,H}\ =\ V^\star_H-V^{\pi,H}\ =\ \Delta_{\mathrm{epis}}(\pi) \]
<p>i.e. <b>everything adaptive execution gains is achievable by a better full chunk</b> — policy improvement can
absorb it.</p>
<div class='callout'><span class='k'>What this theorem does</span>
① it <b>re-derives elementarily</b> DQC's "\(\varepsilon=0\Rightarrow\) gap 0" (Cor. 1 + Prop. 4), ② it extends it
to the <b>adaptive class</b>, which DQC does not treat, and ③ it turns our story ("compile the gain discovered by
breaking the chunk into a single full chunk") into <b>a theorem</b>.</div>

<h4>III.4 The reactivity information set — the general form of the floor</h4>
<p>What the proof of Theorem 2 actually uses is not "determinism" but that <b>no information arrives between
\(t\) and \(t+H\) that is unpredictable from \(s_t\)</b>. We take that as the definition.</p>
<p><b>Definition (reactivity information).</b> \(\mathcal I_t^H\) := information arriving in \((t,t+H]\) that is
not \(\sigma(s_t)\)-measurable. If \(\mathcal I_t^H\) is trivial, adaptive execution gains nothing (the argument of
Theorem 2 applies verbatim, since the realised chunk becomes a deterministic function of \(s_t\)).</p>
<p>Hence the floor has <b>two sources</b>: (i) <b>dynamics stochasticity</b> (DQC's \(\varepsilon\)) and
(ii) <b>partial observability</b>. VLAs observe images, so (ii) is real — occlusions, unobserved contact forces.
The floor in our setting is therefore not zero, and <b>its magnitude is itself a new, per-skill measurable
quantity</b>.</p>

<h4>III.5 Theorem 3 (bounding the floor)</h4>
<p>With \(\Delta_{\mathrm{react}}:=V^\star_{\mathrm{ada}}-V^\star_H\), Lemma A and Theorem 1 give</p>
\[ 0\ \le\ \Delta_{\mathrm{react}}\ \le\ \Delta_{\mathrm{alea}}\ \le\ \varepsilon_HH\bar H,\qquad
\varepsilon_H=3\big(1-(1-\varepsilon)^{H-1}\big) \]
<p>with \(\Delta_{\mathrm{react}}=0\) under determinism (Theorem 2). And for any \(\pi,\kappa\):</p>
\[ V^{\pi,\kappa}-V^{\pi,H}\ \le\ \Delta_{\mathrm{react}}+\Delta_{\mathrm{epis}}(\pi) \]
<p>— the <b>decomposition of the adaptive gain</b>. The first term is unrecoverable (paid in execution horizon);
the second is absorbed by the policy.</p>

<h4>III.6 Why improvement must be taken at \(k=H\)</h4>
<p><b>Lemma B (our objective lower-bounds the deployed value, and is tight at the endpoint).</b> If the selector
maximises the true \(Q^\pi_k\), then \(V^{\pi,\kappa^*}\ge V^{\pi,H}\) for every \(\pi\). Hence raising
\(J_H(\pi):=\mathbb E_s[Q^\pi(s,\mu_\pi(s),H)]\) raises a <b>lower bound on the deployed value</b>, and that bound
becomes <b>exactly tight</b> at the point where adaptive execution stops helping
(\(V^{\pi,\kappa^*}=V^{\pi,H}\)) — the recompose endpoint.<br>
<i>Proof.</i> \(\kappa\equiv H\) is among the candidates, so the argmax under the true \(Q\) cannot be smaller. ∎</p>
<p><b>Contrast (why taking it at short \(k\) fails).</b> If improvement is applied only at the selected \(k<H\),
only the chunk's <b>prefix</b> improves and \(Q(s,\mu(s),H)\) does not. Then \(\Delta_{\mathrm{epis}}\) does not
shrink, the corollary of Theorem 2 never fires, and <b>no chunk-length curriculum arises</b>. This is the
theoretical explanation of why selection-only methods (AQC/ExRL/ACSAC) cannot grow the commitment length.</p>

<h4>III.7 Curriculum — a curve the theory predicts</h4>
<p>Let \(\mathcal S_<(\pi):=\{s:\max_{k<H}Q^\pi_k(s)>Q^\pi_H(s)\}\) be the strictly-short set. By the corollary of
Theorem 2, in the deterministic limit and at the optimal \(\pi\), \(\mathcal S_<=\varnothing\) (\(\kappa\equiv H\)
is optimal). In general the total value obtainable on \(\mathcal S_<\) is bounded by
\(\Delta_{\mathrm{react}}\le\varepsilon_HH\bar H\) (Theorem 3). Hence <b>as policy improvement absorbs
\(\Delta_{\mathrm{epis}}\), the mean execution length rises monotonically toward the aleatoric floor</b> — and it
does so with <b>no added reward</b> (no replan cost); we never touch the return. Residual indifference near
convergence, if we want it resolved, is handled by a <b>lexicographic rule</b> (longest \(k\) within the
return-optimal \(\pm\epsilon\) set), whose only free parameter is a <b>comparison tolerance</b>, not the size of a
cost.</p>

<h4>III.8 Proposition (leakage bias grows with \(k\), and the baseline does not cancel it)</h4>
<p>Teleop data is closed-loop, so Prop. 1's hindsight leakage is structurally present (I.8). Writing
\(b^Q_k:=\hat Q^k-Q^k\) and \(b^V_k:=\hat V^k-V^k\), the selector score decomposes as</p>
\[ \frac{\hat Q^k-\hat V^k}{\gamma^k}=\underbrace{\frac{Q^k-V^k}{\gamma^k}}_{\text{true advantage}}
+\underbrace{\frac{b^Q_k-b^V_k}{\gamma^k}}_{\text{selection bias}} \]
<p>(a) As we verified numerically in I.15, DQC Thm. 1's bias bound is <b>monotonically increasing</b> in \(k\).
(b) Leakage is a phenomenon of <b>conditioning on the chunk</b>, whereas \(V^k(s)\) does not condition on a chunk,
so \(b^V_k\) carries less of it ⇒ the subtraction <b>does not fully cancel</b>. (c) The residual is then
<b>amplified with \(k\)</b> by \(1/\gamma^k\).</p>
<div class='callout'><span class='k'>Consequence</span>
AQC corrects <b>only one</b> of two systematic artifacts that point in <b>opposite directions</b> — discount scale
(→ prefers short) and leakage optimism (→ prefers long) — and that correction <b>amplifies</b> the other. And as
II.2 shows, the sign of the correction itself depends on the reward convention. This is the first thing we must
measure.</div>

<h4>III.9 Pre-registration — what would falsify our claims</h4>
<div class='tblwrap'><table>
<tr><th>Test</th><th>How measured</th><th>Pass criterion / what failure means</th></tr>
<tr><td><b>k-dependent optimism</b></td><td>measure \(b^Q_k\) (per-prefix \(\hat Q^k\) vs discounted MC return) and \(b^V_k\) per \(k\)</td><td>if \(b^Q_k-b^V_k\) is flat in \(k\), the baseline cancellation works (III.8 rejected). If it grows, the selector's long-preference is an artifact — <b>every adaptive conclusion needs revisiting</b></td></tr>
<tr><td><b>OOD-candidate calibration</b></td><td>\(\hat Q\) vs realised return for the <b>non-selected</b> candidates the argmax ranks, not only executed trajectories</td><td>large overestimation there means the legitimacy of the argmax is unverified (a region ACSAC's on-policy calibration cannot see)</td></tr>
<tr><td><b>Causality of the curriculum</b></td><td>compare mean-length trajectories with policy improvement off (selection only) vs on</td><td>if length grows in the off arm too, it is <b>not our contribution</b> (merely reproducing AQC/ExRL). Growth only in the on arm supports III.7</td></tr>
<tr><td><b>The aleatoric floor</b></td><td>residual short-chunk usage per skill at convergence</td><td>going to zero means that skill is effectively fully observed and deterministic (our Thm. 2 limit). Converging to a positive value makes it <b>that skill's intrinsic reactivity demand</b> — a new measurement. Dropping below the floor suggests overestimation</td></tr>
</table></div>

<h4>III.10 Summary</h4>
<p>QC proved the <b>benefit</b> of chunked critics (unbiased h-step backups); DQC first quantified the <b>cost</b>
(OLC, Thm. 1, Cor. 1 — and identified that cost as dynamics stochasticity via Prop. 4); AQC generalised it to
<b>state-dependent re-querying</b> (AOLC, meta-MDP dominance). Across all three, <b>one slot stays empty: making
the policy absorb the gain that adaptive execution discovers.</b> We turn that into a theorem (Theorem 2:
\(V^\star_1=V^\star_H=V^\star_{\mathrm{ada}}\) under determinism, with its absorbability corollary), justify
taking improvement at \(k=H\) (Lemma B), and obtain the chunk-length curriculum as a by-product <b>with no added
reward</b>. What remains is not the objective but the critic's <b>\(k\)-dependent optimism and OOD
calibration</b> — the four pre-registered tests above.</p>
"""
