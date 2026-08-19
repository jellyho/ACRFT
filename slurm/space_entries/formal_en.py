FORMAL_EN = r"""
<h3>8.5 Formalisation — does our scheme actually reach the conclusion we want? (definitions, theorems, proofs)</h3>
<p class='sub'>Up to here it was analogy. Now we define §8's design precisely (<b>evaluate per prefix, execute
adaptively, improve at the full chunk</b>) and <b>prove that it reaches the conclusion we want</b>. All proofs are
elementary — the point is not new machinery but that <b>closing the improvement loop inside the right MDP</b>
makes the standard theorems fire.</p>

<h4>Definition 1 — the full-commitment MDP \(M_H\)</h4>
<p>Given the base MDP \(M=(\mathcal S,\mathcal A,T,r,\gamma)\) and chunk length \(H\), define the MDP whose
<b>action is one whole chunk</b> and whose <b>transition is "execute it open-loop to the end"</b>:</p>
\[ M_H:=\big(\mathcal S,\ \mathcal A^H,\ T_H,\ r_H,\ \gamma^H\big),\qquad
r_H(s,a):=\mathbb E\Big[\sum_{j=0}^{H-1}\gamma^j r(s_j,a_j)\Big],\quad
T_H(\cdot\mid s,a):=\text{law of }s_H \]
<p>\(M_H\) is an ordinary discounted MDP with discount \(\gamma^H\), and its optimal value is exactly the
<b>\(V^\star_H\)</b> of Part III.</p>

<h4>Definition 2 — the decoupled-horizon scheme (DH)</h4>
<div class='tblwrap'><table>
<tr><th>Role</th><th>Definition</th><th>Depends on</th></tr>
<tr><td><b>improvement critic</b></td><td>\(Q^\pi_H(s,a):=r_H(s,a)+\gamma^H\,\mathbb E_{s'\sim T_H}\big[V^{\pi,H}(s')\big]\)</td><td><b>not</b> on \(\kappa\)</td></tr>
<tr><td><b>improvement</b></td><td>\(\pi_{n+1}(s)\in\arg\max_{a\in\mathcal A^H}Q^{\pi_n}_H(s,a)\)</td><td><b>not</b> on \(\kappa\)</td></tr>
<tr><td><b>deployment critic</b></td><td>\(Q^{\pi,\kappa}_k(s,a):=\mathbb E\big[\sum_{j<k}\gamma^jr+\gamma^kV^{\pi,\kappa}(s_k)\big]\)</td><td>per prefix \(k\)</td></tr>
<tr><td><b>execution</b></td><td>\(\kappa(s)\in\arg\max_{k\in\mathcal K}Q^{\pi,\kappa}_k(s,\pi(s))\)</td><td>state-adaptive</td></tr>
</table></div>
<div class='callout'><span class='k'>this is precisely what "decoupled" means</span>
The improvement loop (rows 1–2) <b>never references \(\kappa\)</b>: improvement is <b>closed inside \(M_H\)</b>.
The selector appears only at deployment. That separation is what makes every theorem below work.</div>

<h4>Theorem A — the improvement loop drives the epistemic term to zero</h4>
<p><b>Theorem.</b> For \(\{\pi_n\}\) of Definition 2, \(V^{\pi_{n+1},H}\ge V^{\pi_n,H}\) pointwise and
\(V^{\pi_n,H}\to V^\star_H\). Hence \(\Delta_{\mathrm{epis}}(\pi_n):=V^\star_H-V^{\pi_n,H}\to0\).</p>
<p><i>Proof.</i> Rows 1–2 are exactly <b>policy iteration in \(M_H\)</b> (states \(\mathcal S\), actions
\(\mathcal A^H\), discount \(\gamma^H<1\), bounded \(r_H\)). Standard policy-iteration theory gives monotone
improvement and convergence to the unique optimum \(V^\star_H\). ∎</p>
<p><b>Corollary (approximate version — what actually happens in training).</b> If greedification is
\(\delta\)-approximate, the standard approximate-greedy bound at discount \(\gamma^H\) gives</p>
\[ \limsup_n\ \Delta_{\mathrm{epis}}(\pi_n)\ \le\ \frac{\delta}{1-\gamma^H}\ =\ \delta\,\bar H \]
<p>So the <b>residual epistemic term is the actor's optimisation error times the chunk-effective horizon</b> —
the quantitative reason we want an actor whose maximisation is accurate and cheap (the FQL one-step actor).</p>

<h4>Proposition E — why improvement must be taken at \(H\) (the piano analogy, exactly)</h4>
<p><b>Proposition.</b> Suppose improvement instead maximises the \(k\)-prefix value \(Q_k\) at the deployed
\(k(s)<H\). Then the objective <b>does not depend on the tail</b> \(a_{k(s):H}\) at all; the tail receives no
pressure and \(V^{\pi,H}\) need not increase — \(\Delta_{\mathrm{epis}}\) can stay bounded away from 0 forever.</p>
<p><i>Proof.</i> \(Q^{\pi,\kappa}_k(s,a)\) depends on \(a\) only through \(a_{0:k}\); changing \(a_{k:H}\) leaves
the objective unchanged, so the argmax does not determine the tail. ∎</p>

<h4>Theorem B — adaptive execution is safe (it cannot break the lower bound)</h4>
<p><b>Theorem.</b> For any \(\pi\), if the greedy selector \(\kappa_1(s)\in\arg\max_{k}Q_k(s,\pi(s))\) is formed
with continuation \(V^{\pi,H}\), then \(V^{\pi,\kappa_1}\ge V^{\pi,H}\) pointwise.</p>
<p><i>Proof.</i> For fixed \((\pi,\kappa)\) the operator
\((\mathcal T^{\pi,\kappa}V)(s):=\mathbb E[\sum_{j<\kappa(s)}\gamma^jr+\gamma^{\kappa(s)}V(s_{\kappa(s)})]\) is
<b>monotone</b> and a <b>contraction</b> with modulus \(\max_s\gamma^{\kappa(s)}\le\gamma^{k_{\min}}<1\), so it has
the unique fixed point \(V^{\pi,\kappa}\). Since \(H\in\mathcal K\),
\(V^{\pi,H}(s)=Q_H(s,\pi(s))\le\max_kQ_k(s,\pi(s))=(\mathcal T^{\pi,\kappa_1}V^{\pi,H})(s)\); iterating the
monotone operator drives the left side up to its fixed point, giving \(V^{\pi,H}\le V^{\pi,\kappa_1}\). ∎</p>

<h4>Corollary C — the main result: where the scheme lands</h4>
<p>Combining A and B, the deployed value satisfies \(V^{\pi_n,\kappa}\ge V^{\pi_n,H}\to V^\star_H\), hence</p>
\[ \limsup_n\ \big[V^\star_1-V^{\pi_n,\kappa}\big]\ \le\ V^\star_1-V^\star_H\ =\ \Delta_{\mathrm{alea}}
\ \le\ \varepsilon_H H\bar H,\qquad \varepsilon_H=3\big(1-(1-\varepsilon)^{H-1}\big) \]
<p>(the last step from DQC Corollary 1 + Proposition 4), and <b>\(\Delta_{\mathrm{alea}}=0\) in a deterministic,
fully observed world</b> (our Theorem 2, Part III.3).</p>
<div class='callout'><span class='k'>in words</span>
<b>The decoupled-horizon scheme converges to within the aleatoric floor of the true optimum, and that floor is
exactly zero in a predictable world.</b> What we fail to recover is only what <b>nobody can recover in
principle</b> — and adaptive execution can only <b>add</b> to that guarantee, never break it (Theorem B).</div>
<p class='sub'><b>Honest caveat.</b> Theorem A relies on the improvement critic bootstrapping with the
<b>full-commitment continuation \(V^{\pi,H}\)</b>. If it instead bootstraps with the <b>deployed (adaptive)</b>
continuation, the loop is no longer closed inside \(M_H\) and the convergence argument does not transfer. This is
not an abstract worry but a <b>concrete design prescription</b> — and an ablation we will run (comparing the two
bootstrap targets).</p>

<h4>Theorem D — the endpoint: how far does the length grow?</h4>
<p>Let the strictly-short set be \(\mathcal S_<(\pi):=\{s:\max_{k<H}Q_k(s,\pi(s))>Q_H(s,\pi(s))\}\).</p>
<p><b>Theorem.</b> Under full observability and deterministic dynamics, at \(\pi^\star=\pi^\star_H\) (the limit of
Theorem A) we have \(Q_k(s,\pi^\star(s))=Q_H(s,\pi^\star(s))\) for every \(s\) and every \(k\in\mathcal K\).
Hence \(\mathcal S_<(\pi^\star)=\varnothing\), \(\kappa\equiv H\) is optimal, and under the lexicographic
tie-break the <b>mean commitment length equals \(H\) exactly</b>.</p>
<p><i>Proof.</i> By Theorem 2 (Part III.3), \(V^\star_H=V^\star_1\) and \(\pi^\star\) attains it, so \(\pi^\star\)
traces an <b>optimal trajectory</b> from every \(s\). Executing \(k\) steps and re-querying at \(s_k\) returns
\(\pi^\star(s_k)\), which by determinism, the Markov property and optimality is the <b>continuation of the same
trajectory</b>. The realised trajectory is therefore identical for every \(k\), and so are the values. ∎</p>
<p><b>In general</b>, the total advantage available on \(\mathcal S_<(\pi^\star)\) is bounded by
\(\Delta_{\mathrm{react}}:=V^\star_{\mathrm{ada}}-V^\star_H\le\Delta_{\mathrm{alea}}\) (Part III.5): the mean
length grows but stops at the floor set by genuine reactivity demand.</p>

<h4>What the four theorems say together</h4>
<div class='tblwrap'><table>
<tr><th>Result</th><th>Claim</th><th>What it guarantees</th></tr>
<tr><td><b>A</b></td><td>the improvement loop is policy iteration in \(M_H\)</td><td>epistemic term → 0 (within \(\delta\bar H\) when approximate)</td></tr>
<tr><td><b>E</b></td><td>improving at short \(k\) leaves the tail unconstrained</td><td><b>why it must be \(k=H\)</b></td></tr>
<tr><td><b>B</b></td><td>greedy selection cannot break \(V^{\pi,H}\)</td><td><b>safety</b> of adaptive execution</td></tr>
<tr><td><b>C</b></td><td>deployed value \(\to\) at least \(V^\star_H\); gap ≤ the aleatoric floor</td><td><b>where the scheme lands</b> (optimal in a predictable world)</td></tr>
<tr><td><b>D</b></td><td>at the endpoint no state prefers to break</td><td>the <b>curriculum</b> is a consequence, not a hope</td></tr>
</table></div>
<p>In short, "<b>split the evaluation, join the improvement</b>" is not a taste: it is a <b>provable structure</b>
in which (i) improvement is closed inside the right MDP and therefore converges, (ii) selection cannot damage that
convergence, and (iii) what remains is exactly the part nobody could have recovered.</p>
"""
