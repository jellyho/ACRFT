"""Draw the critic's decision onto a rollout frame.

A video of a critic-guided rollout shows the arm moving and nothing about why. The three things
that actually explain the behaviour are invisible in the pixels:

    the value trace     is the critic getting more confident as the task progresses, or drifting?
                        A failure while the value climbs is a different bug from one while it falls.
    the candidate fan   how far apart the N candidates score. If they sit on top of each other the
                        arg-max is picking noise and best-of-N is a coin flip, whatever the mean.
    the committed span  how many of the H steps the critic agreed to run before re-planning. If this
                        pins to the shortest prefix every time, adaptive chunking has collapsed.

Everything is drawn with numpy slicing on the RGB array, so there is no font or GUI dependency to
carry onto a headless eval node - digits come from a small bitmap table below.
"""

import numpy as np

_PANEL = 78  # height of the HUD strip appended under the frame
_FG = np.array([238, 238, 234], np.uint8)
_DIM = np.array([120, 126, 124], np.uint8)
_WIN = np.array([80, 190, 140], np.uint8)  # the chosen candidate
_TRACE = np.array([90, 150, 220], np.uint8)
_BG = np.array([18, 22, 21], np.uint8)

# 3x5 bitmaps, enough for the numbers and the few words the HUD needs.
_GLYPH = {
    "0": "111101101101111",
    "1": "010010010010010",
    "2": "111001111100111",
    "3": "111001111001111",
    "4": "101101111001001",
    "5": "111100111001111",
    "6": "111100111101111",
    "7": "111001001001001",
    "8": "111101111101111",
    "9": "111101111001111",
    ".": "000000000000010",
    "/": "001001010100100",
    "-": "000000111000000",
    " ": "000000000000000",
    "v": "101101101101010",
    "V": "101101101101010",
    "a": "000111101101111",
    "c": "011100100100011",
    "e": "111100110100111",
    "h": "100100111101101",
    "i": "010000010010010",
    "l": "100100100100100",
    "n": "000111101101101",
    "o": "000111101101111",
    "p": "111101111100100",
    "r": "000111100100100",
    "s": "011100010001110",
    "t": "010111010010011",
    "b": "100100111101111",
    "d": "001001111101111",
    "k": "100101110101101",
    "m": "000111111101101",
    "q": "111101111001001",
    "u": "000101101101111",
    "x": "000101010010101",
    "=": "000111000111000",
    ":": "000010000010000",
    "#": "101111101111101",
}


def _text(img, x, y, s, color, scale=1):
    for ch in s:
        bits = _GLYPH.get(ch, _GLYPH[" "])
        for r in range(5):
            for c in range(3):
                if bits[r * 3 + c] == "1":
                    y0, x0 = y + r * scale, x + c * scale
                    img[y0 : y0 + scale, x0 : x0 + scale] = color
        x += 4 * scale
    return x


def draw(frame, info, trace, *, mode="critic"):
    """frame: HxWx3 uint8. info: eval_critic.Replan or None. trace: the value at every step so far."""
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    out = np.empty((h + _PANEL, w, 3), np.uint8)
    out[:h] = frame
    out[h:] = _BG
    p = out[h:]

    x = _text(p, 4, 5, mode, _FG)
    if info is None:  # the plain VLA baseline has nothing to report
        _text(p, x + 8, 5, "no critic", _DIM)
        return out

    q = np.asarray(info.q)  # [N, P]
    n_cand, n_pref = q.shape
    _text(p, x + 8, 5, f"v={info.value:.3f}", _FG)
    _text(p, x + 8, 13, f"exec {info.n_exec}", _DIM)

    # --- candidate fan: one column per candidate, at its best prefix ---------------------------
    # Scaled to the spread within THIS replan, not to [0,1]: the question the picture has to answer
    # is whether the candidates differ from each other, and on an absolute axis a well-calibrated
    # critic's candidates all sit at nearly the same height and the plot says nothing.
    best = q.max(axis=1)
    lo, hi = float(best.min()), float(best.max())
    rng = max(hi - lo, 1e-6)
    bx, by, bh, bw = 4, 26, 22, max(1, (w // 2 - 8) // max(n_cand, 1))
    for i in range(n_cand):
        v = (best[i] - lo) / rng
        bar = max(1, int(v * bh))
        col = _WIN if i == info.best_cand else _DIM
        p[by + bh - bar : by + bh, bx + i * bw : bx + i * bw + bw - 1] = col
    _text(p, bx, by + bh + 3, f"{n_cand} cand spread {rng:.3f}", _DIM)

    # --- committed prefix: which of the P lengths won ------------------------------------------
    px = w // 2 + 4
    pw = max(1, (w - px - 8) // max(n_pref, 1))
    for k in range(n_pref):
        col = _WIN if k == info.best_prefix else _DIM
        p[by + bh - 4 : by + bh, px + k * pw : px + k * pw + pw - 1] = col
    _text(p, px, by + bh + 3, f"prefix {info.best_prefix + 1}/{n_pref}", _DIM)

    # --- value trace over the episode ----------------------------------------------------------
    if len(trace) > 1:
        t = np.asarray(trace, np.float32)
        tlo, thi = float(t.min()), float(t.max())
        tr = max(thi - tlo, 1e-6)
        ty, th = 5, 16
        xs = np.linspace(0, w // 2 - 10, len(t)).astype(int) + w // 2 + 4
        ys = (ty + th - (t - tlo) / tr * th).astype(int)
        for i in range(len(t) - 1):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(n + 1):
                yy = int(y0 + (y1 - y0) * s / n)
                xx = int(x0 + (x1 - x0) * s / n)
                p[np.clip(yy, 0, _PANEL - 1), np.clip(xx, 0, w - 1)] = _TRACE
    return out
