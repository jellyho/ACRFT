"""Compose a rollout frame that explains the critic's decision, not just the arm's position.

    +--------------------------------+----------------+
    |                                |  wrist camera  |
    |   agent view, with the N        +----------------+
    |   candidate paths drawn and     |  candidate     |
    |   the chosen one highlighted     |  values        |
    +--------------------------------+----------------+
    |  value trace (log) + committed span per replan   |
    +--------------------------------------------------+

Three questions the pixels cannot answer on their own, one panel each:

  candidate values   how far apart the N candidates scored at this replan. If they sit on top of one
                     another the arg-max is ranking noise and best-of-N is a coin flip, whichever
                     number the mean reports - so the panel is scaled to this replan's own spread and
                     prints that spread rather than pretending to an absolute axis.
  value trace        drawn on a LOG axis, because V(s) = gamma^(steps to success) makes log V linear
                     in the steps remaining. A consistent critic traces a straight line towards the
                     goal; the right-hand axis relabels the same quantity as implied steps remaining,
                     which is what the number actually means. A run that fails while the value climbs
                     is a different failure from one that fails while it falls.
  committed span     how many of the H steps each replan agreed to run. Pinned to the shortest prefix
                     every time means adaptive chunking has collapsed to a fixed small k.

The matplotlib panels are re-rendered only when a replan changes them, not per frame - the value and
the candidate set are constant between replans, so this is 5-10x fewer renders with identical output.
"""

import numpy as np

_W_MAIN, _W_SIDE, _H_TOP, _H_BOT = 512, 256, 512, 256
WIDTH, HEIGHT = _W_MAIN + _W_SIDE, _H_TOP + _H_BOT

# A cool, slightly green-biased dark ground: the accents sit on it without vibrating, and it reads as
# instrumentation rather than as a slide.
_BG = (16, 20, 19)
_INK = "#e6ebe8"
_INK2 = "#98a3a0"
_GRID = "#2b3230"
_WIN = "#31c48d"  # the chosen candidate / prefix
_OTHER = "#4d5654"
_TRACE = "#4a9eea"
_SPAN = "#c98500"


def _font(size, *, mono=False):
    from matplotlib import font_manager as fm
    from PIL import ImageFont

    return ImageFont.truetype(fm.findfont("DejaVu Sans Mono" if mono else "DejaVu Sans"), size)


def _fig(w, h, dpi=100):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor(np.array(_BG) / 255)
    FigureCanvasAgg(fig)
    return fig


def _render(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return np.ascontiguousarray(buf)


def _style(ax):
    ax.set_facecolor(np.array(_BG) / 255)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=_INK2, labelsize=7, length=2, width=0.6)
    ax.grid(visible=True, color=_GRID, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)


class Dashboard:
    """Accumulates a trial's trace and composes one frame at a time."""

    def __init__(self, *, mode: str, horizon: int, discount: float = 0.99, camera_size: int = 256):
        self.mode, self.horizon, self.discount, self.cam = mode, horizon, discount, camera_size
        self.values: list[float] = []  # one per replan
        self.spans: list[int] = []
        # QC has no prefix head, so "steps committed per replan" is a constant bar at H and tells the
        # viewer nothing. Its one decision is which of the N candidates to run, so the panel is given
        # over to the value gap that choice is buying. Recorded for every critic; only plotted for QC.
        self.spreads: list[float] = []
        self.has_prefix: bool | None = None
        self.steps: list[int] = []  # step index at which each replan started
        self._cand_png = None
        self._trace_png = None
        self._last_id = None
        self._f_lg = _font(15)
        self._f_md = _font(12)
        self._f_sm = _font(10, mono=True)

    # ---------------------------------------------------------------- panels
    def _candidate_panel(self, info):
        q = np.asarray(info.q)  # [N, P]
        best = q.max(axis=1)
        order = np.argsort(-best)
        lo, hi = float(best.min()), float(best.max())
        spread = hi - lo
        fig = _fig(_W_SIDE, _W_SIDE)
        ax = fig.add_axes([0.20, 0.16, 0.76, 0.70])
        _style(ax)
        y = np.arange(len(best))
        vals = best[order] - lo
        cols = [_WIN if i == info.best_cand else _OTHER for i in order]
        ax.barh(y, vals, color=cols, height=0.74, left=0)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0, max(spread, 1e-9) * 1.08)
        ax.set_xlabel(f"value − {lo:.4f}", color=_INK2, fontsize=7.5)
        # One bar per CANDIDATE (not per prefix): a bar's length is that candidate's best value
        # across its prefix heads, i.e. exactly the score the joint arg-max ranks it by. The prefix
        # axis appears only inside the chosen-candidate annotation below.
        ax.set_title(
            f"{len(best)} candidates   spread {spread:.4f}",
            color=_INK,
            fontsize=8.5,
            pad=16,
            loc="left",
        )
        if q.shape[1] > 1:
            ax.text(0.0, 1.015, f"bar = one candidate's best over {q.shape[1]} prefixes",
                    transform=ax.transAxes, ha="left", va="bottom", color=_INK2, fontsize=6.5)
        # The winner's rank among the candidates, which is the thing best-of-N is buying.
        ax.text(
            0.98,
            0.02,
            f"chosen: cand #{info.best_cand}"
            + (f" @ prefix {info.best_prefix + 1}/{q.shape[1]}" if q.shape[1] > 1 else "  (no prefix head)"),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=_WIN,
            fontsize=7.5,
        )
        return _render(fig)

    def _trace_panel(self):
        fig = _fig(WIDTH, _H_BOT)
        gs = fig.add_gridspec(
            2, 1, height_ratios=[2.0, 0.95], hspace=0.17, left=0.118, right=0.900, top=0.855, bottom=0.155
        )
        ax = fig.add_subplot(gs[0])
        _style(ax)
        x = np.asarray(self.steps)
        # The support is [0, 1]; anything above it is the critic leaving its own histogram, which the
        # header calls out rather than quietly reporting a negative number of steps remaining.
        raw = np.asarray(self.values, np.float64)
        oob = int((raw > 1.0).sum())
        v = np.clip(raw, 1e-6, 1.0)
        ax.plot(x, v, color=_TRACE, lw=1.8, solid_capstyle="round")
        ax.plot(x[-1:], v[-1:], "o", color=_TRACE, ms=5, mec=np.array(_BG) / 255, mew=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("Q of the chosen chunk", color=_INK2, fontsize=7.5, labelpad=2)
        ax.set_title(
            f"value trace   now {raw[-1]:.4f}   implied {np.log(v[-1]) / np.log(self.discount):.0f} steps to success",
            color=_INK,
            fontsize=9,
            pad=6,
            loc="left",
        )
        if oob:
            ax.text(
                0.998,
                1.04,
                f"{oob} above support",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="#d03b3b",
                fontsize=7.5,
            )
        # log V is linear in the steps remaining, so relabelling the same axis in those units costs
        # nothing and is what the number means. Not a second measure - a change of unit.
        rax = ax.twinx()
        rax.set_yscale("log")
        rax.set_ylim(ax.get_ylim())
        ticks = [t for t in (0.9, 0.5, 0.2, 0.05, 0.01, 0.002) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        rax.set_yticks(ticks)
        rax.set_yticklabels([f"{np.log(t) / np.log(self.discount):.0f}" for t in ticks])
        rax.tick_params(colors=_INK2, labelsize=7, length=2, width=0.6)
        for s in rax.spines.values():
            s.set_visible(False)
        rax.set_ylabel("implied steps to success", color=_INK2, fontsize=7.5, rotation=270, labelpad=11)
        ax.tick_params(labelbottom=False)  # the step axis is shared with the panel below

        bx = fig.add_subplot(gs[1], sharex=ax)
        _style(bx)
        w = max(1.0, 0.6 * (np.diff(x).mean() if len(x) > 1 else 4))
        if self.has_prefix:
            bx.bar(x, self.spans, width=w, color=_SPAN)
            bx.axhline(self.horizon, color=_OTHER, lw=0.7, ls="--")
            bx.set_ylim(0, self.horizon * 1.15)
            bx.set_ylabel("steps run", color=_INK2, fontsize=7.5, labelpad=2)
            bx.set_title(
                f"steps committed per replan   mean {float(np.mean(self.spans)):.1f} of {self.horizon}",
                color=_SPAN, fontsize=7.5, pad=3, loc="left",
            )
        else:
            # The whole of best-of-N is the gap between the best candidate and the rest. If this bar
            # is flat at ~0 the selection is a coin flip, which is the claim the numbers make offline
            # and the thing a viewer should be able to check for themselves while watching.
            sp = np.asarray(self.spreads, np.float64)
            bx.bar(x, sp, width=w, color=_SPAN)
            bx.set_ylim(0, max(float(sp.max()) * 1.15, 1e-9))
            bx.set_ylabel("best − worst", color=_INK2, fontsize=7.5, labelpad=2)
            bx.set_title(
                f"candidate value spread per replan   mean {float(sp.mean()):.5f}"
                f"   (≈0 ⇒ best-of-N is a coin flip)",
                color=_SPAN, fontsize=7.5, pad=3, loc="left",
            )
        bx.set_xlabel("env step", color=_INK2, fontsize=7.5)
        return _render(fig)

    # The magnification factor lives in action_overlay.PATH_GAIN and is applied in WORLD space,
    # before projection - this class only draws the projected points and prints the label. Stretching
    # the projected 2-D points instead (the first attempt) kept the anchor but destroyed the
    # perspective foreshortening, so the fan stopped reading as a 3-D trajectory.
    def _draw_paths(self, img, paths, chosen):
        """Overlay the candidate end-effector paths on the (already upscaled) agent view.

        The projector returns (row, col); PIL wants (x=col, y=row), and getting that backwards
        transposes every path into a diagonal streak across the frame. The unchosen candidates are
        drawn translucent, which needs an RGBA layer - alpha is ignored when drawing straight onto
        an RGB image.
        """
        from PIL import Image
        from PIL import ImageDraw

        k = _W_MAIN / self.cam

        def _screen(path):
            """Projected (row, col) -> screen (x, y). None if the anchor is outside the camera.

            robosuite CLIPS out-of-view projections to the border instead of dropping them, so when
            the end-effector leaves the frame every path collapses onto the image edge and draws as
            a bogus line hugging the border. If the anchor itself sits on the border, nothing about
            the fan is real - skip it entirely.
            """
            a = np.asarray(path)
            if a[0, 0] <= 0 or a[0, 0] >= self.cam - 1 or a[0, 1] <= 0 or a[0, 1] >= self.cam - 1:
                return None
            return [(float(c) * k, float(r) * k) for r, c in a]

        base = img.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i, path in enumerate(paths):
            pts = _screen(path)
            if pts is None or len(pts) < 2:
                continue
            if i == chosen:
                continue  # drawn last, on top
            d.line(pts, fill=(150, 175, 200, 120), width=2)
            d.ellipse(
                [pts[-1][0] - 1.5, pts[-1][1] - 1.5, pts[-1][0] + 1.5, pts[-1][1] + 1.5], fill=(150, 175, 200, 170)
            )
        if 0 <= chosen < len(paths):
            pts = _screen(paths[chosen])
            if pts is not None and len(pts) >= 2:
                d.line(pts, fill=(49, 196, 141, 255), width=4, joint="curve")
                d.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3], fill=(49, 196, 141, 255))
        return Image.alpha_composite(base, layer).convert("RGB")

    # ---------------------------------------------------------------- frame
    def frame(self, agent_rgb, wrist_rgb, info, step, *, paths=None, chosen=0, success=False):
        """One composed frame. `paths` is an optional list of projected 2-D candidate EE paths."""
        from PIL import Image
        from PIL import ImageDraw

        if info is not None and id(info) != self._last_id:
            self._last_id = id(info)
            _q = np.asarray(info.q)
            if self.has_prefix is None:
                self.has_prefix = _q.ndim > 1 and _q.shape[1] > 1
            _best = _q.max(axis=1) if _q.ndim > 1 else _q
            self.spreads.append(float(_best.max() - _best.min()))
            self.values.append(float(info.value))
            self.spans.append(int(info.n_exec))
            self.steps.append(int(step))
            self._cand_png = self._candidate_panel(info)
            self._trace_png = self._trace_panel()

        canvas = Image.new("RGB", (WIDTH, HEIGHT), _BG)
        main = Image.fromarray(np.ascontiguousarray(agent_rgb)).resize((_W_MAIN, _H_TOP), Image.LANCZOS)
        if paths is not None:
            main = self._draw_paths(main, paths, chosen)
        canvas.paste(main, (0, 0))
        if wrist_rgb is not None:
            wr = Image.fromarray(np.ascontiguousarray(wrist_rgb)).resize((_W_SIDE, _W_SIDE - 22), Image.LANCZOS)
            canvas.paste(wr, (_W_MAIN, 22))
        if self._cand_png is not None:
            canvas.paste(Image.fromarray(self._cand_png), (_W_MAIN, _W_SIDE))
        if self._trace_png is not None:
            canvas.paste(Image.fromarray(self._trace_png), (0, _H_TOP))

        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, _W_MAIN, 30], fill=(10, 13, 12))
        d.text((10, 7), self.mode, font=self._f_lg, fill=_INK)
        d.text((_W_MAIN - 150, 9), f"step {step:4d}", font=self._f_sm, fill=_INK2)
        if info is not None:
            _hdr = (f"Q {info.value:.4f}   run {info.n_exec}/{self.horizon}" if self.has_prefix
                    else f"Q {info.value:.4f}   spread {self.spreads[-1]:.5f}")
            d.text((90, 9), _hdr, font=self._f_md, fill=_WIN)
        else:
            d.text((90, 10), "no critic — first sample, full chunk", font=self._f_md, fill=_INK2)
        d.rectangle([_W_MAIN, 0, WIDTH, 22], fill=(10, 13, 12))
        d.text((_W_MAIN + 9, 5), "wrist camera", font=self._f_sm, fill=_INK2)
        if paths is not None:
            # Say it on the frame, every frame: the fan is a direction-and-spread indicator drawn
            # PATH_GAIN times life size, not a distance the arm will travel.
            import action_overlay as _ov  # the label must track the constant, not restate it

            d.text((8, _H_TOP - 20), f"chunk paths x{_ov.PATH_GAIN:g}, world-scale (true span ~3 cm)",
                   font=self._f_sm, fill=(150, 175, 200))
        if success:
            d.rectangle([2, 2, WIDTH - 3, HEIGHT - 3], outline=_WIN, width=3)
            d.rectangle([_W_MAIN - 138, _H_TOP - 36, _W_MAIN - 8, _H_TOP - 8], fill=(10, 13, 12))
            d.text((_W_MAIN - 130, _H_TOP - 32), "SUCCESS", font=self._f_lg, fill=_WIN)
        return np.asarray(canvas)
