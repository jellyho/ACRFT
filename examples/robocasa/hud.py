"""Compose a rollout frame that explains the critic's decision, not just the arm's position.

    +---------------------------+---------------------------+------------------+
    | header: mode, Q, step                                                     |
    +---------------------------+---------------------------+------------------+
    |                           |                           |  Q[cand, commit] |
    |   agent view (784px)      |   wrist view (784px)      |  grid            |
    |   candidate paths drawn   |   same overlay, its own   +------------------+
    |   on both cameras         |   camera's projection     |  value trace     |
    |                           |                           |  + commit spans  |
    +---------------------------+---------------------------+------------------+

The two camera views are the same size on purpose: the wrist camera is where fine manipulation is
actually visible, and a thumbnail wastes it. The right column answers what the pixels cannot:

  Q grid             one cell per (candidate, commit length); colour = Q, the chosen cell outlined.
                     This IS the decision surface the joint arg-max picks from - a bar chart of
                     per-candidate maxima hid the prefix axis entirely.
  value trace        on a LOG axis: V = gamma^(steps to success) makes log V linear in the steps
                     remaining, so a consistent critic traces a straight line; the right-hand axis
                     relabels the same curve in implied steps.
  committed span     how many of the H steps each replan agreed to run (or, for QC critics with no
                     prefix head, the per-replan candidate value spread - the only decision QC makes).

The matplotlib panels are re-rendered only when a replan changes them, not per frame - the value and
the candidate set are constant between replans, so this is 5-10x fewer renders with identical output.
"""

import numpy as np

_HDR = 36  # header strip
_CAM = 784  # each camera view, square
_GAP = 4
_RIGHT = 344  # width of the analytics column
_GRID_H = 360
WIDTH = 2 * _CAM + 2 * _GAP + _RIGHT  # 1920
HEIGHT = _HDR + _CAM + _GAP  # 824
_TRACE_H = HEIGHT - _HDR - _GRID_H - _GAP  # 424

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
    ax.tick_params(colors=_INK2, labelsize=9, length=2, width=0.6)
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
        self._f_lg = _font(22)
        self._f_md = _font(17)
        self._f_sm = _font(13, mono=True)

    # ---------------------------------------------------------------- panels
    def _candidate_panel(self, info):
        """The critic's decision surface, drawn as the matrix it actually is.

        One cell per (candidate, prefix length); colour = Q. The joint arg-max picks a single cell,
        which is outlined. A bar chart of per-candidate maxima hid the prefix axis and made "why not
        the longest bar" unanswerable in modes that pin the candidate; the grid shows exactly what
        was scored and exactly what was chosen. Values span a tiny range, so colour is normalised to
        the frame's own [min, max] and both are printed - the SPREAD number is the honest scale.
        """
        from matplotlib.patches import Rectangle

        q = np.asarray(info.q)  # [N, P]
        n, pnum = q.shape
        lo, hi = float(q.min()), float(q.max())
        spread = hi - lo
        fig = _fig(_RIGHT, _GRID_H)
        ax = fig.add_axes([0.15, 0.16, 0.79, 0.70])
        _style(ax)
        ax.imshow(q, aspect="auto", cmap="viridis", vmin=lo, vmax=hi if hi > lo else lo + 1e-9, interpolation="nearest")
        # The chosen cell - the one decision this whole panel exists to explain.
        ax.add_patch(
            Rectangle((info.best_prefix - 0.5, info.best_cand - 0.5), 1, 1, fill=False, edgecolor=_WIN, lw=2.4)
        )
        if self.mode == "prefix":
            # Candidate is pinned: only row 0's cells were ever eligible.
            ax.add_patch(Rectangle((-0.5, -0.5), pnum, 1, fill=False, edgecolor=_INK, lw=1.2, ls=":"))
        ax.set_xticks(range(pnum))
        ax.set_xticklabels([str((k + 1) * (self.horizon // pnum)) for k in range(pnum)], fontsize=8)
        ax.set_yticks([0, n - 1])
        ax.set_yticklabels(["0", str(n - 1)], fontsize=8)
        ax.tick_params(colors=_INK2, length=2, width=0.6)
        ax.set_xlabel("commit steps", color=_INK2, fontsize=9, labelpad=2)
        ax.set_ylabel("candidate", color=_INK2, fontsize=9, labelpad=2)
        ax.set_title(f"Q[cand, commit]   spread {spread:.4f}", color=_INK, fontsize=10.5, pad=6, loc="left")
        pick = (
            f"chosen: cand #{info.best_cand} @ {info.n_exec} steps"
            if pnum > 1
            else f"chosen: cand #{info.best_cand} (no prefix head)"
        )
        if self.mode == "prefix":
            pick += " — pinned #0"
        fig.text(0.15, 0.012, pick, ha="left", va="bottom", color=_WIN, fontsize=9.5)
        return _render(fig)

    def _trace_panel(self):
        fig = _fig(_RIGHT, _TRACE_H)
        gs = fig.add_gridspec(
            2, 1, height_ratios=[2.0, 1.0], hspace=0.30, left=0.185, right=0.83, top=0.90, bottom=0.11
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
        ax.minorticks_off()  # log minor labels shred a narrow range into noise
        ax.set_ylabel("Q chosen", color=_INK2, fontsize=9, labelpad=2)
        ax.set_title(
            f"value   now {raw[-1]:.4f}   ≈{np.log(v[-1]) / np.log(self.discount):.0f} steps left",
            color=_INK,
            fontsize=10.5,
            pad=6,
            loc="left",
        )
        if oob:
            ax.text(
                0.998,
                1.05,
                f"{oob} above support",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="#d03b3b",
                fontsize=8.5,
            )
        # log V is linear in the steps remaining, so relabelling the same axis in those units costs
        # nothing and is what the number means. Not a second measure - a change of unit.
        rax = ax.twinx()
        rax.set_yscale("log")
        rax.minorticks_off()
        rax.set_ylim(ax.get_ylim())
        ticks = [t for t in (0.9, 0.5, 0.2, 0.05, 0.01, 0.002) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        rax.set_yticks(ticks)
        rax.set_yticklabels([f"{np.log(t) / np.log(self.discount):.0f}" for t in ticks])
        rax.tick_params(colors=_INK2, labelsize=8, length=2, width=0.6)
        for s in rax.spines.values():
            s.set_visible(False)
        rax.set_ylabel("steps left", color=_INK2, fontsize=9, rotation=270, labelpad=13)
        ax.tick_params(labelbottom=False)  # the step axis is shared with the panel below

        bx = fig.add_subplot(gs[1], sharex=ax)
        _style(bx)
        w = max(1.0, 0.6 * (np.diff(x).mean() if len(x) > 1 else 4))
        if self.has_prefix:
            bx.bar(x, self.spans, width=w, color=_SPAN)
            bx.axhline(self.horizon, color=_OTHER, lw=0.7, ls="--")
            bx.set_ylim(0, self.horizon * 1.15)
            bx.set_ylabel("steps run", color=_INK2, fontsize=9, labelpad=2)
            bx.set_title(
                f"commit per replan   mean {float(np.mean(self.spans)):.1f}/{self.horizon}",
                color=_SPAN,
                fontsize=9.5,
                pad=4,
                loc="left",
            )
        else:
            # The whole of best-of-N is the gap between the best candidate and the rest. If this bar
            # is flat at ~0 the selection is a coin flip, which is the claim the numbers make offline
            # and the thing a viewer should be able to check for themselves while watching.
            sp = np.asarray(self.spreads, np.float64)
            bx.bar(x, sp, width=w, color=_SPAN)
            bx.set_ylim(0, max(float(sp.max()) * 1.15, 1e-9))
            bx.set_ylabel("best−worst", color=_INK2, fontsize=9, labelpad=2)
            bx.set_title(
                f"cand spread per replan   mean {float(sp.mean()):.5f}  (≈0 ⇒ coin flip)",
                color=_SPAN,
                fontsize=9.5,
                pad=4,
                loc="left",
            )
        bx.set_xlabel("env step", color=_INK2, fontsize=9)
        return _render(fig)

    # The magnification factor lives in action_overlay.PATH_GAIN and is applied in WORLD space,
    # before projection - this class only draws the projected points and prints the label. Stretching
    # the projected 2-D points instead (the first attempt) kept the anchor but destroyed the
    # perspective foreshortening, so the fan stopped reading as a 3-D trajectory.
    def _draw_paths(self, img, paths, chosen):
        """Overlay the candidate end-effector paths on one (already upscaled) camera view.

        The projector returns (row, col); PIL wants (x=col, y=row), and getting that backwards
        transposes every path into a diagonal streak across the frame. The unchosen candidates are
        drawn translucent, which needs an RGBA layer - alpha is ignored when drawing straight onto
        an RGB image.
        """
        from PIL import Image
        from PIL import ImageDraw

        k = _CAM / self.cam

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
            d.line(pts, fill=(150, 175, 200, 120), width=3)
            d.ellipse(
                [pts[-1][0] - 2.5, pts[-1][1] - 2.5, pts[-1][0] + 2.5, pts[-1][1] + 2.5], fill=(150, 175, 200, 170)
            )
        if 0 <= chosen < len(paths):
            pts = _screen(paths[chosen])
            if pts is not None and len(pts) >= 2:
                d.line(pts, fill=(49, 196, 141, 255), width=5, joint="curve")
                d.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4], fill=(49, 196, 141, 255))
        return Image.alpha_composite(base, layer).convert("RGB")

    # ---------------------------------------------------------------- frame
    def frame(self, agent_rgb, wrist_rgb, info, step, *, paths=None, wrist_paths=None, chosen=0, success=False):
        """One composed frame.

        `paths` / `wrist_paths` are optional lists of projected 2-D candidate EE paths, one per
        camera - each camera needs its own projection, and the wrist camera rides the arm so its
        matrix changes every frame.
        """
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
        main = Image.fromarray(np.ascontiguousarray(agent_rgb)).resize((_CAM, _CAM), Image.LANCZOS)
        if paths is not None:
            main = self._draw_paths(main, paths, chosen)
        canvas.paste(main, (0, _HDR))
        if wrist_rgb is not None:
            wr = Image.fromarray(np.ascontiguousarray(wrist_rgb)).resize((_CAM, _CAM), Image.LANCZOS)
            if wrist_paths is not None:
                wr = self._draw_paths(wr, wrist_paths, chosen)
            canvas.paste(wr, (_CAM + _GAP, _HDR))
        rx = 2 * (_CAM + _GAP)
        if self._cand_png is not None:
            canvas.paste(Image.fromarray(self._cand_png), (rx, _HDR))
        if self._trace_png is not None:
            canvas.paste(Image.fromarray(self._trace_png), (rx, _HDR + _GRID_H + _GAP))

        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, WIDTH, _HDR - 2], fill=(10, 13, 12))
        d.text((14, 5), self.mode, font=self._f_lg, fill=_INK)
        d.text((WIDTH - 190, 9), f"step {step:4d}", font=self._f_sm, fill=_INK2)
        if info is not None:
            hdr = (
                f"Q {info.value:.4f}   run {info.n_exec}/{self.horizon}"
                if self.has_prefix
                else f"Q {info.value:.4f}   spread {self.spreads[-1]:.5f}"
            )
            d.text((150, 8), hdr, font=self._f_md, fill=_WIN)
        else:
            d.text((150, 9), "no critic — first sample, full chunk", font=self._f_md, fill=_INK2)
        d.text((14, _HDR + 6), "agent view", font=self._f_sm, fill=_INK2)
        d.text((_CAM + _GAP + 14, _HDR + 6), "wrist view", font=self._f_sm, fill=_INK2)
        if paths is not None:
            import action_overlay as _ov  # the label must track the constant, not restate it

            d.text(
                (14, HEIGHT - 26),
                f"chunk paths x{_ov.PATH_GAIN:g}, world-scale (true span ~3 cm)",
                font=self._f_sm,
                fill=(150, 175, 200),
            )
        if success:
            d.rectangle([2, 2, WIDTH - 3, HEIGHT - 3], outline=_WIN, width=4)
            d.rectangle([_CAM - 190, _HDR + 8, _CAM - 14, _HDR + 46], fill=(10, 13, 12))
            d.text((_CAM - 180, _HDR + 13), "SUCCESS", font=self._f_lg, fill=_WIN)
        return np.asarray(canvas)
