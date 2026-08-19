"""Build report.html from report.template.html.

Inlines the figure as a data URI (the artifact host blocks external requests) and
substitutes the live training numbers, so no value in the report is typed by hand.

    python build_report.py
"""

import base64
import pathlib
import re

import numpy as np

HERE = pathlib.Path(__file__).parent


def data_uri(png: pathlib.Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def main() -> None:
    progress = np.load(HERE / "data" / "progress.npz")
    step, bc = progress["step"], progress["bc_loss"]

    html = (HERE / "report.template.html").read_text()
    html = html.replace("{{FIGURE_PROGRESS}}", data_uri(HERE / "figs" / "progress.png"))
    html = html.replace("{{STEP}}", f"{int(step[-1]):,}")
    html = html.replace("{{LOSS}}", f"{float(bc[-1]):.4f}")

    if left := re.findall(r"\{\{[A-Z_]+\}\}", html):
        raise SystemExit(f"unsubstituted placeholders: {sorted(set(left))}")

    out = HERE / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB) at step {int(step[-1]):,}, bc_loss {float(bc[-1]):.4f}")


if __name__ == "__main__":
    main()
