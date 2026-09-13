# misc/reports — retired report and figure generators

One-off HTML reports and figures that were published once and are not part of any current pipeline.
They live here rather than being deleted because CLAUDE.md's rule is that every published figure has
to stay regenerable from a script — a figure whose generator is gone cannot be audited.

Nothing in `slurm/` imports these. The live publishing path is unchanged and stays in `slurm/`:

| file | role |
|---|---|
| `slurm/make_master_report.py` | the single source of truth — every entry, its 5W1H header, its body |
| `slurm/make_figures.py` | regenerates every JSON-backed figure on each report build |
| `slurm/plot_style.py` | the house plot style (imported by ~20 scripts, including one here) |
| `slurm/sync_hub.py` | publishes to the HF Space, replacing only this worker's entries |
| `slurm/make_weekly_deck.py` | assembles the weekly presentation from `hub_figs/` + result JSONs |

## State of each file here

| file | runs today? |
|---|---|
| `fig_qpilots_rerun.py` | yes — imports `slurm/plot_style.py` via an explicit path |
| `make_autopsy_plots.py`, `make_fit_report.py`, `make_plots.py`, `make_run_level_plot.py`, `make_week_report.py` | self-contained; run if their input JSONs are still on disk |
| `make_bias_report.py`, `make_progress_report.py`, `make_video_dashboard.py` | **no** — they import `slurm/_describe.py`, deleted in commit 3d8e8f5 ("dead files removed"), so they have been broken since then. Restore that file from 3d8e8f5's parent to run them. |

`misc/` is excluded from ruff (see `pyproject.toml`), so these are not linted — they are an archive,
not maintained code.
