# misc/scripts — one-off analysis, figure and report scripts

Moved out of `scripts/` on 2026-09-09. Nothing here is invoked by a launcher in `slurm/`, imported by
`src/`, or documented as a workflow step in `README.md` / `docs/` — that was the test for staying.
They are kept rather than deleted because CLAUDE.md requires every published figure and number to
stay regenerable from a script.

**They are flat on purpose.** Roughly fifteen of the figure scripts do a bare `import report_style`,
and several import each other the same way, so splitting them into subdirectories would break the
imports. Run them from this directory, or with it on `PYTHONPATH`.

`misc/` is excluded from ruff (`pyproject.toml`), so nothing here is linted — this is an archive, not
maintained code.

## What is here

| group | files | what they are |
|---|---|---|
| house style | `report_style.py` | the palette/rcParams every figure script below imports |
| figures | `fig_*.py`, `*_figs.py`, `make_fig_*.py`, `plot_*.py` | one figure or figure set per published claim |
| videos / viz | `viz_*.py`, `render_*.py`, `make_nn_browser.py`, `make_nonmarkov_region_clips.py` | rollout and value-landscape renders |
| hub entries | `make_wa_entry_*.py`, `make_wb_entry_*.py`, `make_overnight_report_*.py`, `make_weekly_figs.py`, `make_weekly_slides.py` | compose one report entry each; the live publisher is `slurm/sync_hub.py` |
| Space tooling | `space_add_entry.py`, `space_build.py`, `space_migrate.py`, `space_offload_figs.py`, `SPACE_ENTRIES_SCHEMA.md` | the `space_v2/` publishing path |
| probes / diagnostics | `diag_*.py`, `probe_*.py`, `measure_*.py`, `replicate_*.py`, `eval_mbac_offline.py`, `eval_onestep_bc.py`, `score_critic_auc.py`, `legoprog_stats.py` | one question each, usually behind a published entry |
| retired training runs | `train_bc_probe.py`, `train_latent_dynamics.py`, `train_mve_critic.py`, `train_v_react.py`, `train_cheapz_dynamics_v1.py`, `fql_smoke.py`, `alphaflow_jvp_stress.py` | experiment arms that are not the current line |
| retired research lines | `train_cheap_z.py`, `train_cheapz_dynamics{,_v1}.py`, `make_cheapz_annot.py`, `probe_cheap_z.py` | the cheap-z study (design doc 2026-08-07), measured entirely on RoboCasa PrepareCoffee and aimed at replacing the RLT VLA token — both retired. The patch critic is what shipped from it. |
| superseded trainers | `train_patch_critic.py` | the pre-cache patch-critic trainer (IQL on transition dirs). The live path is `convert_yam_to_patchcritic` -> `cache_patch_features` -> `train_patch_critic_cached`. |
| data / ops one-offs | `convert_*.py`, `sync_bc_checkpoints.py`, `upload_extraction_arms.py`, `watch_upload_ckpts.py`, `smoke_deploy_hud.py`, `qpilots_steer.py`, `annotate_advantage.py`, `idql_select.py` | |

## Note on published reports

`slurm/make_master_report.py` refers to many of these by their old `scripts/<name>.py` path. Those
strings were left alone: they are the record of how a published result was produced, and rewriting
them would make the entries disagree with the reports already on the hub. Read them as
`misc/scripts/<name>.py`.

## What stayed in `scripts/`

The 41 entry points that are actually wired up: `train.py`, `serve_policy.py`, `compute_norm_stats.py`,
`train_local.sh`, the patch-critic chain (`convert_yam_to_patchcritic`, `cache_patch_features`,
`train_patch_critic{,_cached,_clip}`, `score_critic_cached`, `export_critic_serving`,
`backfill_critic_spec`), the extraction arms (`train_{awr,cfgrl,flowdagger,flowdpg,fqlx,lps,qam}`,
`eval_extraction`, `export_extraction_checkpoint`), and the RLT-critic pair `slurm/train_critic.sbatch`
still submits.
