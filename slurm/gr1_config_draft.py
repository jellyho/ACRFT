# GR1 tabletop 이식용 TrainConfig 초안 — 검토용 문서 (config.py 등록 전).
# 학습 자원 결정 후 이 블록을 src/openpi/training/config.py 에 옮기고 norm stats 를 계산한다.
#
# 데이터 사실 (검증 완료, gr1-port 리포트 참조):
#   /scratch/jellyho/acrft/gr1_data/LeRobot/gr1_unified.<Task>/   (LeRobot, fps 20)
#   observation.images.ego_view [256,256,3] 단일 카메라 · observation.state [44] · action [44]
#
# 필요한 신규 코드 (config 외):
#   1. LeRobotGR1DataConfig: ego_view -> base 카메라 매핑(나머지 카메라 슬롯은 zero/mask),
#      state/action 44d 그대로 (모델 action_dim=48 로 패딩), prompt_from_task.
#   2. norm stats: uv run scripts/compute_norm_stats.py --config-name=pi05_gr1_rlt
#
# TrainConfig(
#     name="pi05_gr1_rlt",
#     model=pi0_rlt.Pi0RLTConfig(
#         pi05=True,
#         action_horizon=16,            # 20fps, PrepareCoffee 와 동일 청크 길이
#         action_dim=48,                # GR1 44d + 패딩 (pi05_base 는 32 — in/out proj 재초기화됨)
#         discrete_state_input=False,
#         rlt_backbone_gradient=False,
#     ),
#     data=LeRobotGR1DataConfig(        # 신규 (위 1)
#         repo_id="gr1_unified.PnPCanToDrawerClose",   # 파일럿 1태스크
#         base_config=DataConfig(prompt_from_task=True),
#     ),
#     batch_size=32,
#     lr_schedule=_optimizer.CosineDecaySchedule(
#         warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5
#     ),
#     # KeepMissing: action_dim 변경으로 새로 생기는 in/out projection 은 fresh 초기화 유지
#     weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(
#         "gs://openpi-assets/checkpoints/pi05_base/params"
#     ),
#     num_train_steps=100_000,
#     save_interval=10_000,
#     rlt_monitor_interval=1_000,
# )
#
# 파일럿 순서 (gr1-port 리포트의 사전등록 순서):
#   미세조정 -> 평가로 headroom(베이스 성공률) + rand-vs-vla 갭(후보 스프레드) 측정
#   -> 유효하면 주석 -> FINAL 레시피 critic -> 페어드 판정.
