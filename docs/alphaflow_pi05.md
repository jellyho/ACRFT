# α-Flow π0.5 — VLA를 few/one-step 생성기로

> 목적은 추론 속도가 아니라 **offline RL 비용**이다. actor-critic 업데이트는 매 스텝 정책에서
> 액션을 뽑아야 하고, 업데이트마다 10-step ODE를 내는 것이 VLA에 RL을 거는 비용의 대부분이다.
> 원스텝 생성기면 actor 업데이트가 forward 1회로 끝난다. 그리고 distillation과 달리 α-Flow는
> **데이터 위 회귀만으로** 거기 도달한다 — 학습 중 VLA가 샘플링할 일이 없다.

구현: `src/openpi/models/pi0_alphaflow.py` · 테스트: `..._test.py`(CPU), `scripts/alphaflow_smoke.py`(GPU)
설정: `pi05_yam_lego_taxi_alphaflow` (단일 run, in-training 커리큘럼)

## 무엇이 바뀌나

π0.5는 **순간 속도** `v_θ(z_t, t)`를 예측한다. α-Flow는 구간 **평균 속도**
`u_θ(z_t, r, t) ≈ (1/(t−r))∫_r^t v(z_τ,τ)dτ` 를 예측하므로, 한 번의 점프
`z_r = z_t − (t−r)·u_θ(z_t, r, t)` 가 ODE를 대체한다.

`r`은 t와 같은 adaRMS 경로로 들어간다. **r-MLP의 출력층은 zero-init** — 그래서 step 0에서
모든 r에 대해 `u_θ(z_t, r, t) = v_pretrained(z_t, t)` 이고, 이건 파인튜닝이지 재학습이 아니다.
(GPU 스모크 체크 [1]/[2]가 실기 모델에서 이 두 성질을 확인한다.)

## 목적함수

논문 Def. 1, 그리고 둘이 어긋나는 곳에서는 레퍼런스 구현
(`snap-research/alphaflow`, `src/training/loss.py`)을 따랐다.

```
s     = α·r + (1−α)·t                       # dt := t − s = α·(t − r)
z_s   = z_t − (t − s)·v_t
u_tgt = α·v_t + (1−α)·u_θ⁻(z_s, r, s)
L_α   = ‖u_θ(z_t,r,t) − sg(u_tgt)‖²  ×  sg( α / (‖·‖² + ε) )
```

α는 *consistency step ratio*이고, 커리큘럼 자체다:

| α | 목적함수 | 비용 |
|---|---|---|
| 1 | `u_tgt = v_t` 정확히 = **trajectory flow matching**. r 입력만 추가된 π0.5 BC loss | forward **1회** |
| (0,1) | v_t와 중간 s에서의 자기 예측을 혼합 (α=1/2면 Shortcut model) | forward 2회, 2번째는 stop-grad라 **backward 없음** |
| →0 | gradient가 MeanFlow와 일치. 타깃 `v_t − (t−r)·du/dt`에 **JVP 필요** | JVP |

**왜 커리큘럼인가**: 논문은 `L_MF = L_TFM + L_TC`로 분해되고 두 항의 gradient가 강하게 음의
상관이라 동시 최적화가 충돌함을 보인다. α를 1→0으로 어닐링하면 저분산 항을 먼저 맞추고
고분산 항으로 넘어간다.

## 이 리포에서의 두 가지 설계 결정

**1. JVP 단계는 기본 OFF** (`meanflow_jvp=False`). α=0은 VLA 전체를 통과하는 JVP가 필요한
유일한 구간이고, 레퍼런스 구현도 clamp 값에서 멈추는 모드(`discrete_training`)를 지원한다.
건너뛰면 1-step 품질을 조금 잃고 훨씬 싸고 단순한 학습 스텝을 얻는다 — `anneal` 이후의
1-step 품질이 부족할 때만 별도 phase-3로 돌린다.

**2. 커리큘럼은 한 run 안에서, 진행도(progress)의 함수로 돈다.** `train.py`가
`progress = step / num_train_steps`를 `wants_progress` 훅으로 loss까지 넘기므로,
`--num-train-steps`를 바꾸면 커리큘럼 전체가 같이 리스케일된다 (240-step 스모크와 60k 본 run이
같은 곡선을 그린다). 등록된 config은 하나다:

```
pi05_yam_lego_taxi_alphaflow   60k steps 기본, 단일 run
  스케줄 = 공식 alpha-Flow 레시피 그대로 (experiments-alphaflow.yaml):
    sigmoid(γ=25)가 run 전체 [0,1]에 걸쳐 1→0, 양끝 clamp 5e-3, fm_ratio 상수 0.5
  clamp가 3-phase를 스스로 만든다:
    progress 0–0.29    α=1 (TFM; zero-init r MLP 덕에 π0.5가 정확한 시작점)
    progress 0.29–0.71 α 어닐링 (중점 0.5에서 α=0.5)
    progress 0.71–1    α=5e-3 floor (discrete, JVP 없음 — reference의 discrete_training과 동일)
```

공식과의 유일한 의도적 차이: 공식은 clamp 밑에서 α→0(JVP MeanFlow 꼬리)로 가지만, 우리 기본은
floor=5e-3 discrete로 멈춘다 (`meanflow_jvp=False`). α=1 구간에도 두 번째(stop-grad, backward
없는) forward 비용을 낸다 — BC 스텝의 ~15%, run을 쪼개지 않는 값이다. JVP polish가 필요해지면
`Pi0AlphaFlowConfig(alpha_init=0, alpha_final=0, meanflow_jvp=True)`를 이 run 출력에 이어 붙인다.

## 비용이 왜 ~1.15배인가

두 번째 forward는 `z_s`에서의 **stop-grad 타깃**이라 backward가 없다. 그리고 prefix
(PaliGemma 2B, 이미지 토큰 ~800개)는 한 번만 돌고 KV 캐시를 두 suffix pass가 공유한다 —
추가 비용은 300M action expert의 forward 1회뿐이다 (`_prefix_forward` / `_u`).

## 검증 (실측)

CPU 단위테스트 10건 통과: α 스케줄·클램핑, α=1에서 타깃이 정확히 v_t, α→0에서 자기일관성 극한,
border(r=t) 행의 fallback, `z_s` 오일러 스텝, adaptive weight, phase별 `two_pass`, config 검증.
기존 `pi0_test`+`pi0_rlt_test` 포함 32 passed (무회귀).

GPU 스모크 (`scripts/alphaflow_smoke.py`, gemma_2b_lora/300m_lora, batch 2):

```
[1] max |u(z,t,t) - u(z,0,t)|      = 0.000e+00   r 조건 zero-init 확인
[2] max |pi05_ode - alphaflow_ode| = 0.000e+00   같은 모델, 확장이지 변경이 아님
[3] 1-step vs 10-step ODE spread   = 1.506e+00   샘플러가 실제로 평균속도를 씀
tfm      step 0/10000 : alpha=1.0000            finite
anneal   step 0       : alpha=1.0000            finite
         step 1000    : alpha=0.5025 (중점)      finite
         step 5000    : alpha=0.0050 (floor)     finite
meanflow step 0       : alpha=0.0000 (JVP)      finite
```

## 모니터링 함정: `loss`가 아니라 `delta2`를 봐라

MeanFlow의 adaptive weight 때문에 보고되는 loss는 `‖Δ‖²/(‖Δ‖²+ε)`, ε=1e-3 이라
`‖Δ‖² ≫ ε` 인 동안 **항상 ~1.0에 붙어 있다** (스모크에서 전 phase 0.9997). 학습이 안 되는 게
아니라 설계다 — stop-grad weight가 업데이트를 정규화된 gradient(≈ `d log‖Δ‖`)로 만든다.
실제 진척은 aux의 `delta2`(생 오차)와 `loss_tfm`(π0.5 BC loss와 직접 비교 가능)으로 본다.

## 열려 있는 것

- `time_pair` 기본값 `minmax`는 (t,r)을 π0.5의 Beta(1.5,1) 두 draw의 max/min으로 잡는다.
  1-step 품질에 결정적인 건 (t=1, r=0) 근처 질량이라 `scaled`(r = t·U)도 넣어뒀다 — 어느 쪽이
  YAM에서 나은지는 측정 대상이다.
- `fm_ratio=0.5` 상수는 공식 main run들(B/2, XL/2)의 값이다; cfg-B/2는 0.25를 썼다.
- 원스텝 변환이 액션 품질을 깎는지는 **BC-only 성공률로 먼저 검증**해야 한다. RL을 얹기 전에
  `pi05_yam_lego_taxi_alphaflow`의 1-step 정책이 현행 `pi05_yam_lego_taxi`(10-step)와 붙어야 한다.

## Deploy — 학습된 α-Flow 체크포인트 서빙/사용 예시

학습 산출물(200k run)은 `/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000`
(orbax; `assets/`에 norm stats 동봉). **`sample_actions`의 기본이 `num_steps=1`이라, 기존 π0.5 서빙
경로에 그대로 꽂으면 자동으로 1-step 추론**이 된다 — 클라이언트는 아무것도 몰라도 된다.

### A. Python API (오프라인 평가·스크립트)

```python
import numpy as np
from openpi.policies import policy_config
from openpi.training import config as train_config

ckpt = "/data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000"
cfg = train_config.get_config("pi05_yam_lego_taxi_alphaflow")

# 기본 = 1-step (게이트 판정 기준 demo-MSE 0.00096 — 같은 모델 10-step보다 낫다)
policy = policy_config.create_trained_policy(cfg, ckpt)

# few-step이 필요하면 (2-step이 self-gap 최소):
# policy = policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={"num_steps": 2})

obs = {
    "observation/image":       agentview_hwc_uint8,     # [H,W,3] uint8
    "observation/wrist_image": wrist_left_hwc_uint8,
    "observation/image_right": wrist_right_hwc_uint8,
    "observation/state":       state_42d.astype(np.float32),
    "prompt": "put the lego block into the taxi",
}
chunk = policy.infer(obs)["actions"]   # [30, 14] 로봇 공간 (자체 norm stats로 unnormalize됨)
```

주의: `num_steps`는 jit 아래에서 tracer로 전달되므로 커스텀 호출 시 정수 리터럴이든 배열이든 모두
동작한다(`lax.fori_loop`, bac7ac4). π0.5-동등 대조군이 필요하면 모델의 `sample_actions_ode(num_steps=10)`
— 같은 가중치를 r=t로 읽는 원래 ODE다.

### B. WebSocket 서빙 (실로봇/원격 클라이언트)

```bash
# 서버 (GPU 노드, SLURM로)
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_yam_lego_taxi_alphaflow \
  --policy.dir /data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000
# serve_policy는 sample_kwargs를 넘기지 않으므로 α-Flow 기본값 = 1-step 추론.
```

```python
# 클라이언트 (openpi-client) — 기존 π0.5 클라이언트 코드와 완전 동일
from openpi_client import websocket_client_policy
policy = websocket_client_policy.WebsocketClientPolicy(host="...", port=8000)
action_chunk = policy.infer(obs)["actions"]   # 지연시간만 ~10× 줄어든다
```

### C. RL 스택에서 (FQL actor 자리)

`Pi0AlphaFlow.sample_actions(rng, obs, num_steps=1, noise=z)` 가 FQL의 one-step actor 계약
(`z → chunk, 미분 가능 1-forward`)을 그대로 충족한다 — distill 타깃도 이 1-step forward로 대체 가능
(teacher ODE 비용 소멸). 결합 실험은 다음 사이클.
\n

### D. 다중 샘플링 (`num_samples`) — 그대로 적용

`MultiSamplePolicy`는 모델 무관이라 α-Flow에도 그대로 붙는다 — 클라이언트가 관측에
`{"num_samples": N}`을 실으면 `action_samples`(per-step, 실행 청크가 후보 0)로 N개가 돌아온다.
α-Flow에선 **샘플당 forward 1회**라 이 경로가 실용적이 된다: N=16이 π0.5 한 번 뽑는 비용의 ~1.6배.

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_yam_lego_taxi_alphaflow --policy.dir <ckpt> --num-samples 16
```

### E. 패치-크리틱 BoN / adaptive-commit 서빙 — 어댑터로 적용

`serve_patch_critic.py`(BoN/adaptive 배포)는 원래 Pi0RLT 전용 `extract_token_and_base_actions`
(N 후보를 한 backbone pass로)를 요구했다. α-Flow에 같은 계약의
**`sample_n_actions`**(prefix KV 1회 계산 → N 노이즈를 1-step expert로, KVCache의 batch axis=1에 tile)를
추가했고, `PatchCriticPolicy`가 모델에 따라 자동 선택한다(크리틱은 토큰을 안 쓰므로 무손실).
검증: 후보 0이 단일-샘플 경로와 **비트 일치**.

```bash
# α-Flow 1-step × patch-critic BoN/adaptive — BoN-16이 π0.5 draw ~1.6개 비용
uv run python scripts/serve_patch_critic.py \
  --config pi05_yam_lego_taxi_alphaflow \
  --checkpoint /data1/jellyho/acrft_ckpts/pi05_yam_lego_taxi_alphaflow/yam_alphaflow_200k/200000 \
  --critic .scratch/patch_critic_yam_s347_g5_pi05_cont \
  --mode adaptive --flow-steps 1 --port 8000
# 클라이언트: 관측에 {"critic_select": True, "num_samples": 16} — 기존 계약 그대로
```

주의: 크리틱 input_spec이 pi05-normalized joint delta = 샘플러 출력 그 자체이므로 변환이 없다
(`--flow-steps 1`이 α-Flow의 정상 모드; 10을 주면 mean-velocity 세분 샘플링).

