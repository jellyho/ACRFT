# ACRFT 실험 보고 · 게시 방법론

> **전 워커 공통 규칙의 단일 원천은 Space 리포의 `RULES.md`다**
> (https://huggingface.co/spaces/jellyho/acrft-reports/blob/main/RULES.md).
> 이 파일(CLAUDE.md)은 그 요약 + 이 리포 특화 절차이며, 충돌 시 RULES.md가 우선한다.
> RULES.md 개정 시 날짜·워커명을 남긴다.

이 저장소의 실험 보고는 **HF Space 허브(`jellyho/acrft-reports`, private)** 로 게시된다.
링크는 반드시 `https://huggingface.co/spaces/jellyho/acrft-reports` 를 쓴다
(`*.hf.space` 직링크는 private이라 익명 401).

## 단일 진실 원천: `slurm/make_master_report.py`

- 모든 리포트는 `ENTRIES` 리스트의 `entry(date, eid, title, status, body)` 한 건이다.
  body는 HTML 조각. 상태는 `완결` / `진행 중` / `살아있음`.
- **육하원칙**: 모든 entry는 `META[eid]`에 who/when(date)/where/what/how/why를 채운다.
  `_decorate()`가 표준 5W1H 헤더 표를 자동 삽입하므로 body에 중복 작성하지 않는다.
- **상호 연결 (필수 절차)**: 새 포스트를 쓸 때는 반드시 기존 eid 목록을 훑어 이 실험이
  근거로 삼거나(선행) 후속으로 잇는(파생) 포스트를 찾아 `META[eid]["links"]`에 채운다.
  스레드·관계도(force-directed 그래프)는 sync 때 links에서 **자동 생성**되므로,
  links만 정확하면 관계도가 스스로 자란다. 컬럼(국면) 배정이 없는 새 eid도 자동으로
  '신규' 색을 받아 노드가 된다.
- **date는 날짜+시간**(`YYYY-MM-DD HH:MM`, KST): 허브 목록·스레드가 시간순으로 흐른다.
  새 엔트리는 게시 시각을 적는다.
- 수치 표는 원본 JSON에서 **자동 재계산**한다(`run_level()`/`ci_row()` 패턴).
  손으로 옮겨 적은 숫자는 감사(audit)에서 어긋나기 쉬우니 금지.
- 빈 칸 금지: 데이터 미도착 칸은 "평가 대기 (사유/참조)"를 명시한다.
- **pre-commit 필수**: 훅이 `.git/hooks/pre-commit`에 설치되어 있다(ruff --fix / ruff-format /
  uv-lock). 커밋이 훅에서 실패하면 고치고 다시 커밋 — `--no-verify` 우회 금지. 훅 파이썬이
  miniconda 3.12를 잡으면 깨지므로 `uvx --python 3.11 pre-commit`으로 재설치한다.
- **이중 언어 필수**: 모든 엔트리는 `en(eid, title, body)`로 영어판을 함께 작성한다 —
  허브의 KO/EN 토글이 이를 렌더링한다. 수치 표·figure는 공유, 산문은 완역.
- **git 스탬프 필수**: 모든 게시는 branch@hash(+dirty)를 포함한다 — 5W1H 헤더의 '코드' 행에
  자동 삽입되고(`GIT_STAMP`), 허브 커밋 메시지에도 붙는다. 게시 전 커밋을 먼저 하는 습관을 권장
  (dirty 스탬프는 재현 불가능한 게시라는 뜻이다).

## 그림: `slurm/plot_style.py` + `slurm/make_figures.py`

- 스타일은 Seohong Park 논문 관례: 흰 배경, y-그리드만, 위/오른쪽 스파인 제거,
  seaborn deep 팔레트, 프레임 없는 legend, CI는 음영 밴드 또는 오차막대.
- **타이틀은 아주 간결하게** (예: "v11 demo-only"). 조건·n·프로토콜 등 추가 정보는
  리포트 본문 산문과 spec 표에 쓴다. regular weight (bold 금지).
- 모든 색은 legend/colorbar로 의미를 밝힌다.
- JSON 기반 그림은 `make_figures.py`가 리포트 생성 때마다 원본에서 재생성한다 —
  그림과 데이터가 어긋날 수 없게. GPU가 필요한 프로브 그림도 스크립트로 남겨
  언제든 재생성 가능해야 한다 (일회성 손그림 금지).

## 게시 흐름: `slurm/sync_hub.py`

```
uv run --no-sync python slurm/sync_hub.py
```

1. `make_master_report`를 import (→ 로컬 master_report.html + figure 재생성).
2. Space의 `index.html`에서 `const REPORTS = [...]`를 파싱, 기존 워커B 엔트리를 교체.
   **다른 워커의 엔트리는 절대 건드리지 않는다.**
3. 상단 고정 2건을 자동 생성: 🧵 데일리 스레드(일자별 포스트 다이제스트),
   🗺️ 마인드맵(국면 5컬럼 × links 간선 SVG, 노드 클릭 이동).
4. PR 생성 후 즉시 머지.

게시는 **Space 하나로 일원화** — Claude Artifact는 갱신하지 않는다 (2026-08-08 폐지).

비디오는 생성 즉시 HF dataset `jellyho/acrft-rollout-videos`(아카이브)와
Space `videos/`(갤러리 서빙)에 `upload_folder`로 올린다 — 배치 대기 금지.

## 주간 발표 (2026-08-23 추가)

실험은 **일주일마다 발표 형식으로 보고**한다. 청중이 결과를 이해하려면 결과 plot 앞에 배경이 그림으로
있어야 하므로, 실험마다 자료를 세 겹으로 준비한다:

1. **배경/설정 도식** — 환경이 어떻게 생겼는지, 무엇이 관측되고 무엇이 숨는지, 비교 대상이 무엇인지.
2. **기제 그림** — 왜 그 결과가 나오는지 (롤아웃 궤적, 커밋 타임라인, 크리틱의 값 프로파일 등).
3. **결과 그림** — 수치와 CI.

주간 묶음은 `slurm/make_weekly_deck.py`가 조립한다 — 그림은 `hub_figs/`에서, 수치는 결과 JSON에서
**다시 계산**하며 손으로 옮겨 적지 않는다. 개별 실험 리포트는 그때그때 허브에 게시하고, 주간 발표는
그것들을 하나의 서사로 엮는 별도 산출물이다.

## 대원칙

1. **처음 보는 사람이 처음부터 끝까지 이해할 수 있게 쓴다.** 모든 리포트는 사전 맥락
   없이 읽혀야 한다: 왜 이 실험을 했는지(동기), 무엇을 어떻게 했는지(설계), 무엇이
   나왔는지(결과), 그래서 결론이 무엇이고 다음이 무엇인지(판정·후속)를 그 안에서
   완결한다. 프로젝트 은어·이전 실험 참조는 링크와 한 줄 설명을 붙인다.
2. **엄밀하고 공정한 과학적 실험을 목표로 한다.** 비교군은 method-only-diff로 통제,
   판정 기준은 사전 등록, 통계는 run-level CI, 분류는 프로그램적으로. 부정적 결과와
   실패·사고도 같은 엄밀함으로 보고한다. 확정과 잠정을 구분해 표기한다.

## 보고 원칙

- **즉시성**: 새 정보(평가 도착, 원인 규명, 사고)가 생기면 그 사이클에 게시한다.
  "평가 JSON이 없다"는 재게시를 거를 이유가 아니다 — 조사 결과·수집 통계도 산출물이다.
- **전체 provenance**: 체크포인트·데이터·장면 풀·페어링·n을 항상 명시.
- 비교는 method-only-diff 체크포인트끼리만. 페어드 비교는 in-job vla와만.
- 판정은 run(시드)-level Δ̄ ± 95% t-CI. 잠정치는 "잠정(n=k)" 라벨을 단다.
- 분류(성공/실패/단계)는 항상 프로그램적으로 — 육안 분류 금지.
- 사고·버그도 리포트에 남긴다(가설 기각 사다리 표 형식이 좋다: td-segv 탭 참조).
