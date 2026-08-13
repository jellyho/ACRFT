# 실험 리포트 허브

`index.html`을 브라우저로 열면 전체 실험 리포트를 날짜·태그·검색으로 열람할 수 있다.

## 새 리포트 추가 규약
1. 리포트를 **자기완결 HTML**(인라인 CSS)로 `docs/reports/YYYY-MM-DD_slug.html`에 저장한다.
2. `index.html`의 `REPORTS` 배열 **맨 앞**에 항목을 추가한다:
   `{date, title, summary, tags, file, status}` — status는 `done`/`ongoing`/`finding`.
3. 사용자에게는 항상 파일 아티팩트로 전달한다 (개별 리포트 + 필요 시 index).

상세 수치·인용이 필요한 리포트는 같은 이름의 `.md`를 `docs/`에 병행 유지한다
(예: `docs/overnight_report_2026-08-07.md`).
