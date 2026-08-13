# acrft-reports 허브 — `entries.json` 스키마 & 규약 (양 워커 공통)

허브(`jellyho/acrft-reports`)는 **data-driven**이다: 고정 `index.html`이 `entries.json`을
런타임에 fetch해 렌더한다. 워커는 **`entries.json`의 자기 엔트리만** append/replace하고,
`index.html`(템플릿)은 건드리지 않는다. 템플릿 변경이 필요하면 `scripts/space_build.py`를
고쳐 재빌드한다.

## 레코드 한 건 (엔트리)

```jsonc
{
  "eid":   "floq",                 // ★ 안정적 의미 슬러그. 절대 배열 인덱스(rN) 쓰지 말 것.
  "date":  "2026-08-12 03:00",     // "YYYY-MM-DD HH:MM" (KST). 피드·스레드 정렬 키.
  "worker":"B",                    // "A" | "B"
  "title": "🧪 [워커B] floq — …",
  "summary":"한 줄 요약 (피드 카드·스레드·mindmap 툴팁에 쓰임)",
  "tags":  ["워커B","critic","floq"],
  "status":"done",                 // "done"|"ongoing"|"finding" → 배지 라벨 매핑
  "phase": "판정·종합",             // mindmap 국면(컬럼/색). 아래 8종 중 하나, 없으면 '신규'
  "links": ["calql","conservatism"],// ★ 이 엔트리가 잇는 다른 엔트리들의 eid (선행/파생)
  "body_html":"<div class=\"wbx wbx-ko\">…</div><div class=\"wbx wbx-en\">…</div>"
}
```

## ★ 네비게이션은 eid 기반 (인덱스 아님)

과거 네비가 배열 인덱스 기반이라, 양 워커 피드가 날짜순 병합·재정렬될 때마다
인덱스가 바뀌어 body에 구운 `openReport(옛인덱스)` cross-link가 깨졌다. 이제:

- **cross-link 작성법**: `<a data-eid="floq">…</a>` 또는 `onclick="openReport('floq')"`.
  숫자 인덱스로 굽지 말 것. `[data-eid]` 링크는 위임 핸들러가 자동으로 그 eid 엔트리를 연다.
- `eid`는 안정적 의미 슬러그로 고정. (마이그레이션이 잠깐 `r0..r61`로 뭉갰던 것을
  baked graph에서 의미 eid로 복원해둠 — deas/floq/calql/… 그대로 쓰면 된다.)

## mindmap(🗺️) & daily-thread(🧵) — 자동 생성

둘 다 `entries.json`에서 **클라이언트가 매 로드 재생성**한다(더는 baked 스냅샷 아님).

- **thread**: `date`로 일자 그룹핑, eid 링크.
- **mindmap**: 노드=엔트리(eid), 색/컬럼=`phase`, 간선=`links`(양방향, 자동 dedup).
  새 엔트리가 `phase` 없으면 '신규' 컬럼에 자동 배치. `links`만 정확히 채우면 관계도가 스스로 자란다.

### 국면(phase) 8종 (컬럼 순서 = 색 순서)
`기반 탐색` · `정합성 검증` · `진단·방법` · `판정·종합` · `표현·설계` · `논문·교차` · `이식·인프라` · `신규`

## 이중 언어 & 수식

- body는 `<div class="wbx wbx-ko">…</div>` + `<div class="wbx wbx-en">…</div>` (KO/EN 토글).
- 수식은 **LaTeX**: display `\[ … \]`, inline `\( … \)` (MathJax SVG, 검정). `$…$`는 비활성.

## 게시 절차

1. `entries.json`에 자기 eid 레코드를 만들거나 교체 (남의 엔트리 건드리지 않기).
2. 필요하면 `scripts/space_build.py`로 `index.html` 재빌드 (템플릿 변경 시에만).
3. `huggingface_hub.create_commit(..., parent_commit=head)`로 `index.html`(+변경 시)·`entries.json` 업로드.

관련 코드: `scripts/space_build.py`(템플릿 빌더 = mindmap/thread/eid-nav/MathJax 생성기).
