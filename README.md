# 코딩 학원 수강 후기 API

수강 후기를 등록하고 조회합니다. 저장된 후기의 별점과 본문을 바탕으로 AI가 강사 답글 초안을 작성합니다. 후기와 답글 초안은 메모리에 저장되며 서버를 재시작하면 초기화됩니다.

## 실행

Python 3.14와 uv를 사용합니다. 프로젝트 폴더에서 다음 명령을 실행한 뒤 [Swagger UI](http://127.0.0.1:8000/docs)를 엽니다.

```powershell
uv sync --locked
uv run fastapi dev
```

API 키가 없으면 가짜 LLM으로 실행됩니다. 키가 있는 환경에서 가짜 LLM으로 실행하려면 서버를 시작하기 전에 PowerShell에서 `$env:FAKE_LLM = "true"`를 설정합니다.

OpenAI를 사용하려면 `.env.example`을 참고해 로컬 `.env`의 `OPENAI_API_KEY`에 키를 설정하고 `FAKE_LLM=false`로 실행합니다. PowerShell에서 `FAKE_LLM`을 설정했다면 환경 변수 값이 `.env`보다 우선합니다. 모델은 `OPENAI_MODEL`로 지정할 수 있으며 기본값은 `gpt-5.4-mini`입니다. `.env`는 Git에서 제외됩니다.

## API 확인

| 메서드 | 경로 | 성공 | 오류 |
|---|---|---|---|
| `POST` | `/api/reviews` | `201`과 등록된 후기 | 별점이 1~5 범위를 벗어나면 `422` |
| `GET` | `/api/reviews?rating=2` | `200`과 해당 별점의 후기 목록. `rating`을 생략하면 전체 목록 | 잘못된 별점 필터는 `422` |
| `GET` | `/api/reviews/{review_id}` | `200`과 `replyDraft`가 포함된 후기 | 없는 번호면 `404` |
| `POST` | `/api/reviews/{review_id}/reply` | 초안 저장 후 `200`과 `reviewId`, `reply`, `model` | 없는 번호면 `404`, LLM 호출 실패 시 `502` |
| `PATCH` | `/api/reviews/{review_id}/reply` | `200`과 초안이 수정된 후기 | 없는 번호면 `404`, 빈 초안은 `422` |

Swagger UI 또는 Postman에서 `POST /api/reviews`에 다음 JSON을 보냅니다.

```json
{
  "studentName": "김수강",
  "rating": 2,
  "content": "비동기 예제 진행이 너무 빨라서 따라가기 어려웠어요."
}
```

응답에서 받은 `reviewId`로 조회와 답글 생성을 호출합니다. 답글 생성 요청에는 본문이 필요 없습니다. 낮은 별점의 답글은 사과로 시작하며, 서버 터미널에서 `thread=MainThread`를 확인할 수 있습니다.

저장된 초안은 후기 조회 응답의 `replyDraft`에서 확인합니다. 초안이 없는 후기는 `null`을 반환합니다. 답글을 다시 생성하면 새 초안이 기존 초안을 대체하고, 생성에 실패하면 기존 초안이 유지됩니다.

강사가 초안을 직접 수정하려면 `PATCH /api/reviews/{review_id}/reply`에 다음 JSON을 보냅니다.

```json
{
  "replyDraft": "후기를 남겨주셔서 감사합니다. 다음 수업에서 비동기 예제를 단계별로 복습하겠습니다."
}
```

## 자동 테스트

```powershell
uv run pytest -q
```

가짜 LLM으로 초안 생성·저장·수정, 별점 필터, 입력 검증, LLM 호출 실패 후 초안 유지를 검증합니다.
