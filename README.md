# 코딩 학원 수강 후기 API

저장된 수강 후기의 **별점과 본문**으로 AI가 강사 답글 초안을 작성합니다. 생성된 초안은 후기에 저장되며, 강사가 직접 수정할 수 있습니다.

**Python 3.14 · FastAPI · Pydantic · OpenAI Responses API · pytest**

## 답글 생성 흐름

```mermaid
flowchart LR
    A[후기 ID로 답글 생성 요청] --> B[저장된 별점과 본문 조회]
    B --> C[OpenAI 비동기 호출]
    C --> D[답글 초안 저장]
    D --> E[초안과 모델명 응답]
```

- **비동기 답글 생성** — 후기 등록·조회와 초안 수정은 `def`로 처리합니다. 답글 생성은 `async def`에서 `AsyncOpenAI`를 `await`하며, 서버 로그에서 `thread=MainThread`를 확인할 수 있습니다. [라우터](app/routers/review_router.py) · [LLM 클라이언트](app/clients/llm_client.py)
- **저장 기준** — AI 호출에 성공하면 새 초안을 저장합니다. 호출에 실패하면 `502`를 반환하고 기존 `replyDraft`를 유지합니다. [답글 서비스](app/services/reply_service.py)
- **테스트 가능한 LLM 구성** — FastAPI lifespan에서 OpenAI 또는 가짜 LLM을 주입합니다. 테스트에서는 LLM 장애를 재현해 오류 응답과 기존 초안 보존을 함께 검증합니다. [앱 초기화](app/main.py) · [API 테스트](tests/test_reviews_api.py)

후기와 초안은 인메모리 저장소에 보관되며 서버 재시작 시 초기화됩니다.

## API

| 메서드 | 경로 | 성공 | 오류 |
|---|---|---|---|
| `POST` | `/api/reviews` | `201` 후기 등록 | `422` 입력 검증 실패 |
| `GET` | `/api/reviews` | `200` 후기 목록, 선택적으로 `?rating=2` | `422` 유효하지 않은 별점 필터 |
| `GET` | `/api/reviews/{review_id}` | `200` 후기와 `replyDraft` | `404` 후기 없음 |
| `POST` | `/api/reviews/{review_id}/reply` | `200` 초안 생성·저장, `reviewId`·`reply`·`model` 반환 | `404` 후기 없음, `502` LLM 호출 실패 |
| `PATCH` | `/api/reviews/{review_id}/reply` | `200` 초안이 수정된 후기 | `404` 후기 없음, `422` 빈 초안 |

### 요청 예시

`POST /api/reviews`에 후기 한 건을 등록합니다.

```json
{
  "studentName": "김수강",
  "rating": 2,
  "content": "비동기 예제 진행이 빨라서 await가 실행되는 흐름을 따라가기 어려웠어요."
}
```

반환된 `reviewId`로 답글 생성 요청을 보내고, 후기 조회에서 저장된 `replyDraft`를 확인합니다. 답글 생성 요청의 본문은 비워 둡니다. 수정할 때는 `PATCH`에 `{"replyDraft": "수정할 답글"}`을 보냅니다.

## 실행 및 테스트

```bash
uv sync --locked
uv run fastapi dev
```

API 명세와 실행 예시는 [Swagger UI](http://127.0.0.1:8000/docs)에서 확인할 수 있습니다. API 키가 없거나 `FAKE_LLM=true`이면 가짜 LLM으로 실행합니다. OpenAI를 사용하려면 로컬 `.env`에 `OPENAI_API_KEY`를 설정합니다. 기본 모델은 `gpt-5.4-mini`이며 `OPENAI_MODEL`로 변경할 수 있습니다. [환경 변수 예시](.env.example)

```bash
uv run pytest -q
```
