from collections.abc import Iterator
from functools import partial
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from app.core.config import Settings
from app.dependencies import get_reply_chain
from app.main import app
from app.services.reply_service import REPLY_PROMPT


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FAKE_LLM", "true")
    monkeypatch.setattr("app.main.Settings", partial(Settings, _env_file=None))
    with TestClient(app) as test_client:
        yield test_client


def register_review(client: TestClient, rating: int = 2) -> dict:
    response = client.post(
        "/api/reviews",
        json={"studentName": "김수강", "rating": rating, "content": "비동기 예제가 너무 빨랐어요."},
    )
    assert response.status_code == 201
    return response.json()


def test_generated_reply_is_saved_with_review(client: TestClient) -> None:
    review = register_review(client)
    assert review["replyDraft"] is None

    path = f"/api/reviews/{review['reviewId']}"
    response = client.post(f"{path}/reply")

    assert response.status_code == 200
    body = response.json()
    assert body["reviewId"] == review["reviewId"]
    assert body["model"] == "fake-llm"
    assert review["content"] in body["reply"]
    assert body["reply"].startswith("수업에서 불편을 드려 죄송합니다.")
    fetched = client.get(path)
    assert fetched.status_code == 200
    assert fetched.json() == {**review, "replyDraft": body["reply"]}


def test_reviews_can_be_filtered_by_rating(client: TestClient) -> None:
    assert client.get("/api/reviews").json() == []
    low_rating_review = register_review(client, rating=2)
    high_rating_review = register_review(client, rating=5)

    response = client.get("/api/reviews")
    assert response.status_code == 200
    assert response.json() == [low_rating_review, high_rating_review]
    assert client.get("/api/reviews", params={"rating": 2}).json() == [low_rating_review]
    assert client.get("/api/reviews", params={"rating": 3}).json() == []


@pytest.mark.parametrize("rating", [0, 6, "invalid"])
def test_invalid_rating_filter_returns_422(client: TestClient, rating: int | str) -> None:
    assert client.get("/api/reviews", params={"rating": rating}).status_code == 422


def test_instructor_can_edit_generated_reply(client: TestClient) -> None:
    review = register_review(client)
    path = f"/api/reviews/{review['reviewId']}"
    assert client.post(f"{path}/reply").status_code == 200
    edited_reply = "다음 시간에 비동기 예제를 다시 설명하겠습니다."

    response = client.patch(f"{path}/reply", json={"replyDraft": edited_reply})

    assert response.status_code == 200
    expected_review = {**review, "replyDraft": edited_reply}
    assert response.json() == expected_review
    assert client.get(path).json() == expected_review


@pytest.mark.parametrize("body", [{}, {"replyDraft": ""}, {"replyDraft": "  \t"}])
def test_edit_requires_reply_text(client: TestClient, body: dict) -> None:
    review_id = register_review(client)["reviewId"]

    response = client.patch(f"/api/reviews/{review_id}/reply", json=body)

    assert response.status_code == 422


def test_editing_missing_review_returns_404(client: TestClient) -> None:
    response = client.patch("/api/reviews/99/reply", json={"replyDraft": "감사합니다."})

    assert response.status_code == 404
    assert response.json()["code"] == "REVIEW_NOT_FOUND"


def test_llm_failure_keeps_existing_draft(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    review_id = register_review(client)["reviewId"]
    path = f"/api/reviews/{review_id}"
    original = client.post(f"{path}/reply")
    assert original.status_code == 200
    original_draft = original.json()["reply"]

    failing_chain = AsyncMock()
    failing_chain.ainvoke.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1")
    )
    monkeypatch.setitem(app.dependency_overrides, get_reply_chain, lambda: failing_chain)

    response = client.post(f"{path}/reply")

    assert response.status_code == 502
    assert response.json()["code"] == "AI_REQUEST_FAILED"
    assert client.get(path).json()["replyDraft"] == original_draft
    failing_chain.ainvoke.assert_awaited_once()


def test_reply_prompt_keeps_review_in_human_message() -> None:
    content = "비동기 예제가 너무 빨랐어요."
    system, human = REPLY_PROMPT.format_messages(rating=2, content=content)

    assert system.type == "system"
    assert "강사" in system.content
    assert content not in system.content
    assert human.type == "human"
    assert human.content == f"별점: 2\n후기 본문: {content}"
