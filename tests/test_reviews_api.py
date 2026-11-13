import json
from collections.abc import Iterator
from functools import partial
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from openai import APIConnectionError
from pydantic import ValidationError

from app.core.config import Settings
from app.dependencies import get_reply_graph
from app.main import app
from app.services.reply_graph import (
    REPLY_PROMPT,
    FakeDraftChatModel,
    ReplyAssessment,
    build_draft_chain,
    build_reply_graph,
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FAKE_LLM", "true")
    monkeypatch.setattr("app.main.Settings", partial(Settings, _env_file=None))
    with TestClient(app) as test_client:
        yield test_client


def register_review(client: TestClient, rating: int = 2, content: str = "비동기 예제가 너무 빨랐어요.") -> dict:
    response = client.post(
        "/api/reviews",
        json={"studentName": "김수강", "rating": rating, "content": content},
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
    assert body["tone"] == "사과"
    assert body["needsFollowup"] is False
    fetched = client.get(path)
    assert fetched.status_code == 200
    assert fetched.json() == {**review, "replyDraft": body["reply"]}


@pytest.mark.parametrize(
    ("rating", "content", "tone", "needs_followup"),
    [(5, "실습 설명이 이해하기 쉬웠어요.", "감사", False), (2, "수강료 환불을 요청합니다.", "사과", True)],
)
def test_generated_reply_metadata(
    client: TestClient, rating: int, content: str, tone: str, needs_followup: bool
) -> None:
    review_id = register_review(client, rating, content)["reviewId"]

    response = client.post(f"/api/reviews/{review_id}/reply")

    assert response.status_code == 200
    assert response.json()["tone"] == tone
    assert response.json()["needsFollowup"] is needs_followup


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


@pytest.mark.parametrize("failure", ["connection", "parsing", "validation"])
def test_llm_failure_keeps_existing_draft(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    review_id = register_review(client)["reviewId"]
    path = f"/api/reviews/{review_id}"
    original = client.post(f"{path}/reply")
    assert original.status_code == 200
    original_draft = original.json()["reply"]

    failing_graph = AsyncMock()
    if failure == "connection":
        failing_graph.ainvoke.side_effect = APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1")
        )
    elif failure == "parsing":
        failing_graph.ainvoke.side_effect = OutputParserException("테스트용 출력 파싱 오류")
    else:
        with pytest.raises(ValidationError) as validation_error:
            ReplyAssessment(passed=True, reason="테스트", tone="응원", needs_followup=False)
        failing_graph.ainvoke.side_effect = validation_error.value
    monkeypatch.setitem(app.dependency_overrides, get_reply_graph, lambda: failing_graph)

    response = client.post(f"{path}/reply")

    assert response.status_code == 502
    assert response.json()["code"] == "AI_REQUEST_FAILED"
    assert client.get(path).json()["replyDraft"] == original_draft
    failing_graph.ainvoke.assert_awaited_once()


@pytest.mark.parametrize(("passed", "expected_attempts"), [(True, 1), (False, 2)])
def test_reply_graph_stops_at_pass_or_attempt_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, passed: bool, expected_attempts: int
) -> None:
    review_id = register_review(client)["reviewId"]
    path = f"/api/reviews/{review_id}"
    drafts = ["첫 답글 초안입니다.", "수정한 답글 초안입니다."]
    feedback = "비동기 예제에 대한 의견을 구체적으로 언급하세요."
    writer = AsyncMock()
    writer.ainvoke.side_effect = drafts
    reviewer = AsyncMock()
    reviewer.ainvoke.return_value = ReplyAssessment(
        passed=passed, reason=feedback, tone="사과", needs_followup=True
    )
    graph = build_reply_graph(writer, reviewer)
    monkeypatch.setitem(app.dependency_overrides, get_reply_graph, lambda: graph)

    response = client.post(f"{path}/reply")

    assert response.status_code == 200
    assert writer.ainvoke.await_count == reviewer.ainvoke.await_count == expected_attempts
    assert response.json() == {
        "reviewId": review_id,
        "reply": drafts[expected_attempts - 1],
        "model": "fake-llm",
        "tone": "사과",
        "needsFollowup": True,
    }
    assert client.get(path).json()["replyDraft"] == drafts[expected_attempts - 1]
    if not passed:
        revision_input = writer.ainvoke.await_args_list[1].args[0]
        assert revision_input["revision_messages"][0].content == f"직전 답글: {drafts[0]}\n검토 피드백: {feedback}"


def test_reply_prompt_keeps_review_in_human_message() -> None:
    content = "비동기 예제가 너무 빨랐어요."
    system, human = REPLY_PROMPT.format_messages(rating=2, content=content)

    assert system.type == "system"
    assert "강사" in system.content
    assert content not in system.content
    assert human.type == "human"
    assert human.content == f"별점: 2\n후기 본문: {content}"


def read_sse(body: str) -> list[tuple[str, dict[str, str | int | bool]]]:
    events = []
    for frame in body.strip().split("\n\n"):
        event_line, data_line = frame.splitlines()
        assert event_line.startswith("event: ") and data_line.startswith("data: ")
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def test_reply_stream_sends_tokens_and_saves_final_reply(client: TestClient) -> None:
    review_id = register_review(client)["reviewId"]
    path = f"/api/reviews/{review_id}"

    response = client.post(f"{path}/reply/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = read_sse(response.text)
    assert events[0] == ("attempt", {"attempt": 1})
    assert all(event == "token" for event, _ in events[1:-1])
    assert len(events[1:-1]) >= 2
    reply = "".join(data["text"] for _, data in events[1:-1])
    assert events[-1] == ("done", {"reply": reply, "tone": "사과", "needsFollowup": False})
    assert client.get(path).json()["replyDraft"] == reply


def test_reply_stream_for_missing_review_returns_json_404(client: TestClient) -> None:
    response = client.post("/api/reviews/99/reply/stream")

    assert response.status_code == 404
    assert response.json()["code"] == "REVIEW_NOT_FOUND"


def test_reply_stream_keeps_existing_draft_when_review_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_id = register_review(client)["reviewId"]
    path = f"/api/reviews/{review_id}"
    assert client.patch(f"{path}/reply", json={"replyDraft": "기존 초안"}).status_code == 200
    reviewer = AsyncMock()
    reviewer.ainvoke.side_effect = OutputParserException("테스트용 파싱 오류")
    graph = build_reply_graph(build_draft_chain(FakeDraftChatModel()), reviewer)
    monkeypatch.setitem(app.dependency_overrides, get_reply_graph, lambda: graph)

    response = client.post(f"{path}/reply/stream")

    assert response.status_code == 200
    events = read_sse(response.text)
    assert any(event == "token" for event, _ in events)
    assert events[-1][0] == "error"
    assert all(event != "done" for event, _ in events)
    assert client.get(path).json()["replyDraft"] == "기존 초안"
