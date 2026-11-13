import json
import logging
from asyncio import Lock
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, contextmanager
from weakref import WeakValueDictionary

from langchain_core.exceptions import ModelError, OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from openai import OpenAIError
from pydantic import ValidationError

from app.core.exceptions import AiRequestError, ReplyNotGeneratedError
from app.schemas.review import ReviewReplyResponse
from app.services.reply_graph import INCOMPLETE_REPLY_MESSAGE, ReplyGraph, ReplyInput, ReplyState
from app.services.review_service import ReviewService

logger = logging.getLogger(__name__)


@contextmanager
def _translate_ai_errors() -> Iterator[None]:
    try:
        yield
    except (OutputParserException, ValidationError) as error:
        logger.warning("답글 파싱 실패 (%s)", type(error).__name__)
        raise AiRequestError(INCOMPLETE_REPLY_MESSAGE) from error
    except (OpenAIError, ModelError, ValueError) as error:
        logger.warning("AI 답글 생성 실패 (%s)", type(error).__name__)
        raise AiRequestError("AI 서비스에서 답글을 생성하지 못했습니다.") from error


def _sse(event: str, data: dict[str, str | int | bool]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _generation_input(rating: int, content: str) -> ReplyInput:
    return {
        "rating": rating,
        "content": content,
        "reply": None,
        "attempts": 0,
        "feedback": None,
        "revision_request": None,
        "passed": False,
        "tone": None,
        "needs_followup": None,
    }


def _thread_config(review_id: int) -> RunnableConfig:
    return {"configurable": {"thread_id": str(review_id)}}


def _has_completed_reply(snapshot: StateSnapshot) -> bool:
    return (
        not snapshot.next
        and bool(snapshot.values.get("reply"))
        and snapshot.values.get("tone") is not None
        and snapshot.values.get("needs_followup") is not None
    )


class ReplyService:
    def __init__(
        self,
        review_service: ReviewService,
        reply_graph: ReplyGraph,
        model_name: str,
        reply_locks: WeakValueDictionary[int, Lock],
    ) -> None:
        self._review_service = review_service
        self._reply_graph = reply_graph
        self._model_name = model_name
        self._reply_locks = reply_locks

    async def create_reply(self, review_id: int) -> ReviewReplyResponse:
        async with self._reply_locks.setdefault(review_id, Lock()):
            review = self._review_service.get_review(review_id)
            reply_input = _generation_input(review.rating, review.content)
            return await self._run_and_save(review_id, reply_input, _thread_config(review_id))

    async def revise_reply(self, review_id: int, instruction: str) -> ReviewReplyResponse:
        async with self._reply_locks.setdefault(review_id, Lock()):
            review = self._review_service.get_review(review_id)
            if review.reply_draft is None:
                raise ReplyNotGeneratedError("먼저 AI 답글을 생성해 주세요.")
            config = _thread_config(review_id)
            checkpoint = await self._completed_checkpoint(config)
            reply_input: ReplyInput = {"revision_request": instruction}
            if review.reply_draft != checkpoint.values["reply"]:
                reply_input["reply"] = review.reply_draft
            return await self._run_and_save(review_id, reply_input, checkpoint.config)

    async def _completed_checkpoint(self, config: RunnableConfig) -> StateSnapshot:
        latest = await self._reply_graph.aget_state(config)
        if _has_completed_reply(latest):
            return latest
        # 실패한 실행의 중간 상태가 있으면 직전에 완료된 초안을 찾아 복원한다.
        async with aclosing(self._reply_graph.aget_state_history(config)) as history:
            async for snapshot in history:
                if _has_completed_reply(snapshot):
                    return snapshot
        raise ReplyNotGeneratedError("먼저 AI 답글을 생성해 주세요.")

    async def _run_and_save(
        self, review_id: int, reply_input: ReplyInput, config: RunnableConfig
    ) -> ReviewReplyResponse:
        with _translate_ai_errors():
            result = await self._reply_graph.ainvoke(reply_input, config=config, durability="exit")
            response = self._build_response(review_id, result)
        self._review_service.save_reply_draft(review_id, response.reply)
        return response

    def stream_reply(self, review_id: int) -> AsyncIterator[str]:
        # 응답 헤더를 보내기 전에 후기를 조회한다.
        review = self._review_service.get_review(review_id)
        return self._stream_reply(review_id, _generation_input(review.rating, review.content))

    async def _stream_reply(self, review_id: int, reply_input: ReplyInput) -> AsyncIterator[str]:
        async with self._reply_locks.setdefault(review_id, Lock()):
            async with aclosing(self._stream_events(review_id, reply_input)) as events:
                async for event in events:
                    yield event

    async def _stream_events(self, review_id: int, reply_input: ReplyInput) -> AsyncIterator[str]:
        result: ReplyState | None = None
        try:
            with _translate_ai_errors():
                stream = self._reply_graph.astream(
                    reply_input,
                    config=_thread_config(review_id),
                    stream_mode=["custom", "messages", "values"],
                    durability="exit",
                )
                async with aclosing(stream):
                    async for mode, data in stream:
                        if mode == "custom":
                            yield _sse("attempt", data)
                        elif mode == "messages":
                            message, metadata = data
                            if metadata.get("langgraph_node") == "draft" and message.text:
                                yield _sse("token", {"text": message.text})
                        elif mode == "values":
                            result = data
                if result is None:
                    raise OutputParserException(INCOMPLETE_REPLY_MESSAGE)
                response = self._build_response(review_id, result)
        except AiRequestError as error:
            yield _sse("error", {"message": str(error)})
            return

        self._review_service.save_reply_draft(review_id, response.reply)
        yield _sse("done", {"reply": response.reply, "tone": response.tone, "needsFollowup": response.needs_followup})

    def _build_response(self, review_id: int, result: ReplyState) -> ReviewReplyResponse:
        reply = result.get("reply")
        tone = result.get("tone")
        needs_followup = result.get("needs_followup")
        if not reply or tone is None or needs_followup is None:
            raise OutputParserException(INCOMPLETE_REPLY_MESSAGE)
        return ReviewReplyResponse(
            review_id=review_id,
            reply=reply,
            model=self._model_name,
            tone=tone,
            needs_followup=needs_followup,
        )
