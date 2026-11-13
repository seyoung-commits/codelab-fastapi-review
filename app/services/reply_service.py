import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, contextmanager

from langchain_core.exceptions import ModelError, OutputParserException
from openai import OpenAIError
from pydantic import ValidationError

from app.core.exceptions import AiRequestError
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


class ReplyService:
    def __init__(
        self, review_service: ReviewService, reply_graph: ReplyGraph, model_name: str
    ) -> None:
        self._review_service = review_service
        self._reply_graph = reply_graph
        self._model_name = model_name

    async def create_reply(self, review_id: int) -> ReviewReplyResponse:
        review = self._review_service.get_review(review_id)
        reply_input: ReplyInput = {"rating": review.rating, "content": review.content}
        with _translate_ai_errors():
            result = await self._reply_graph.ainvoke(reply_input)
            response = self._build_response(review_id, result)
        self._review_service.save_reply_draft(review_id, response.reply)
        return response

    def stream_reply(self, review_id: int) -> AsyncIterator[str]:
        # 응답 헤더를 보내기 전에 후기를 조회한다.
        review = self._review_service.get_review(review_id)
        return self._stream_reply(review_id, {"rating": review.rating, "content": review.content})

    async def _stream_reply(self, review_id: int, reply_input: ReplyInput) -> AsyncIterator[str]:
        result: ReplyState | None = None
        try:
            with _translate_ai_errors():
                stream = self._reply_graph.astream(reply_input, stream_mode=["custom", "messages", "values"])
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
        if not {"reply", "tone", "needs_followup"}.issubset(result):
            raise OutputParserException(INCOMPLETE_REPLY_MESSAGE)
        return ReviewReplyResponse(
            review_id=review_id,
            reply=result["reply"],
            model=self._model_name,
            tone=result["tone"],
            needs_followup=result["needs_followup"],
        )
