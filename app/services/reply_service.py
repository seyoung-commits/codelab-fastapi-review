import logging

from langchain_core.exceptions import ModelError, OutputParserException
from langchain_core.runnables import Runnable
from openai import OpenAIError
from pydantic import ValidationError

from app.core.exceptions import AiRequestError
from app.schemas.review import ReviewReplyResponse
from app.services.reply_graph import INCOMPLETE_REPLY_MESSAGE, ReplyInput, ReplyState
from app.services.review_service import ReviewService

logger = logging.getLogger(__name__)


class ReplyService:
    def __init__(
        self, review_service: ReviewService, reply_graph: Runnable[ReplyInput, ReplyState], model_name: str
    ) -> None:
        self._review_service = review_service
        self._reply_graph = reply_graph
        self._model_name = model_name

    async def create_reply(self, review_id: int) -> ReviewReplyResponse:
        review = self._review_service.get_review(review_id)
        reply_input: ReplyInput = {"rating": review.rating, "content": review.content}
        try:
            result = await self._reply_graph.ainvoke(reply_input)
        except (OutputParserException, ValidationError) as error:
            logger.warning("답글 파싱 실패 (%s)", type(error).__name__)
            raise AiRequestError(INCOMPLETE_REPLY_MESSAGE) from error
        except (OpenAIError, ModelError, ValueError) as error:
            logger.warning("AI 답글 생성 실패 (%s)", type(error).__name__)
            raise AiRequestError("AI 서비스에서 답글을 생성하지 못했습니다.") from error
        response = ReviewReplyResponse(
            review_id=review.review_id,
            reply=result["reply"],
            model=self._model_name,
            tone=result["tone"],
            needs_followup=result["needs_followup"],
        )
        self._review_service.save_reply_draft(review_id, response.reply)
        return response
