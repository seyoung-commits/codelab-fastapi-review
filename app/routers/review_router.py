import logging
from threading import current_thread
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.dependencies import ReplyServiceDep, ReviewServiceDep
from app.schemas.review import (
    ReviewCreateRequest,
    ReviewReplyResponse,
    ReviewReplyUpdateRequest,
    ReviewResponse,
)

router = APIRouter(prefix="/api/reviews", tags=["수강 후기"])
logger = logging.getLogger("uvicorn.error")


@router.post("", status_code=status.HTTP_201_CREATED, summary="수강 후기 등록")
def create_review(request: ReviewCreateRequest, service: ReviewServiceDep) -> ReviewResponse:
    return ReviewResponse.model_validate(service.create_review(request))


@router.get("", summary="수강 후기 목록 조회")
def list_reviews(
    service: ReviewServiceDep, rating: Annotated[int | None, Query(ge=1, le=5)] = None
) -> list[ReviewResponse]:
    return [ReviewResponse.model_validate(review) for review in service.list_reviews(rating)]


@router.get("/{review_id}", summary="수강 후기 조회")
def get_review(review_id: int, service: ReviewServiceDep) -> ReviewResponse:
    return ReviewResponse.model_validate(service.get_review(review_id))


@router.post("/{review_id}/reply", summary="AI 강사 답글 초안 생성")
async def create_reply(review_id: int, service: ReplyServiceDep) -> ReviewReplyResponse:
    logger.info("[후기 답글] reviewId=%s thread=%s", review_id, current_thread().name)
    return await service.create_reply(review_id)


@router.patch("/{review_id}/reply", summary="강사 답글 초안 수정")
def update_reply(
    review_id: int, request: ReviewReplyUpdateRequest, service: ReviewServiceDep
) -> ReviewResponse:
    return ReviewResponse.model_validate(service.save_reply_draft(review_id, request.reply_draft))
