from typing import Annotated

from fastapi import Depends, Request

from app.clients.llm_client import LlmClient
from app.repositories.review_repository import ReviewRepository
from app.services.reply_service import ReplyService
from app.services.review_service import ReviewService


def get_review_repository(request: Request) -> ReviewRepository:
    return request.app.state.review_repository


def get_llm_client(request: Request) -> LlmClient:
    return request.app.state.llm_client


def get_review_service(
    repository: Annotated[ReviewRepository, Depends(get_review_repository)],
) -> ReviewService:
    return ReviewService(repository)


ReviewServiceDep = Annotated[ReviewService, Depends(get_review_service)]


def get_reply_service(
    review_service: ReviewServiceDep,
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
) -> ReplyService:
    return ReplyService(review_service, llm_client)


ReplyServiceDep = Annotated[ReplyService, Depends(get_reply_service)]
