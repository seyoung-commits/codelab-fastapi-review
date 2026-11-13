from typing import Annotated

from fastapi import Depends, Request
from langchain_core.runnables import Runnable

from app.clients.llm_client import ReplyDraft, ReplyInput
from app.repositories.review_repository import ReviewRepository
from app.services.reply_service import ReplyService
from app.services.review_service import ReviewService


def get_review_repository(request: Request) -> ReviewRepository:
    return request.app.state.review_repository


def get_reply_chain(request: Request) -> Runnable[ReplyInput, ReplyDraft]:
    return request.app.state.reply_chain


def get_llm_model(request: Request) -> str:
    return request.app.state.llm_model


def get_review_service(
    repository: Annotated[ReviewRepository, Depends(get_review_repository)],
) -> ReviewService:
    return ReviewService(repository)


ReviewServiceDep = Annotated[ReviewService, Depends(get_review_service)]


def get_reply_service(
    review_service: ReviewServiceDep,
    reply_chain: Annotated[Runnable[ReplyInput, ReplyDraft], Depends(get_reply_chain)],
    model_name: Annotated[str, Depends(get_llm_model)],
) -> ReplyService:
    return ReplyService(review_service, reply_chain, model_name)


ReplyServiceDep = Annotated[ReplyService, Depends(get_reply_service)]
