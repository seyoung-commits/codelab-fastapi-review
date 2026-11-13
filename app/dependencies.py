from typing import Annotated

from fastapi import Depends, Request
from langchain_core.runnables import Runnable

from app.repositories.review_repository import ReviewRepository
from app.services.reply_graph import ReplyInput, ReplyState
from app.services.reply_service import ReplyService
from app.services.review_service import ReviewService


def get_review_repository(request: Request) -> ReviewRepository:
    return request.app.state.review_repository


def get_reply_graph(request: Request) -> Runnable[ReplyInput, ReplyState]:
    return request.app.state.reply_graph


def get_llm_model(request: Request) -> str:
    return request.app.state.llm_model


def get_review_service(
    repository: Annotated[ReviewRepository, Depends(get_review_repository)],
) -> ReviewService:
    return ReviewService(repository)


ReviewServiceDep = Annotated[ReviewService, Depends(get_review_service)]


def get_reply_service(
    review_service: ReviewServiceDep,
    reply_graph: Annotated[Runnable[ReplyInput, ReplyState], Depends(get_reply_graph)],
    model_name: Annotated[str, Depends(get_llm_model)],
) -> ReplyService:
    return ReplyService(review_service, reply_graph, model_name)


ReplyServiceDep = Annotated[ReplyService, Depends(get_reply_service)]
