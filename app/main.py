from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.llm_client import create_chat_model
from app.core.config import Settings
from app.core.exceptions import register_exception_handlers
from app.repositories.review_repository import ReviewRepository
from app.routers.review_router import router
from app.services.reply_graph import build_fake_reply_graph, build_openai_reply_graph


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.review_repository = ReviewRepository()

    if settings.fake_llm or not settings.openai_api_key:
        app.state.reply_graph = build_fake_reply_graph()
        app.state.llm_model = "fake-llm"
    else:
        app.state.reply_graph = build_openai_reply_graph(create_chat_model(settings))
        app.state.llm_model = settings.openai_model

    print(f"[lifespan] 답글 그래프 초기화 완료 (모델: {app.state.llm_model})", flush=True)
    yield


app = FastAPI(title="코딩 학원 수강 후기 API", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(router)
