from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from weakref import WeakValueDictionary

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.clients.llm_client import create_chat_model
from app.core.config import Settings
from app.core.exceptions import register_exception_handlers
from app.repositories.review_repository import ReviewRepository
from app.routers.review_router import router
from app.services.policy_retriever import build_chroma_policy_retriever, build_fake_policy_retriever
from app.services.reply_graph import build_fake_reply_graph, build_openai_reply_graph


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.review_repository = ReviewRepository()
    app.state.reply_locks = WeakValueDictionary()

    if settings.fake_llm or not settings.openai_api_key:
        app.state.policy_retriever = build_fake_policy_retriever()
        app.state.reply_graph = build_fake_reply_graph(app.state.policy_retriever)
        app.state.llm_model = "fake-llm"
    else:
        app.state.policy_retriever = await run_in_threadpool(build_chroma_policy_retriever, settings)
        app.state.reply_graph = build_openai_reply_graph(create_chat_model(settings), app.state.policy_retriever)
        app.state.llm_model = settings.openai_model

    print(f"[lifespan] 학원 규정 검색과 답글 그래프 초기화 완료 (모델: {app.state.llm_model})", flush=True)
    yield


app = FastAPI(title="코딩 학원 수강 후기 API", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(router)
