from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from app.clients.llm_client import FakeLlmClient, OpenAiLlmClient
from app.core.config import Settings
from app.core.exceptions import register_exception_handlers
from app.repositories.review_repository import ReviewRepository
from app.routers.review_router import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.review_repository = ReviewRepository()

    if settings.fake_llm or not settings.openai_api_key:
        app.state.llm_client = FakeLlmClient()
        yield
        return

    async with AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(), timeout=30.0) as client:
        app.state.llm_client = OpenAiLlmClient(client, settings.openai_model)
        yield


app = FastAPI(title="코딩 학원 수강 후기 API", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(router)
