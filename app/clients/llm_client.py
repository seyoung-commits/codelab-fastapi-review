from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from app.core.config import Settings


class ReplyInput(TypedDict):
    rating: int
    content: str


def create_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        timeout=30.0,
        max_retries=1,
        max_tokens=500,
        use_responses_api=True,
        store=False,
    )


async def _generate_fake_reply(review: ReplyInput) -> str:
    if review["rating"] <= 2:
        opening = "수업에서 불편을 드려 죄송합니다."
    else:
        opening = "수강 후기를 남겨주셔서 감사합니다."
    excerpt = review["content"][:80]
    return f"{opening} 남겨주신 의견(‘{excerpt}’)을 수업 준비에 참고하겠습니다."


def build_fake_reply_chain() -> RunnableLambda[ReplyInput, str]:
    return RunnableLambda(_generate_fake_reply)
