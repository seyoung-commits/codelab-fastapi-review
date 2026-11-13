from langchain_openai import ChatOpenAI

from app.core.config import Settings


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
