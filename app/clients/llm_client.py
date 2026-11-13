from typing import Literal, TypedDict

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings


class ReplyInput(TypedDict):
    rating: int
    content: str


class ReplyDraft(BaseModel):
    reply: str = Field(description="수강생에게 보낼 정중한 한국어 답글 초안 2~3문장")
    tone: Literal["사과", "감사", "안내"] = Field(
        description="답글의 주된 말투. 별점 1~2점은 사과, 5점 칭찬은 감사, 절차 설명이 중심이면 안내"
    )
    needs_followup: bool = Field(
        description="환불, 수강 취소, 반 변경, 결강, 수업 미진행, 강사 태도, 개인정보 등 학원 담당자의 확인이나 조치가 필요한지 여부"
    )


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


async def _generate_fake_reply(review: ReplyInput) -> ReplyDraft:
    if review["rating"] <= 2:
        opening = "수업에서 불편을 드려 죄송합니다."
        tone = "사과"
    else:
        opening = "수강 후기를 남겨주셔서 감사합니다."
        tone = "감사"
    content = review["content"]
    excerpt = content[:80]
    followup_keywords = ("환불", "수강 취소", "반 변경", "결강", "수업 미진행", "강사 태도", "개인정보")
    return ReplyDraft(
        reply=f"{opening} 남겨주신 의견(‘{excerpt}’)을 수업 준비에 참고하겠습니다.",
        tone=tone,
        needs_followup=any(keyword in content for keyword in followup_keywords),
    )


def build_fake_reply_chain() -> RunnableLambda[ReplyInput, ReplyDraft]:
    return RunnableLambda(_generate_fake_reply)
