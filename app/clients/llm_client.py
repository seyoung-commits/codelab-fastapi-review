import json
from typing import Protocol

from openai import AsyncOpenAI, OpenAIError

from app.core.exceptions import AiRequestError


class LlmClient(Protocol):
    @property
    def model(self) -> str: ...

    async def generate_reply(self, instructions: str, prompt: str) -> str: ...


class OpenAiLlmClient:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self.model = model

    async def generate_reply(self, instructions: str, prompt: str) -> str:
        try:
            response = await self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=prompt,
                max_output_tokens=500,
                store=False,
            )
        except OpenAIError as error:
            raise AiRequestError("AI 서비스에서 답글을 생성하지 못했습니다.") from error

        reply = response.output_text.strip()
        if response.status != "completed" or not reply:
            raise AiRequestError("AI 서비스가 완성된 답글을 반환하지 않았습니다.")
        return reply


class FakeLlmClient:
    model = "fake-llm"

    async def generate_reply(self, instructions: str, prompt: str) -> str:
        review = json.loads(prompt)
        if review["rating"] <= 2:
            opening = "수업에서 불편을 드려 죄송합니다."
        else:
            opening = "수강 후기를 남겨주셔서 감사합니다."
        excerpt = review["content"][:80]
        return f"{opening} 남겨주신 의견(‘{excerpt}’)을 수업 준비에 참고하겠습니다."
