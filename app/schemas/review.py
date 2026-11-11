from typing import Annotated

from pydantic import ConfigDict, Field, StringConstraints

from app.schemas.base import CamelModel


class ReviewCreateRequest(CamelModel):
    student_name: str
    rating: int = Field(ge=1, le=5)
    content: str


class ReviewResponse(CamelModel):
    model_config = ConfigDict(from_attributes=True)

    review_id: int
    student_name: str
    rating: int
    content: str
    reply_draft: str | None = None


class ReviewReplyUpdateRequest(CamelModel):
    reply_draft: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ReviewReplyResponse(CamelModel):
    review_id: int
    reply: str
    model: str
