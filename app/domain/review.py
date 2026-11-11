from dataclasses import dataclass


@dataclass
class Review:
    review_id: int
    student_name: str
    rating: int
    content: str
    reply_draft: str | None = None
