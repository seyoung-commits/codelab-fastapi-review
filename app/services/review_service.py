from app.core.exceptions import ReviewNotFoundError
from app.domain.review import Review
from app.repositories.review_repository import ReviewRepository
from app.schemas.review import ReviewCreateRequest


class ReviewService:
    def __init__(self, repository: ReviewRepository) -> None:
        self._repository = repository

    def create_review(self, request: ReviewCreateRequest) -> Review:
        return self._repository.create(request.student_name, request.rating, request.content)

    def get_review(self, review_id: int) -> Review:
        review = self._repository.find_by_id(review_id)
        if review is None:
            raise ReviewNotFoundError(review_id)
        return review

    def list_reviews(self, rating: int | None = None) -> list[Review]:
        return self._repository.find_all(rating)

    def save_reply_draft(self, review_id: int, draft: str) -> Review:
        review = self._repository.update_reply_draft(review_id, draft)
        if review is None:
            raise ReviewNotFoundError(review_id)
        return review
