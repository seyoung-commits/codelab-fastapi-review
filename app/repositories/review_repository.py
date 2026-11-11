from dataclasses import replace
from threading import Lock

from app.domain.review import Review


class ReviewRepository:
    def __init__(self) -> None:
        self._reviews: dict[int, Review] = {}
        self._next_id = 1
        self._lock = Lock()

    def create(self, student_name: str, rating: int, content: str) -> Review:
        # 번호 발급과 저장을 한 번에 처리해 동시 등록에서도 번호가 겹치지 않는다.
        with self._lock:
            review = Review(self._next_id, student_name, rating, content)
            self._reviews[review.review_id] = review
            self._next_id += 1
        return review

    def save(self, review: Review) -> Review:
        with self._lock:
            self._reviews[review.review_id] = review
            self._next_id = max(self._next_id, review.review_id + 1)
        return review

    def find_by_id(self, review_id: int) -> Review | None:
        with self._lock:
            return self._reviews.get(review_id)

    def find_all(self, rating: int | None = None) -> list[Review]:
        with self._lock:
            reviews = list(self._reviews.values())
        if rating is None:
            return reviews
        return [review for review in reviews if review.rating == rating]

    def update_reply_draft(self, review_id: int, draft: str) -> Review | None:
        with self._lock:
            review = self._reviews.get(review_id)
            if review is None:
                return None
            updated_review = replace(review, reply_draft=draft)
            self._reviews[review_id] = updated_review
            return updated_review
