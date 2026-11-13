from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ReviewNotFoundError(Exception):
    def __init__(self, review_id: int) -> None:
        self.review_id = review_id
        super().__init__(f"해당 후기가 존재하지 않습니다. reviewId={review_id}")


class AiRequestError(Exception):
    pass


class ReplyNotGeneratedError(Exception):
    pass


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ReviewNotFoundError)
    async def handle_review_not_found(
        _request: Request, error: ReviewNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"code": "REVIEW_NOT_FOUND", "detail": str(error)},
        )

    @app.exception_handler(AiRequestError)
    async def handle_ai_request_error(_request: Request, error: AiRequestError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"code": "AI_REQUEST_FAILED", "detail": str(error)},
        )

    @app.exception_handler(ReplyNotGeneratedError)
    async def handle_reply_not_generated(_request: Request, error: ReplyNotGeneratedError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"code": "REPLY_NOT_GENERATED", "detail": str(error)},
        )
