from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from httpx import AsyncClient, Timeout

from .llm import Settings, analyze_abstract
from .schemas import AnalyzeRequest, ErrorResponse, PaperAnalysis


RESEARCH_QUESTION = "Помогает ли интервальное повторение запоминать термины?"
REQUEST_EXAMPLES = {
    "with_method": {
        "summary": "Аннотация с описанием метода",
        "value": {
            "research_question": RESEARCH_QUESTION,
            "abstract": (
                "Цель исследования — оценить влияние интервального повторения на запоминание терминов. "
                "В эксперименте 60 студентов случайным образом разделили на две группы: "
                "интервального и обычного повторения. "
                "Через неделю группа интервального повторения вспомнила больше терминов."
            ),
            "temperature": 0,
        },
    },
    "without_method": {
        "summary": "Аннотация без описания метода",
        "value": {
            "research_question": RESEARCH_QUESTION,
            "abstract": (
                "Цель исследования — оценить влияние интервального повторения на запоминание терминов. "
                "Авторы сообщают об улучшении запоминания терминов при интервальном повторении."
            ),
            "temperature": 0,
        },
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"

    async with AsyncClient(
        base_url=f"{str(settings.base_url).rstrip('/')}/",
        headers=headers,
        timeout=Timeout(settings.timeout_seconds, connect=5.0),
    ) as client:
        app.state.settings = settings
        app.state.llm_client = client
        yield


app = FastAPI(
    title="AI Research Automation",
    description="Анализ аннотаций научных статей для обзора литературы.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post(
    "/papers/analyze",
    summary="Подготовить карточку статьи",
    responses={
        400: {"description": "Некорректная кодировка тела запроса"},
        502: {"model": ErrorResponse, "description": "Ошибка модели, формата ответа или цитаты"},
        503: {"model": ErrorResponse, "description": "Сервер модели недоступен"},
        504: {"model": ErrorResponse, "description": "Тайм-аут ответа модели"},
    },
)
async def analyze_paper(
    payload: Annotated[AnalyzeRequest, Body(openapi_examples=REQUEST_EXAMPLES)],
    request: Request,
) -> PaperAnalysis:
    """Извлекает сведения из аннотации и оценивает её связь с вопросом исследователя."""
    return await analyze_abstract(payload, request.app.state.llm_client, request.app.state.settings)
