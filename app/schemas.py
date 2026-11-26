from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints


def require_nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Строка должна содержать текст.")
    return value


Text = Annotated[str, StringConstraints(min_length=1), AfterValidator(require_nonblank)]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    research_question: Text = Field(max_length=500, description="Исследовательский вопрос.")
    abstract: Text = Field(max_length=5000, description="Текст аннотации одной статьи.")
    temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        allow_inf_nan=False,
        description="Температура генерации. Передаётся модели без изменений.",
    )


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: Text = Field(description="Краткое изложение одного результата.")
    quote: Text = Field(description="Дословный фрагмент аннотации, подтверждающий результат.")


class PaperAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    objective: Text | None = Field(description="Цель исследования или null, если она не указана.")
    method: Text | None = Field(description="Метод исследования или null, если он не описан.")
    findings: list[Finding] = Field(max_length=3, description="До трёх результатов с цитатами.")
    relevance: Literal["high", "medium", "low"] = Field(
        description="Прямая, частичная или слабая связь аннотации с вопросом.",
    )


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail
