import json
import unittest
from functools import partial
from unittest.mock import patch

import httpx

from app.llm import Settings
from app.main import REQUEST_EXAMPLES, app


class PaperAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.payload = REQUEST_EXAMPLES["with_method"]["value"].copy()
        self.card = {
            "objective": "Оценить влияние интервального повторения на запоминание терминов",
            "method": "Сравнение интервального и обычного повторения в двух группах",
            "findings": [
                {
                    "summary": "Через неделю интервальное повторение дало лучший результат",
                    "quote": "Через неделю группа интервального повторения вспомнила больше терминов.",
                }
            ],
            "relevance": "high",
        }
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.finish_reason = "stop"
        self.content = None
        self.transport_error = None

        settings = Settings(
            _env_file=None, base_url="http://llm.test/v1", model="test-model",
            api_key=None, timeout_seconds=5, seed=42, top_p=1.0,
        )
        make_client = partial(httpx.AsyncClient, transport=httpx.MockTransport(self.respond))
        self.enterContext(patch("app.main.Settings", return_value=settings))
        self.enterContext(patch("app.main.AsyncClient", make_client))
        await self.enterAsyncContext(app.router.lifespan_context(app))
        self.client = await self.enterAsyncContext(
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://app.test")
        )

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.transport_error:
            raise self.transport_error("LLM test failure", request=request)

        content = self.content if self.content is not None else json.dumps(self.card)
        return httpx.Response(
            self.status,
            json={"choices": [{"finish_reason": self.finish_reason, "message": {"content": content}}]},
        )

    async def test_sends_temperature_and_strict_schema(self) -> None:
        self.payload["temperature"] = 0.7
        response = await self.client.post("/papers/analyze", json=self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.card)
        self.assertEqual(self.requests[0].url.path, "/v1/chat/completions")
        sent = json.loads(self.requests[0].content)
        self.assertEqual(sent["temperature"], 0.7)
        self.assertEqual(sent["seed"], 42)
        self.assertEqual(sent["top_p"], 1.0)
        self.assertEqual(json.loads(sent["messages"][1]["content"])["abstract"], self.payload["abstract"])
        self.assertEqual(sent["response_format"]["type"], "json_schema")
        schema_options = sent["response_format"]["json_schema"]
        self.assertIs(schema_options["strict"], True)
        self.assertEqual(set(schema_options["schema"]["required"]), set(self.card))
        self.assertIs(schema_options["schema"]["additionalProperties"], False)

    async def test_method_can_be_null_and_temperature_defaults_to_zero(self) -> None:
        self.payload = REQUEST_EXAMPLES["without_method"]["value"].copy()
        del self.payload["temperature"]
        self.card["method"] = None
        self.card["findings"][0] = {
            "summary": "Интервальное повторение улучшило запоминание терминов",
            "quote": "Авторы сообщают об улучшении запоминания терминов при интервальном повторении.",
        }

        response = await self.client.post("/papers/analyze", json=self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["method"])
        self.assertEqual(json.loads(self.requests[0].content)["temperature"], 0.0)

    async def test_rejects_invalid_input_before_calling_the_model(self) -> None:
        for changes in ({"abstract": " \n "}, {"temperature": 2.1}, {"temperature": "0"}):
            with self.subTest(changes=changes):
                response = await self.client.post("/papers/analyze", json=self.payload | changes)
                self.assertEqual(response.status_code, 422)
        data = json.dumps(self.payload | {"temperature": float("nan")})
        response = await self.client.post("/papers/analyze", content=data, headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.requests)

    async def test_rejects_a_quote_absent_from_the_abstract(self) -> None:
        self.card["findings"][0]["quote"] = "Через две недели группа вспомнила больше терминов."
        response = await self.client.post("/papers/analyze", json=self.payload)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"]["code"], "unverified_quote")

    async def test_rejects_malformed_json_and_invalid_fields(self) -> None:
        for content in ("{", json.dumps(self.card | {"relevance": "excellent"})):
            with self.subTest(content=content):
                self.content = content
                response = await self.client.post("/papers/analyze", json=self.payload)
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json()["detail"]["code"], "invalid_llm_response")

    async def test_rejects_truncated_generation(self) -> None:
        self.finish_reason = "length"
        response = await self.client.post("/papers/analyze", json=self.payload)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"]["code"], "invalid_llm_response")

    async def test_maps_upstream_failures_to_json(self) -> None:
        for error, status, code in (
            (httpx.ConnectError, 503, "llm_unavailable"),
            (httpx.ReadTimeout, 504, "llm_timeout"),
            (None, 502, "llm_http_error"),
        ):
            with self.subTest(code=code):
                self.transport_error = error
                self.status = 404
                response = await self.client.post("/papers/analyze", json=self.payload)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["detail"]["code"], code)
