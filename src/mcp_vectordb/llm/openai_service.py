import logging
from typing import Optional

from openai import AsyncOpenAI

from .base import LLMService
from ..utils.exceptions import LLMServiceError

try:
    from langsmith.wrappers import wrap_openai
except ImportError:
    wrap_openai = None

logger = logging.getLogger(__name__)


class OpenAILLMService(LLMService):
    """LLM completion service backed by the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        client = AsyncOpenAI(api_key=api_key)
        # wrap_openai instruments the client for LangSmith tracing; actual
        # trace export is gated by the LANGSMITH_TRACING/LANGCHAIN_TRACING_V2
        # env vars, so wrapping is a no-op unless those are set.
        self._client = wrap_openai(client) if wrap_openai is not None else client

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise LLMServiceError(f"LLM completion failed: {exc}") from exc

        content = response.choices[0].message.content
        logger.info(
            "OpenAI completion model=%r usage=%r response=%r",
            self.model, response.usage, content,
        )
        return content
