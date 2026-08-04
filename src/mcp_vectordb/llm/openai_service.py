import logging
from typing import Optional

from openai import AsyncOpenAI

from .base import LLMService
from ..utils.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class OpenAILLMService(LLMService):
    """LLM completion service backed by the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

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
