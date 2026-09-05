"""OpenAI-backed LLM judge for Ragas retrieval metrics.

Wraps `langchain_openai.ChatOpenAI` in Ragas' `LangchainLLMWrapper` so
`ragas.metrics.LLMContextPrecisionWithReference`/`LLMContextRecall` can score
hybrid_search's retrieved contexts against each golden question's reference
answer. Requires `OPENAI_API_KEY` in the environment.
"""

import os

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def get_judge_llm(model: str = DEFAULT_JUDGE_MODEL) -> LangchainLLMWrapper:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The Ragas judge metrics call OpenAI directly "
            "and need it in the environment (e.g. via .env)."
        )
    return LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0))
