"""Ragas context_precision, scored by an OpenAI judge against the reference answer."""

from typing import List


async def score_context_precision(
    question: str, retrieved_texts: List[str], reference: str, judge_llm
) -> float:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import LLMContextPrecisionWithReference

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=retrieved_texts,
        reference=reference,
    )
    metric = LLMContextPrecisionWithReference(llm=judge_llm)
    return await metric.single_turn_ascore(sample)
