"""Tests for the hybrid search pipeline: BM25, RRF fusion, and the hybrid_search tool."""

import pytest
import pytest_asyncio

import mcp_vectordb.services as services
from mcp_vectordb.adapters.chroma import ChromaAdapter
from mcp_vectordb.config.config import VectorDBConfig, get_settings
from mcp_vectordb.core.bm25 import BM25Index, tokenize
from mcp_vectordb.core.document import Document
from mcp_vectordb.core.fusion import reciprocal_rank_fusion
from mcp_vectordb.tools import search as search_tool


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

CORPUS = [
    ("1", "The quick brown fox jumps over the lazy dog"),
    ("2", "Vector databases store embeddings for semantic search"),
    ("3", "BM25 is a keyword ranking function used in search engines"),
    ("4", "The dog barked at the fox in the yard"),
]


def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert tokenize("Hello, World! BM25-test") == ["hello", "world", "bm25", "test"]


def test_tokenize_simple():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_bm25_ranks_documents_with_more_query_term_overlap_higher():
    index = BM25Index(CORPUS)
    results = index.search("fox dog", top_k=10)
    result_ids = [doc_id for doc_id, _ in results]

    # Docs 1 and 4 both mention fox and dog; docs 2 and 3 mention neither.
    assert set(result_ids) == {"1", "4"}
    assert all(score > 0 for _, score in results)


def test_bm25_excludes_documents_with_no_term_overlap():
    index = BM25Index(CORPUS)
    results = index.search("vector embeddings", top_k=10)
    result_ids = [doc_id for doc_id, _ in results]
    assert result_ids == ["2"]


def test_bm25_respects_top_k():
    index = BM25Index(CORPUS)
    results = index.search("the", top_k=1)
    assert len(results) == 1


def test_bm25_empty_query_returns_no_results():
    index = BM25Index(CORPUS)
    assert index.search("???", top_k=10) == []


def test_bm25_empty_corpus_returns_no_results():
    index = BM25Index([])
    assert index.search("fox", top_k=10) == []


def test_bm25_unknown_terms_return_no_results():
    index = BM25Index(CORPUS)
    assert index.search("xyloquartz", top_k=10) == []


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

def test_rrf_promotes_doc_ranked_high_in_both_lists():
    vector_ranking = ["a", "b", "c"]
    bm25_ranking = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
    fused_ids = [doc_id for doc_id, _ in fused]

    # "a" and "b" each appear near the top of both lists, so they should
    # outrank "c" and "d", which only appear in one list.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert fused_ids[2:] == ["c", "d"] or fused_ids[2:] == ["d", "c"]


def test_rrf_is_sorted_descending_by_score():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_weights_bias_toward_the_heavier_leg():
    vector_ranking = ["a", "b"]
    bm25_ranking = ["b", "a"]

    fused_vector_heavy = dict(reciprocal_rank_fusion([vector_ranking, bm25_ranking], weights=[10.0, 1.0]))
    fused_bm25_heavy = dict(reciprocal_rank_fusion([vector_ranking, bm25_ranking], weights=[1.0, 10.0]))

    assert fused_vector_heavy["a"] > fused_vector_heavy["b"]
    assert fused_bm25_heavy["b"] > fused_bm25_heavy["a"]


def test_rrf_rejects_mismatched_weights_length():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert {doc_id for doc_id, _ in fused} == {"a", "b"}


# ---------------------------------------------------------------------------
# hybrid_search tool (integration, ChromaAdapter-backed)
# ---------------------------------------------------------------------------

class _FakeEmbeddingService:
    """Deterministic fake embedding: always returns a fixed vector."""

    def __init__(self, vector=(1.0, 0.0)):
        self._vector = list(vector)

    async def generate_embedding(self, text):
        return self._vector


@pytest_asyncio.fixture
async def hybrid_env(tmp_path, monkeypatch):
    config = VectorDBConfig(path=str(tmp_path / "chroma_db"))
    db = ChromaAdapter(config)
    await db.initialize()
    await db.create_collection("documents", dimension=2)

    documents = [
        Document(id="1", text="The quick brown fox jumps over the lazy dog", embedding=[1.0, 0.0]),
        Document(id="2", text="Vector databases store embeddings for semantic search", embedding=[0.0, 1.0]),
        Document(id="3", text="BM25 is a keyword ranking function used in search engines", embedding=[0.7, 0.7]),
        Document(id="4", text="The dog barked at the fox in the yard", embedding=[0.9, 0.1]),
    ]
    await db.store_documents(documents, "documents")

    monkeypatch.setattr(services, "vector_db", db)
    monkeypatch.setattr(services, "embedding_service", _FakeEmbeddingService())
    search_tool._bm25_cache.clear()

    yield db

    await db.close()


DOC_TEXT = {
    "1": "The quick brown fox jumps over the lazy dog",
    "2": "Vector databases store embeddings for semantic search",
    "3": "BM25 is a keyword ranking function used in search engines",
    "4": "The dog barked at the fox in the yard",
}


@pytest.mark.asyncio
async def test_hybrid_search_returns_fused_results(hybrid_env, monkeypatch):
    monkeypatch.setattr(get_settings().search, "default_top_k", 3)
    result = await search_tool.hybrid_search(query="fox dog", collection="documents")

    assert isinstance(result, list)
    assert DOC_TEXT["1"] in result
    assert DOC_TEXT["4"] in result
    # Doc 3 shares no keyword overlap and the query embedding points away from it,
    # so it should rank behind 1 and 4.
    pos_1 = result.index(DOC_TEXT["1"])
    pos_4 = result.index(DOC_TEXT["4"])
    pos_3 = result.index(DOC_TEXT["3"])
    assert pos_1 < pos_3
    assert pos_4 < pos_3


@pytest.mark.asyncio
async def test_hybrid_search_respects_top_k(hybrid_env, monkeypatch):
    monkeypatch.setattr(get_settings().search, "default_top_k", 2)
    result = await search_tool.hybrid_search(query="the", collection="documents")
    assert len(result) == 2


@pytest.mark.asyncio
async def test_hybrid_search_missing_collection(hybrid_env):
    result = await search_tool.hybrid_search(query="fox", collection="does_not_exist")
    assert result == []


@pytest.mark.asyncio
async def test_hybrid_search_no_results_for_unmatched_query(hybrid_env):
    # An orthogonal embedding and a keyword that matches nothing should
    # still surface vector-search neighbours, so use a completely disjoint
    # embedding space via filters to force an empty candidate set instead.
    result = await search_tool.hybrid_search(
        query="fox",
        collection="documents",
        filters={"nonexistent_key": "nonexistent_value"},
    )
    assert result == []


@pytest.mark.asyncio
async def test_hybrid_search_min_score_filters_low_vector_similarity(hybrid_env, monkeypatch):
    # Doc 2's embedding [0.0, 1.0] is orthogonal to the query embedding [1.0, 0.0],
    # so its vector similarity is near zero and should be filtered out.
    monkeypatch.setattr(get_settings().search, "default_min_score", 0.5)
    monkeypatch.setattr(get_settings().search, "default_top_k", 4)
    result = await search_tool.hybrid_search(query="vector databases", collection="documents")
    assert DOC_TEXT["2"] not in result


@pytest.mark.asyncio
async def test_hybrid_search_uses_reranker_when_requested(hybrid_env, monkeypatch):
    async def _fake_rerank(query, candidates, model_name=None):
        # Score by ascending position in the fused order, so sorting
        # descending by score reverses whatever order fusion produced.
        doc_ids = [doc_id for doc_id, _ in candidates]
        return [(doc_id, float(i)) for i, doc_id in enumerate(doc_ids)]

    monkeypatch.setattr(search_tool, "cross_encoder_rerank", _fake_rerank)
    monkeypatch.setattr(get_settings().search, "default_top_k", 4)

    baseline = await search_tool.hybrid_search(query="fox dog", collection="documents")

    monkeypatch.setattr(get_settings().search, "use_reranker", True)
    reranked = await search_tool.hybrid_search(query="fox dog", collection="documents")

    assert reranked == list(reversed(baseline))


@pytest.mark.asyncio
async def test_bm25_index_is_cached_per_collection(hybrid_env):
    index_first = await search_tool._get_bm25_index(hybrid_env, "documents")
    index_second = await search_tool._get_bm25_index(hybrid_env, "documents")
    assert index_first is index_second


@pytest.mark.asyncio
async def test_hybrid_search_filters_apply_to_bm25_leg_too(hybrid_env):
    # "fox" matches docs 1 and 4 by keyword, but neither carries this
    # metadata. If BM25 ignored the filter it would leak them back in.
    result = await search_tool.hybrid_search(
        query="fox",
        collection="documents",
        filters={"nonexistent_key": "nonexistent_value"},
    )
    assert result == []


@pytest.mark.asyncio
async def test_bm25_index_rebuilds_after_document_count_changes(hybrid_env):
    index_before = await search_tool._get_bm25_index(hybrid_env, "documents")

    await hybrid_env.store_documents(
        [Document(id="5", text="A brand new document about penguins", embedding=[0.1, 0.9])],
        "documents",
    )

    index_after = await search_tool._get_bm25_index(hybrid_env, "documents")
    assert index_after is not index_before
    assert "5" in index_after.doc_ids
