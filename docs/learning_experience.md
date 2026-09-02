# Learning Experience with Model Context Protocol (MCP)

I started encountering the term **MCP (Model Context Protocol)** quite frequently. Initially, when I searched for it using ChatGPT or Gemini, I received unrelated results until I explicitly specified "Model Context Protocol."

Once I identified a potential use case for MCP in my project, I began noticing its presence in various tools across my work environment — **Slack, Confluence, GitHub**, and others. In some cases, even the software vendors provide their own MCP implementations, which may be more secure and reliable than developing a custom one. It’s often better to leverage these existing, secure versions when possible.

---

## Why I Needed MCP in My Project

In one of my projects, I was developing a **Learning AI Agent**, and I wanted to implement **context engineering** — a technique that involves passing the user's conversation history to the **LLM (Large Language Model)** to provide additional context, enhancing the relevance and accuracy of responses.

While a well-structured prompt is essential, incorporating relevant historical context can significantly improve outcomes.

However, providing the **entire conversation history** can hit the **context window limit** of the model (even though some newer models offer larger windows). This also increases **token usage and costs**. To optimize both performance and cost, I aimed to pass only the **minimal relevant context** to the LLM.

That’s where **Retrieval-Augmented Generation (RAG)** comes into play. Rather than manually coding this context-passing logic for every application, I wanted to abstract it into a **reusable protocol (MCP)** which would allow different tools to plug into the same context framework.

With the latest models supporting **tool calling**, there's no need to manually invoke functions anymore. For example, **OpenAI’s Response API** can automatically invoke tools based on the user's query if we pass the tools with `mcp` as type along with the server host details.

---

## How RAG Works

To understand RAG’s effectiveness, we need to look at its underlying process:

1. **Input Text or Document**
2. **Chunking** – Split text into manageable parts.
3. **Embedding** – Each chunk is embedded using a model.
4. **Storage** – Store these embeddings in a **vector database**.
5. **Querying** – Convert user query into an embedding and use **similarity search** to retrieve the top-k relevant chunks.

> A poor chunking strategy can lead to **loss of context** or **misleading results**. Therefore, it’s essential to define chunks carefully — often by embedding some surrounding context within each chunk to help guide the LLM.

If the documents are too large or complex to manually add context, we can use an LLM to help **identify and extract appropriate context** before embedding and storing it.

Some organizations also use classic techniques such as:

- **TF-IDF**
  - **Term Frequency (TF):** How often a word appears in a document.
  - **Inverse Document Frequency (IDF):** How rare that word is across all documents.
- **BM25** – A more advanced version of TF-IDF used in modern search systems.

**More details here:** [Anthropic – Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)

---

## Practical Note

If your application’s main purpose is to retrieve answers from a document, and the document fits within the model's **context window**, you can **skip the embedding process entirely**. Just pass the full document directly to the model.

This approach reduces cost, especially since the model may **cache** the content efficiently. You can also use these documents as **resources** in MCP, especially if there are only a few. (This varies per use case, but it's a viable idea.)

---

## Clarifying MCP’s Role

Initially, I misunderstood MCP as being only about tool integration. In reality, **MCP supports**:

- **Prompts**
- **Resources** (read-only data)
- **Interactions with LLMs**
- **Tool invocation**

MCP supports both **stdio** and **streamable HTTP**. I’ve personally used the **Python Model Context Protocol SDK** to implement it.

---

## Retrieval Tuning: top-k, Similarity Scores, and Metadata

While building out search on top of the vector DB, I picked up a few practical lessons about tuning retrieval quality:

- **Don't limit `top_k` too aggressively** – it should be a moderate/mid value, not a very small number. Too small a `top_k` can silently drop relevant chunks before any scoring or filtering even gets a chance to work with them.
- **`score` is a similarity score, not a relevance guarantee** – so the **minimum score threshold shouldn't be set too high**. An overly strict threshold filters out chunks that are still contextually useful just because they didn't hit a high similarity value.
- **Metadata is critical, not optional** – it's what makes **filtering** possible (by source, section, date, type, etc.) on top of pure vector similarity. Good metadata design early on pays off heavily during retrieval.

**Cosine vs. Euclidean distance:** both are common distance metrics for comparing embeddings, but they answer slightly different questions.
- **Cosine similarity** measures the *angle* between two vectors, ignoring their magnitude — it captures whether two chunks point in the same "semantic direction," regardless of how long the vectors are.
- **Euclidean distance** measures the *straight-line distance* between two vectors, so it's sensitive to magnitude as well as direction.

For this project I **selected cosine similarity** for search, since embedding magnitude isn't meaningful for relevance here — what matters is semantic direction/orientation, not vector length. Cosine tends to be the more standard choice for text embedding search for this reason.

---

## Embedding Beyond Plain Text

- **Images are not embedded the same way as text.** Normal text embedding models don't handle images — they need a separate multimodal/image embedding approach (or a captioning step before embedding as text). In practice, images get **sent as base64-encoded data to a multimodal model**, rather than run through a text-embedding pipeline directly.
- **Structural embedding matters for structured documents:**
  - For **Markdown**, structure (headings, sections) can and should be preserved during chunking/embedding rather than treating it as flat text.
  - For **PDFs**, page structure matters too. Tools like **Unstructured** use a **model behind the scenes to partition** the document when the strategy is set to `hi_res` (or a `hi_res_model_name` is specified) — this gives better structural boundaries than naive text extraction, especially for documents with tables, columns, or mixed layouts.

---

## Agentic Embedding: When Format Isn't Uniform

While embedding **"World Power Made Easy"**, a book structured as a series of question-and-answer pairs, I ran into a format problem rather than a content problem: the question/answer structure wasn't consistent across the book — some entries were short one-liners, others spanned multiple paragraphs, and the boundary between "question" and "answer" wasn't always textually obvious.

A single static chunking rule (fixed size, or splitting on headings/titles) doesn't hold up well against that kind of variability — it either splits a question away from its answer or lumps unrelated Q&A pairs into one chunk. That's what surfaced the need for **agentic embedding**: instead of applying one fixed chunking strategy uniformly, use an LLM in the loop to reason about each section's actual structure and decide chunk boundaries (and any context to carry along) dynamically, per section, rather than statically.

---

## Performance Lesson: hi_res Cost and Per-Page Model Reloads

When generating embeddings for a **500+ page PDF** using the `hi_res` partitioning strategy for every page, the job took **over 8 hours locally**. Two separate cost problems were compounding:

- **`hi_res` runs a layout-detection model on every page**, which is expensive — and that cost is wasted on pages that are plain text with no images or tables, where the cheaper `fast` strategy extracts the same information just as well.
  - **Fix:** classify each page first (does it contain images or drawings?) and only route pages that actually need it through `hi_res`; everything else uses `fast`. See `_is_complex_pdf_page` and its use in `_process_pdf_range` in `src/mcp_vectordb/chunking/process_document.py`.
- **Calling `partition_pdf` once per single page reloads the hi_res layout model each time** — the model's load/init cost dominated total runtime even for a small number of pages, independent of the actual page content size.
  - **Fix:** group consecutive pages by strategy into contiguous runs first, then call `partition_pdf` once per run instead of once per page, so the hi_res model loads once per run rather than once per page.

**Takeaway:** when a step in a pipeline has a heavy one-time cost (model load, connection setup, etc.), batch by that cost profile — group work so the expensive step happens once per batch, not once per unit of work.

---

## Docling: Formula Handling, Images, and a Latency Comparison with Unstructured

While evaluating **Docling** as an alternative document-parsing pipeline, a few practical gaps showed up:

- **Formula inference is opt-in and costly** – Docling doesn't include mathematical formulas in the chunk output by default. Turning on formula inference does recover them, but it noticeably **increases processing latency**, so it's a tradeoff to enable only when formulas are actually needed downstream.
- **Images aren't returned within the chunk either** – similar to formulas, image content isn't embedded into the chunk by default. Extracting images requires a **separate processing step** after the initial document conversion, rather than being available inline.
- **Latency vs. Unstructured** – when compared against **Unstructured**, Docling's overall processing latency was higher on the same documents. This matters for pipelines where document-processing time is on the critical path (e.g., ingestion-time chunking before embedding).

**Takeaway:** Docling's default output favors plain text extraction — formulas and images both need to be explicitly opted into or processed separately, and that comes at a latency cost that should be weighed against Unstructured's `hi_res`/`fast` tradeoff already noted above.

---

## Fusion & Approximate Search Concepts

When combining results from multiple retrieval methods (e.g., dense vector search + keyword/BM25 search), a few fusion concepts came up:

- **Reciprocal Rank Fusion (RRF)** and its variant **Weighted Reciprocal Rank Fusion** – ways to merge ranked result lists from different retrievers into a single combined ranking.
- **RRF only needs the *position/rank* of a document in each result list** – it doesn't care about the underlying similarity scores at all, which makes it useful for combining results from retrievers whose scores aren't directly comparable (e.g., BM25 score vs. cosine similarity).
- **RRF uses a damping constant `k`** in the formula, roughly: `score = 1/(k + rank_1) + 1/(k + rank_2) + ...` summed across each retriever's rank for that document. The constant `k` softens the impact of rank differences (especially for lower-ranked, less certain results).

On the vector index side:

- **ANN (Approximate Nearest Neighbour)** search is what makes vector search scale — instead of an exact brute-force comparison against every vector, it trades a small amount of accuracy for large gains in speed. Concepts like **graph-based navigation (e.g., HNSW-style "highway" long-range links) and neighborhood exploration** are what let ANN algorithms jump close to the right region of the vector space quickly instead of scanning everything.

---

## Final Thoughts

This learning journey helped me appreciate how powerful and flexible **MCP** can be for building **context-aware AI systems**, especially when combined with **RAG** and **modern tool-calling capabilities**. The retrieval-tuning and fusion concepts above are the next layer down — they're what actually determine whether a RAG system returns *useful* context or just *technically similar* context.
