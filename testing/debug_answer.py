"""
Debug why answer is empty after PII scrubbing.
Run: python -m testing.debug_answer
"""
from src.embedding_layer.embedding_service import get_embedding_service
from src.vectordb.qdrant_store import QdrantStore
from src.retrieval.retriever_service import RetrieverService
from src.prompts_layer.prompts import get_rag_prompt
from src.llm_layer.llm_connecter import LLMConnector
from src.pil_guardrils.pil_guard import PIIGuardrail
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_settings

settings = get_settings()


def main():

    print("=" * 55)
    print("DEBUG — RAG Pipeline Step by Step")
    print("=" * 55)

    question   = "What is the leave policy?"
    department = "general"

    # ── Init ──────────────────────────────────────────────────────────
    embedder  = get_embedding_service()
    store     = QdrantStore(
        embedding_service=embedder,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )
    retriever = RetrieverService(store=store, default_k=5)
    llm       = LLMConnector().get_llm()
    parser    = StrOutputParser()
    pii       = PIIGuardrail()

    # ── Step 1: PII scrub input ────────────────────────────────────────
    print("\nStep 1 — PII scrub input")
    pii_result = pii.scrub_input(question)
    print(f"  original    : {question}")
    print(f"  clean       : {pii_result.clean_text}")
    print(f"  was_scrubbed: {pii_result.was_scrubbed}")
    print(f"  pii_found   : {pii_result.pii_found}")

    clean_question = pii_result.clean_text

    # ── Step 2: Retrieve ───────────────────────────────────────────────
    print("\nStep 2 — Retrieve")
    docs = retriever.retrieve(
        question=clean_question,
        department=department,
        k=5,
    )
    print(f"  docs found  : {len(docs)}")
    for i, doc in enumerate(docs, 1):
        print(
            f"  [{i}] score={doc.metadata.get('score', 0):.4f} | "
            f"file={doc.metadata.get('filename')}"
        )
        print(f"       {doc.page_content[:100]}...")

    # ── Step 3: Format context ─────────────────────────────────────────
    print("\nStep 3 — Format context")
    context = RetrieverService.format_context(docs)
    print(f"  context length: {len(context)} chars")
    print(f"  preview:\n{context[:300]}...")

    # ── Step 4: LLM generation ─────────────────────────────────────────
    print("\nStep 4 — LLM generation")
    prompt = get_rag_prompt(department)
    chain  = prompt | llm | parser

    raw_answer = chain.invoke({
        "context":  context,
        "question": clean_question,
    })

    print(f"  type        : {type(raw_answer)}")
    print(f"  raw answer  : {str(raw_answer)[:500]}")

    # ── Step 5: PII scrub output ───────────────────────────────────────
    print("\nStep 5 — PII scrub output")
    pii_detected = pii.detect_pii(str(raw_answer))
    print(f"  PII detected in answer: {pii_detected}")

    clean_answer = pii.scrub_output(str(raw_answer))
    print(f"  clean answer: {clean_answer[:500]}")

    print("\n" + "=" * 55)
    print("FINAL ANSWER:")
    print("=" * 55)
    print(clean_answer)
    print("=" * 55)


if __name__ == "__main__":
    main()