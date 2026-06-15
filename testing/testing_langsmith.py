"""
Test LangSmith tracing setup.
Run: python -m testing.test_langsmith
"""
from src.monitoring.langsmith_tracer import (
    setup_langsmith,
    is_tracing_enabled,
)
from config.settings import get_settings

settings = get_settings()


def main():

    print("=" * 50)
    print("LangSmith Tracer Test")
    print("=" * 50)

    print(f"\nSettings:")
    print(f"  langsmith_enabled : {settings.langsmith_enabled}")
    print(f"  langsmith_project : {settings.langsmith_project}")
    print(
        f"  langsmith_api_key : "
        f"{'set ✓' if settings.langsmith_api_key else 'missing ❌'}"
    )

    print(f"\nRunning setup_langsmith()...")
    result = setup_langsmith()

    print(f"\nResult      : {'enabled ✓' if result else 'disabled'}")
    print(f"Is active   : {is_tracing_enabled()}")

    if result:
        print("\nTest LangChain call with tracing...")
        from src.rag_chain.chain_pipeline import RAGChain
        chain = RAGChain()
        out   = chain.invoke(
            question="What is the leave policy?",
            department="hr",
        )
        print(f"Answer      : {out['answer'][:100]}...")
        print(
            "\n✅ Check https://smith.langchain.com "
            f"→ project '{settings.langsmith_project}'"
        )
    else:
        print(
            "\n⚠️  Tracing disabled — "
            "set LANGSMITH_ENABLED=true and "
            "LANGSMITH_API_KEY in .env"
        )

    print("=" * 50)


if __name__ == "__main__":
    main()