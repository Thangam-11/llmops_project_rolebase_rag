# tests/unit/test_department_prompts.py
import pytest

from src.prompts_layer.prompts import (
    SYSTEM_PROMPTS,
    get_rag_prompt,
    get_system_prompt_str,
)


ALL_DEPARTMENTS = ["finance", "marketing", "hr", "engineering", "c_level", "general"]


# ---------------------------------------------------------------------------
# 1. Each real department has its own distinct prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_department_exists_in_system_prompts(department):
    assert department in SYSTEM_PROMPTS


def test_all_department_prompts_are_distinct():
    prompts = list(SYSTEM_PROMPTS.values())
    assert len(prompts) == len(set(prompts))   # no accidental duplicates/copy-paste


@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_get_system_prompt_str_returns_correct_department_prompt(department):
    result = get_system_prompt_str(department)
    assert result == SYSTEM_PROMPTS[department]


# ---------------------------------------------------------------------------
# 2. Unknown department falls back to "general" — the security-relevant case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_department", [
    "unknown_dept",
    "Finance",          # wrong case — dict lookup is case-sensitive, must NOT match "finance"
    "hr_data",          # a folder name from data_loader, not a department — must not leak through
    "",
    "admin",
    "c-level",          # typo/variant of "c_level"
])
def test_unknown_department_falls_back_to_general(bad_department):
    result = get_system_prompt_str(bad_department)
    assert result == SYSTEM_PROMPTS["general"]


@pytest.mark.parametrize("bad_department", ["unknown_dept", "", "Finance"])
def test_get_rag_prompt_falls_back_to_general_for_unknown_department(bad_department):
    template = get_rag_prompt(bad_department)
    system_message = template.messages[0].prompt.template
    assert system_message == SYSTEM_PROMPTS["general"]


def test_unknown_department_never_falls_back_to_a_privileged_prompt():
    """
    Explicitly confirms the fallback is 'general' and not any of the more
    sensitive department prompts — this is the core access-control guarantee.
    """
    result = get_system_prompt_str("does_not_exist")
    assert result != SYSTEM_PROMPTS["finance"]
    assert result != SYSTEM_PROMPTS["hr"]
    assert result != SYSTEM_PROMPTS["c_level"]
    assert result != SYSTEM_PROMPTS["engineering"]
    assert result != SYSTEM_PROMPTS["marketing"]
    assert result == SYSTEM_PROMPTS["general"]


# ---------------------------------------------------------------------------
# 3. "general" prompt's confidentiality instruction is actually present
# ---------------------------------------------------------------------------

def test_general_prompt_restricts_sensitive_data():
    prompt = SYSTEM_PROMPTS["general"]
    assert "Do NOT reveal financial, HR, or technical data" in prompt


def test_hr_prompt_mentions_confidentiality():
    prompt = SYSTEM_PROMPTS["hr"]
    assert "confidentiality" in prompt.lower()


@pytest.mark.parametrize("department", ["finance", "marketing", "hr", "engineering", "general"])
def test_non_c_level_prompts_require_answering_only_from_context(department):
    prompt = SYSTEM_PROMPTS[department]
    assert "Answer ONLY using the provided context" in prompt


@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_every_prompt_has_the_no_context_fallback_phrase(department):
    """
    Ensures every department, including future additions, tells the model
    what to say when the answer isn't in context — this is what prevents
    hallucination when retrieval comes back empty or irrelevant.
    """
    prompt = SYSTEM_PROMPTS[department]
    assert "I don't have that information." in prompt


@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_every_prompt_requires_citation(department):
    prompt = SYSTEM_PROMPTS[department]
    assert "Source" in prompt


def test_c_level_prompt_uses_dept_qualified_citation_format():
    """
    c_level is the only role expected to see cross-department data,
    so it needs dept/filename citations, not just filename.
    """
    prompt = SYSTEM_PROMPTS["c_level"]
    assert "(Source: dept/filename)" in prompt


@pytest.mark.parametrize("department", ["finance", "marketing", "hr", "engineering", "general"])
def test_non_c_level_prompts_use_filename_only_citation_format(department):
    prompt = SYSTEM_PROMPTS[department]
    assert "(Source: filename)" in prompt


# ---------------------------------------------------------------------------
# 4. get_rag_prompt() — template structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_get_rag_prompt_returns_chat_prompt_template(department):
    from langchain_core.prompts import ChatPromptTemplate
    result = get_rag_prompt(department)
    assert isinstance(result, ChatPromptTemplate)


@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_get_rag_prompt_has_system_and_human_messages(department):
    template = get_rag_prompt(department)
    assert len(template.messages) == 2


@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_get_rag_prompt_system_message_matches_department(department):
    template = get_rag_prompt(department)
    system_message = template.messages[0].prompt.template
    assert system_message == SYSTEM_PROMPTS[department]


@pytest.mark.parametrize("department", ALL_DEPARTMENTS)
def test_get_rag_prompt_input_variables_are_context_and_question(department):
    template = get_rag_prompt(department)
    assert set(template.input_variables) == {"context", "question"}


# ---------------------------------------------------------------------------
# 5. Actual formatting works end-to-end
# ---------------------------------------------------------------------------

def test_get_rag_prompt_formats_with_context_and_question():
    template = get_rag_prompt("finance")

    messages = template.format_messages(
        context="Q3 revenue was $5M.",
        question="What was Q3 revenue?",
    )

    human_message_content = messages[1].content
    assert "Q3 revenue was $5M." in human_message_content
    assert "What was Q3 revenue?" in human_message_content


def test_get_rag_prompt_formatting_fails_without_required_variables():
    template = get_rag_prompt("finance")

    with pytest.raises(KeyError):
        template.format_messages(context="only context, missing question")


def test_get_rag_prompt_human_template_asks_to_cite_sources():
    template = get_rag_prompt("hr")
    human_template = template.messages[1].prompt.template
    assert "cite sources" in human_template.lower()