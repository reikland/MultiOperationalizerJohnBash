from datetime import date

from forecast_brief.utils import (
    add_months,
    all_blocks_have_required_fields,
    ensure_online_model,
    split_clusters,
    split_question_blocks,
)


def test_add_months_handles_end_of_month():
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)


def test_ensure_online_model_appends_suffix_once():
    assert ensure_online_model("openai/gpt") == "openai/gpt:online"
    assert ensure_online_model("openai/gpt:online") == "openai/gpt:online"


def test_split_clusters_extracts_blocks():
    text = """=== TIER 3: CLUSTERS ===
CLUSTER: A
SCOPE:\n- includes: a

CLUSTER: B
SCOPE:\n- includes: b
END TIER 3"""
    assert [name for name, _ in split_clusters(text)] == ["A", "B"]


def test_question_block_validation():
    question = """QUESTION FOR CLUSTER: A
TITLE:
TYPE: BINARY
QUESTION (one sentence):
BACKGROUND (shared baseline, 3-6 sentences):
CLUSTER CONTEXT (2-4 sentences):
FERMI STARTING POINT (rough prior + rationale):
WHAT TO CONSIDER (5-8):
- x
ANSWER GUIDANCE (units / interpretation / MC options):
AMBIGUITY CHECKS (3-5):
- x"""
    assert len(split_question_blocks(question)) == 1
    assert all_blocks_have_required_fields(question)
